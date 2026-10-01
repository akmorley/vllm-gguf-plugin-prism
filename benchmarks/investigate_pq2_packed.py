# SPDX-License-Identifier: Apache-2.0
"""Offline packed-load/unpack/scheduling investigation; no serving changes."""

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
import triton
from benchmark_prism import PROJECTIONS
from triton.testing import do_bench_cudagraph

from vllm_gguf_plugin.triton.pq2_int_gemv import pq2_int_gemv
from vllm_gguf_plugin.triton.pq2_mmq import _quantize
from vllm_gguf_plugin.triton.pq2_packed_experiment import (
    launch_packed,
    packed_gemv,
    prepare_pq2,
)

VARIANTS = [
    ("byte", "raw1", False, 1, 1, ""),
    ("byte_spread", "raw1", True, 1, 1, ""),
    ("half", "raw2", False, 1, 1, ""),
    ("half_spread", "raw2", True, 1, 1, ""),
    ("soa_byte", "soa1", False, 1, 1, ""),
    ("soa_byte_spread", "soa1", True, 1, 1, ""),
    ("soa_word", "soa4", False, 1, 1, ""),
    ("soa_word_spread", "soa4", True, 1, 1, ""),
    ("byte_pipeline", "raw1", True, 2, 2, ""),
    ("half_pipeline", "raw2", True, 2, 2, ""),
    ("soa_word_pipeline", "soa4", True, 2, 2, ""),
    ("half_cg", "raw2", True, 1, 1, ".cg"),
    ("soa_word_cg", "soa4", True, 1, 1, ".cg"),
    ("half_keep", "raw2", True, 1, 1, ""),
    ("soa_half_keep", "soa2", True, 1, 1, ""),
    ("soa_word_keep", "soa4", True, 1, 1, ""),
]


def disassemble(kernel, output, prefix):
    cubin = output / (prefix + ".cubin")
    cubin.write_bytes(kernel.asm["cubin"])
    result = subprocess.run(
        ["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-sass", str(cubin)],
        capture_output=True,
        text=True,
        check=True,
    )
    (output / (prefix + ".sass")).write_text(result.stdout)
    opcodes = re.findall(
        r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)", result.stdout
    )
    return dict(Counter(opcodes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=list(PROJECTIONS),
        default=["ffn_gate_up", "ffn_down"],
    )
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--variants", nargs="+", choices=[v[0] for v in VARIANTS])
    args = parser.parse_args()
    variants = [v for v in VARIANTS if not args.variants or v[0] in args.variants]
    if any(m < 1 or m > 16 for m in args.tokens):
        parser.error("tokens must be between 1 and 16")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(97)
    records = []
    preparations = []
    for projection in args.projections:
        n, k = PROJECTIONS[projection]
        w = torch.randint(0, 256, (n, k // 128, 34), dtype=torch.uint8, device="cuda")
        ws = torch.rand(n, k // 128, dtype=torch.float16, device="cuda") * 0.02
        w[:, :, :2] = ws.view(torch.uint8).reshape(n, k // 128, 2)
        w = w.reshape(n, -1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        soa = prepare_pq2(w, separate=True)
        torch.cuda.synchronize()
        preparations.append(
            dict(
                projection=projection,
                first_prepare_ms=(time.perf_counter() - start) * 1000,
                logical_bytes=soa.storage_bytes,
                original_bytes=w.numel(),
                note=(
                    "Original retained during benchmark; separate buffers "
                    "allocate one extra packed copy."
                ),
            )
        )
        formats = {
            "raw1": prepare_pq2(w),
            "raw2": prepare_pq2(w, load_bytes=2),
            "soa1": soa,
            "soa2": replace(soa, codes=soa.codes.view(torch.int16), load_bytes=2),
            "soa4": replace(soa, codes=soa.codes.view(torch.int32), load_bytes=4),
        }
        for m in args.tokens:
            tile = (64, 8, 8) if projection == "ffn_gate_up" and m == 4 else (32, 8, 8)
            if projection == "ffn_down" and m == 8:
                tile = (32, 8, 4)
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            q = torch.empty_like(x, dtype=torch.int8)
            s = torch.empty(m, k // 128, dtype=torch.float32, device="cuda")
            y = torch.empty(m, n, dtype=x.dtype, device="cuda")
            _quantize[(m * (k // 128),)](x, q, s, k)
            reference = packed_gemv(x, formats["raw1"], tile=tile)
            old = pq2_int_gemv(x, w)
            old_ms = do_bench_cudagraph(partial(pq2_int_gemv, x, w), rep=100)
            case = []
            for name, layout, fast, stages, unroll, cache in variants:
                config = dict(
                    tile=tile,
                    fast_pack=fast,
                    stages=stages,
                    unroll=unroll,
                    cache=cache,
                    keep_lanes=name.endswith("_keep"),
                )
                prepared = formats[layout]
                run = partial(launch_packed, q, s, prepared, y, m, **config)
                kernel = run()
                torch.testing.assert_close(y, reference, atol=0.02, rtol=0.002)
                delta = y.double() - reference.double()
                row = dict(
                    projection=projection,
                    m=m,
                    variant=name,
                    format=layout,
                    config=config,
                    registers=kernel.n_regs,
                    spills=kernel.n_spills,
                    shared_bytes=kernel.metadata.shared,
                    prequantized_ms=do_bench_cudagraph(run, rep=100),
                    full_ms=do_bench_cudagraph(
                        partial(packed_gemv, x, prepared, **config), rep=100
                    ),
                    current_serving_tile_ms=old_ms,
                    different_outputs=(y != reference).sum().item(),
                    relative_rms=(
                        delta.square().sum() / reference.double().square().sum()
                    )
                    .sqrt()
                    .item(),
                    original_tile_relative_rms=(
                        (y.double() - old.double()).square().sum()
                        / old.double().square().sum()
                    )
                    .sqrt()
                    .item(),
                )
                if m == 4:
                    row["sass_static_opcodes"] = disassemble(
                        kernel, args.output, projection + "-" + name
                    )
                case.append(row)
                print(
                    {k: v for k, v in row.items() if k != "sass_static_opcodes"},
                    flush=True,
                )
            for row in reversed(case):
                prepared = formats[row["format"]]
                row["full_reverse_ms"] = do_bench_cudagraph(
                    partial(packed_gemv, x, prepared, **row["config"]), rep=100
                )
            records.extend(case)
            (args.output / "results.json").write_text(
                json.dumps(
                    dict(
                        records=records,
                        preparations=preparations,
                        gpu=torch.cuda.get_device_name(),
                        torch=torch.__version__,
                        triton=triton.__version__,
                        timing=(
                            "Warmed CUDA graph replay; "
                            "separate preparation excluded from full call"
                        ),
                        instruction_note=(
                            "Static SASS opcode counts are not "
                            "dynamic hardware counters"
                        ),
                    ),
                    indent=2,
                )
            )
    print("complete", len(records), flush=True)


if __name__ == "__main__":
    main()
