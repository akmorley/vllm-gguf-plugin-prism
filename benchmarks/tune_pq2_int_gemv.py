# SPDX-License-Identifier: Apache-2.0
"""Offline DP4A tile sweep; prequantized timing is diagnostic, not full-call speed."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton
from benchmark_prism import PROJECTIONS
from triton.testing import do_bench_cudagraph

from vllm_gguf_plugin.triton.pq2_int_gemv import _int_gemv
from vllm_gguf_plugin.triton.pq2_mmq import _quantize

TILES = [
    (4, 2, 4),
    (4, 4, 4),
    (4, 8, 4),
    (8, 2, 4),
    (8, 4, 4),
    (8, 8, 4),
    (16, 4, 4),
    (8, 4, 8),
]


def launch(q, s, w, y, m, n, k, bn, bc, warps):
    return _int_gemv[(triton.cdiv(n, bn),)](
        q,
        s,
        w,
        y,
        m,
        n,
        k,
        w.stride(0),
        triton.next_power_of_2(m),
        bn,
        bc,
        num_warps=warps,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4])
    args = parser.parse_args()
    if any(m < 1 or m > 16 for m in args.tokens):
        parser.error("tokens must be between 1 and 16")
    tiles = TILES
    if args.extended:
        tiles = [(16, 4, 4)] + [
            (bn, bc, warps)
            for bn in (8, 16, 32, 64)
            for bc in (4, 8, 16)
            for warps in (4, 8)
            if (bn, bc, warps) != (16, 4, 4)
        ]
    torch.manual_seed(7)
    records = []
    for name in ("ffn_gate_up", "ffn_down"):
        n, k = PROJECTIONS[name]
        w = torch.randint(0, 256, (n, k // 128, 34), dtype=torch.uint8, device="cuda")
        scales = torch.rand(n, k // 128, dtype=torch.float16, device="cuda") * 0.02
        w[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
        w = w.reshape(n, -1)
        for m in args.tokens:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            q = torch.empty_like(x, dtype=torch.int8)
            s = torch.empty(m, k // 128, dtype=torch.float32, device="cuda")
            y = torch.empty(m, n, dtype=x.dtype, device="cuda")
            _quantize[(m * (k // 128),)](x, q, s, k)
            q = q.view(torch.int32)
            reference = None
            for bn, bc, warps in tiles:
                run = partial(launch, q, s, w, y, m, n, k, bn, bc, warps)
                kernel = run()
                if reference is None:
                    reference = y.clone()
                else:
                    torch.testing.assert_close(y, reference, atol=0.02, rtol=0.002)
                record = {
                    "projection": name,
                    "m": m,
                    "bn": bn,
                    "bc": bc,
                    "warps": warps,
                    "latency_ms": do_bench_cudagraph(run, rep=100),
                    "registers": kernel.n_regs,
                    "spills": kernel.n_spills,
                    "shared_bytes": kernel.metadata.shared,
                    "dp4a_ptx": sorted(
                        {
                            line.strip()
                            for line in kernel.asm["ptx"].splitlines()
                            if "dp4a." in line
                        }
                    ),
                }
                records.append(record)
                print(record, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "timing": "prequantized DP4A only; BF16 contiguous synthetic inputs",
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
