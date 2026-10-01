# SPDX-License-Identifier: Apache-2.0
"""Confirm packed experiment winners across dtype, row strides, timing order."""

import argparse
import json
import statistics
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
from benchmark_prism import PROJECTIONS
from confirm_pq2_int_tiles import complete as original_complete
from triton.testing import do_bench_cudagraph

from vllm_gguf_plugin.triton.pq2_packed_experiment import packed_gemv, prepare_pq2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sweep = json.loads(args.sweep.read_text())["records"]
    records = []
    torch.manual_seed(101)
    for projection in ("ffn_gate_up", "ffn_down"):
        n, k = PROJECTIONS[projection]
        for m in (1, 4, 8):
            choices = [
                r for r in sweep if r["projection"] == projection and r["m"] == m
            ]
            winner = min(
                choices,
                key=lambda r: statistics.mean((r["full_ms"], r["full_reverse_ms"])),
            )
            tile = winner["config"]["tile"]
            for dtype in (torch.bfloat16, torch.float16):
                for layout in ("contiguous", "strided"):
                    storage = torch.randint(
                        0,
                        256,
                        (n * (2 if layout == "strided" else 1), k // 128 * 34),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    weight = storage[::2] if layout == "strided" else storage
                    blocks = weight.view(n, k // 128, 34)
                    ws = (
                        torch.rand(n, k // 128, dtype=torch.float16, device="cuda")
                        * 0.02
                    )
                    blocks[:, :, :2] = ws.view(torch.uint8).reshape(n, k // 128, 2)
                    x = torch.randn(m, k, dtype=dtype, device="cuda")
                    raw = prepare_pq2(weight)
                    soa = prepare_pq2(weight, separate=True)
                    formats = {
                        "raw1": raw,
                        "raw2": prepare_pq2(weight, load_bytes=2),
                        "soa1": soa,
                        "soa4": replace(
                            soa, codes=soa.codes.view(torch.int32), load_bytes=4
                        ),
                    }
                    methods = {
                        "original_tuned": partial(original_complete, x, weight, tile),
                        "tuned_byte": partial(packed_gemv, x, raw, tile=tile),
                        "raw_spread": partial(
                            packed_gemv, x, raw, tile=tile, fast_pack=True
                        ),
                        "winner": partial(
                            packed_gemv,
                            x,
                            formats[winner["format"]],
                            **winner["config"],
                        ),
                    }
                    expected = methods["tuned_byte"]()
                    metrics = {}
                    for name, run in methods.items():
                        actual = run()
                        torch.testing.assert_close(
                            actual, expected, atol=0.02, rtol=0.002
                        )
                        metrics[name] = dict(
                            different_outputs=(actual != expected).sum().item(),
                            relative_rms=(
                                (actual.double() - expected.double()).square().sum()
                                / expected.double().square().sum()
                            )
                            .sqrt()
                            .item(),
                        )
                    for repeat, order in enumerate(
                        (list(methods), list(reversed(methods))), 1
                    ):
                        timings = {
                            name: do_bench_cudagraph(methods[name], rep=150)
                            for name in order
                        }
                        row = dict(
                            projection=projection,
                            m=m,
                            dtype=str(dtype),
                            layout=layout,
                            winner=winner["variant"],
                            winner_config=winner["config"],
                            repeat=repeat,
                            timings_ms=timings,
                            metrics=metrics,
                        )
                        records.append(row)
                        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                records=records,
                timing=(
                    "Complete call including quantization and allocation; "
                    "preparation excluded; reversed method order"
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
