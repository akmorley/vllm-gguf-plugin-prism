# SPDX-License-Identifier: Apache-2.0
"""Confirm offline tile winners using complete-call, reversed-order timings."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton
from benchmark_prism import PROJECTIONS
from triton.testing import do_bench_cudagraph
from tune_pq2_int_gemv import launch

from vllm_gguf_plugin.triton.pq2_mmq import _quantize


def complete(x, w, tile):
    m, k = x.shape
    n = w.shape[0]
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    scales = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _quantize[(m * (k // 128),)](x, q, scales, k)
    launch(q.view(torch.int32), scales, w, output, m, n, k, *tile)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sweep = json.loads(args.sweep.read_text())["records"]
    torch.manual_seed(71)
    records = []
    for projection in ("ffn_gate_up", "ffn_down"):
        n, k = PROJECTIONS[projection]
        for m in (1, 4, 8):
            choices = [
                r for r in sweep if r["projection"] == projection and r["m"] == m
            ]
            best = min(choices, key=lambda r: r["latency_ms"])
            tile = (best["bn"], best["bc"], best["warps"])
            for dtype in (torch.bfloat16, torch.float16):
                for layout in ("contiguous", "strided"):
                    storage = torch.randint(
                        0,
                        256,
                        (n * (2 if layout == "strided" else 1), k // 128 * 34),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    w = storage[::2] if layout == "strided" else storage
                    blocks = w.view(n, k // 128, 34)
                    scales = (
                        torch.rand(n, k // 128, dtype=torch.float16, device="cuda")
                        * 0.02
                    )
                    blocks[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
                    x = torch.randn(m, k, dtype=dtype, device="cuda")
                    current = partial(complete, x, w, (16, 4, 4))
                    candidate = partial(complete, x, w, tile)
                    a, b = current(), candidate()
                    torch.testing.assert_close(a, b, atol=0.02, rtol=0.002)
                    delta = b.double() - a.double()
                    error = (
                        (delta.square().sum() / a.double().square().sum()).sqrt().item()
                    )
                    differences = (a != b).sum().item()
                    for repeat, order in enumerate(
                        (
                            (("current", current), ("candidate", candidate)),
                            (("candidate", candidate), ("current", current)),
                        ),
                        1,
                    ):
                        timings = {
                            name: do_bench_cudagraph(run, rep=200)
                            for name, run in order
                        }
                        row = dict(
                            projection=projection,
                            m=m,
                            dtype=str(dtype),
                            layout=layout,
                            tile=tile,
                            repeat=repeat,
                            timings_ms=timings,
                            speedup=timings["current"] / timings["candidate"],
                            different_outputs=differences,
                            relative_rms=error,
                        )
                        records.append(row)
                        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                triton=triton.__version__,
                records=records,
                timing=(
                    "complete call including quantization and allocation; "
                    "CUDA graph replay"
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
