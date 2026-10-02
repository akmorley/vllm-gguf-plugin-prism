# SPDX-License-Identifier: Apache-2.0
"""Compare integrated byte expansion with a frozen pre-change integer kernel."""

import argparse
import importlib.util
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_prism import PROJECTIONS
from triton.testing import do_bench_cudagraph

from vllm_gguf_plugin.triton.pq2_int_gemv import pq2_int_gemv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--projections", nargs="+", choices=PROJECTIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "vllm_gguf_plugin.triton.previous_int_gemv", args.previous
    )
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    torch.manual_seed(113)
    records = []
    for projection in args.projections:
        n, k = PROJECTIONS[projection]
        for m in (1, 2, 4, 5, 8):
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
                    scales = (
                        torch.randn(n, k // 128, dtype=torch.float16, device="cuda")
                        * 0.02
                    )
                    weight.view(n, k // 128, 34)[:, :, :2] = scales.view(
                        torch.uint8
                    ).reshape(n, k // 128, 2)
                    x = torch.randn(m, k, dtype=dtype, device="cuda")
                    methods = {
                        "previous": partial(previous.pq2_int_gemv, x, weight),
                        "integrated": partial(pq2_int_gemv, x, weight),
                    }
                    expected = methods["previous"]()
                    actual = methods["integrated"]()
                    torch.testing.assert_close(
                        actual.view(torch.int16),
                        expected.view(torch.int16),
                        atol=0,
                        rtol=0,
                    )
                    timings = []
                    for order in (list(methods), list(reversed(methods))):
                        timings.append(
                            {
                                name: do_bench_cudagraph(methods[name], rep=100)
                                for name in order
                            }
                        )
                    row = dict(
                        projection=projection,
                        m=m,
                        dtype=str(dtype),
                        layout=layout,
                        different_outputs=0,
                        timings_ms=timings,
                    )
                    records.append(row)
                    print(json.dumps(row), flush=True)
                    del methods, storage, weight, scales, x, expected, actual
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                records=records,
                protocol=(
                    "Fixed 16/4/4 tile; full call including quantization "
                    "and allocations; "
                    "warmed CUDA graphs; forward/reverse order; 100ms per measurement"
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
