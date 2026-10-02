# SPDX-License-Identifier: Apache-2.0
"""Matched raw/prepared complete-call timings and real-activation replay."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_prism import PROJECTIONS
from triton.testing import do_bench_cudagraph

from vllm_gguf_plugin.triton.pq2_int_gemv import pq2_int_gemv
from vllm_gguf_plugin.triton.pq2_layout import prepare_pq2_layout
from vllm_gguf_plugin.triton.prism import pq2_matmul

PREPARED_PROJECTIONS = {
    **PROJECTIONS,
    "attention_gdn_output": (5120, 6144),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projections", nargs="+", choices=PREPARED_PROJECTIONS)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(131)
    records = []
    replay = []
    for projection in args.projections or []:
        n, k = PREPARED_PROJECTIONS[projection]
        for m in (1, 4, 8, 128, 512):
            for dtype in (torch.bfloat16, torch.float16):
                for layout in ("contiguous", "strided") if m <= 8 else ("contiguous",):
                    storage = torch.randint(
                        0,
                        256,
                        (n * (2 if layout == "strided" else 1), k // 128 * 34),
                        device="cuda",
                        dtype=torch.uint8,
                    )
                    weight = storage[::2] if layout == "strided" else storage
                    ws = (
                        torch.randn(n, k // 128, device="cuda", dtype=torch.float16)
                        * 0.02
                    )
                    weight.view(n, k // 128, 34)[:, :, :2] = ws.view(
                        torch.uint8
                    ).reshape(n, k // 128, 2)
                    prepared = prepare_pq2_layout(weight)
                    assert prepared.untyped_storage().nbytes() == weight.numel()
                    x = torch.randn(m, k, device="cuda", dtype=dtype)
                    method = (
                        pq2_int_gemv
                        if m <= 8 and projection != "attention_gdn_output"
                        else pq2_matmul
                    )
                    methods = {
                        "raw": partial(method, x, weight),
                        "prepared": partial(method, x, prepared, prepared=True),
                    }
                    expected = methods["raw"]()
                    torch.testing.assert_close(
                        methods["prepared"]().view(torch.uint8),
                        expected.view(torch.uint8),
                        atol=0,
                        rtol=0,
                    )
                    timings = (
                        []
                        if args.validate_only
                        else [
                            {
                                name: do_bench_cudagraph(methods[name], rep=100)
                                for name in order
                            }
                            for order in (list(methods), list(reversed(methods)))
                        ]
                    )
                    row = dict(
                        projection=projection,
                        m=m,
                        dtype=str(dtype),
                        layout=layout,
                        method=method.__name__,
                        different_outputs=0,
                        resident_prepared_bytes=prepared.numel(),
                        timings_ms=timings,
                    )
                    records.append(row)
                    print(json.dumps(row), flush=True)
                    del methods, storage, weight, prepared, ws, x, expected
    if args.captures:
        for probe in json.loads(
            (args.captures / "projection-comparison.json").read_text()
        ):
            saved = torch.load(args.captures / probe["artifact"], weights_only=True)
            x, weight = saved["x"].cuda(), saved["packed_rows"].cuda()
            prepared = prepare_pq2_layout(weight)
            for name, method, reference in (
                ("integer", pq2_int_gemv, saved["candidate"]),
                ("floating", pq2_matmul, saved["native"]),
            ):
                result = method(x, prepared, prepared=True)
                torch.testing.assert_close(
                    result.cpu().view(torch.uint8),
                    reference.to(x.dtype).view(torch.uint8),
                    atol=0,
                    rtol=0,
                )
                replay.append(
                    dict(artifact=probe["artifact"], method=name, different_outputs=0)
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                records=records,
                replay=replay,
                protocol=(
                    "Complete calls; graph replay; 100ms; forward/reverse order; "
                    "preparation excluded; fixed serving tiles"
                ),
            ),
            indent=2,
        )
    )
    print("cases", len(records), "replays", len(replay), flush=True)


if __name__ == "__main__":
    main()
