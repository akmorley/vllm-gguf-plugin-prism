# SPDX-License-Identifier: Apache-2.0
"""Check packed variants against previously captured real model activations."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from vllm_gguf_plugin.triton.pq2_int_gemv import pq2_int_gemv
from vllm_gguf_plugin.triton.pq2_packed_experiment import packed_gemv, prepare_pq2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for probe in json.loads((args.captures / "projection-comparison.json").read_text()):
        saved = torch.load(args.captures / probe["artifact"], weights_only=True)
        x, weight = saved["x"].cuda(), saved["packed_rows"].cuda()
        expected = pq2_int_gemv(x, weight)
        torch.testing.assert_close(
            expected.cpu().view(torch.int16),
            saved["candidate"].to(x.dtype).view(torch.int16),
            rtol=0,
            atol=0,
        )
        raw = prepare_pq2(weight)
        half = prepare_pq2(weight, load_bytes=2)
        soa = prepare_pq2(weight, separate=True)
        variants = [
            ("raw_spread", raw, False),
            ("soa_spread", soa, False),
            ("half_keep", half, True),
            (
                "soa_half_keep",
                replace(soa, codes=soa.codes.view(torch.int16), load_bytes=2),
                True,
            ),
            (
                "soa_word_keep",
                replace(soa, codes=soa.codes.view(torch.int32), load_bytes=4),
                True,
            ),
        ]
        for name, prepared, keep in variants:
            actual = packed_gemv(x, prepared, fast_pack=True, keep_lanes=keep)
            torch.testing.assert_close(
                actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
            )
            records.append(
                dict(
                    artifact=probe["artifact"],
                    variant=name,
                    n=probe["n"],
                    k=probe["k"],
                    outputs=actual.numel(),
                    different_outputs=0,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                projection_samples=len(records) // 5,
                variant_comparisons=len(records),
                result=(
                    "bitwise match against saved integer outputs; original 16/4/4 tile"
                ),
                records=records,
            ),
            indent=2,
        )
    )
    print("bitwise replay comparisons", len(records))


if __name__ == "__main__":
    main()
