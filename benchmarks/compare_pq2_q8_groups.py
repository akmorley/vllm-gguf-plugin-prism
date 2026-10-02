# SPDX-License-Identifier: Apache-2.0
"""Offline activation-group diagnostic; neither serving nor llama.cpp parity."""

import argparse
import json
from pathlib import Path

import torch


def relative_rms(actual, expected):
    return ((actual - expected).square().sum() / expected.square().sum()).sqrt().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    records = []
    for probe in json.loads((args.captures / "projection-comparison.json").read_text()):
        saved = torch.load(args.captures / probe["artifact"], weights_only=True)
        x = saved["x"].float()
        m, k = x.shape
        blocks = saved["packed_rows"].reshape(-1, k // 128, 34)
        scales = blocks[:, :, :2].contiguous().view(torch.float16).double()
        codes = ((blocks[:, :, 2:, None].int() >> (2 * torch.arange(4))) & 3) - 1
        weights = (codes.reshape(-1, k // 128, 128).double() * scales).reshape(-1, k)
        reference = x.double() @ weights.T
        metrics = {}
        for group in (128, 32):
            activations = x.reshape(m, k // group, group)
            # Triton lowers division by this constant to an FP32 reciprocal
            # multiply. Preserve that scale rounding in both group variants.
            scale = activations.abs().amax(-1, keepdim=True) * (1.0 / 127)
            quantized = (
                (activations / torch.where(scale > 0, scale, 1))
                .round()
                .clamp(-127, 127)
            )
            reconstructed = (quantized.double() * scale.double()).reshape(m, k)
            projected = reconstructed @ weights.T
            if group == 128:
                torch.testing.assert_close(
                    projected, saved["quantized_fp64"], atol=1e-12, rtol=1e-12
                )
            metrics[str(group)] = dict(
                activation_relative_rms=relative_rms(reconstructed, x.double()),
                projection_relative_rms=relative_rms(projected, reference),
            )
        records.append(
            dict(artifact=probe["artifact"], n=probe["n"], k=k, metrics=metrics)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                records=records,
                protocol=(
                    "CPU FP64 reference preserving stored FP16 weight scales; "
                    "both groups use FP32 absmax/127 and nearest-even rounding; "
                    "isolates group size, not llama.cpp parity or model quality"
                ),
            ),
            indent=2,
        )
    )
    print("projection samples", len(records))


if __name__ == "__main__":
    main()
