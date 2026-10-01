# SPDX-License-Identifier: Apache-2.0
"""PQ2 projection matrix; run on an idle GPU with the repository virtualenv."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import triton
from gguf import GGMLQuantizationType
from triton.testing import do_bench_cudagraph

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.triton.pq2_gemv import _batched_gemv, pq2_batched_gemv
from vllm_gguf_plugin.triton.pq2_int_gemv import _int_gemv, pq2_int_gemv
from vllm_gguf_plugin.triton.pq2_mmq import _mmq, _quantize, pq2_mmq
from vllm_gguf_plugin.triton.prism import _pq2, _pq2_gemv, pq2_matmul

PROJECTIONS = {
    "ffn_gate_up": (34816, 5120),
    "ffn_down": (5120, 17408),
    "gdn_input": (16384, 5120),
    "attention_qkv": (14336, 5120),
    "output_head": (248320, 5120),
}
TOKENS = (1, 2, 4, 5, 8, 16, 32, 128, 512, 1024)


def kernel_resources():
    """Record compiled resources when exposed by this Triton version."""
    result = []
    for function in (_pq2, _pq2_gemv, _mmq, _quantize, _batched_gemv, _int_gemv):
        for cache in function.device_caches.values():
            for kernel in cache[0].values():
                result.append(
                    {
                        "kernel": function.__name__,
                        "registers": getattr(kernel, "n_regs", None),
                        "spills": getattr(kernel, "n_spills", None),
                        "shared_bytes": getattr(kernel.metadata, "shared", None),
                        "integer_dp4a_ptx": sorted(
                            {
                                line.strip()
                                for line in kernel.asm.get("ptx", "").splitlines()
                                if "dp4a." in line
                            }
                        ),
                        "integer_mma_ptx": sorted(
                            {
                                line.strip()
                                for line in kernel.asm.get("ptx", "").splitlines()
                                if "mma.sync" in line and ".s8.s8." in line
                            }
                        ),
                    }
                )
    return result


def benchmark(m, n, k, dtype, layout, methods, rep):
    stride = k // 128 * 34
    storage = torch.randint(
        0,
        256,
        (n * (2 if layout == "strided" else 1), stride),
        device="cuda",
        dtype=torch.uint8,
    )
    w = storage[::2] if layout == "strided" else storage
    blocks = w.view(n, k // 128, 34)
    scales = torch.rand(n, k // 128, device="cuda", dtype=torch.float16) * 0.02
    blocks[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
    x = torch.randn(m, k, device="cuda", dtype=dtype)

    def dequantize():
        # The diagnostic backend expects contiguous packed rows.
        return ops.ggml_dequantize(
            w.contiguous(), int(GGMLQuantizationType.PQ2_0), n, k, dtype
        )

    # Validate against FP32 accumulation of dtype-rounded dequantized weights.
    dense = dequantize()
    expected = (x.float() @ dense.float().T).to(dtype)
    torch.cuda.synchronize()
    start = time.perf_counter()
    actual = pq2_matmul(x, w)
    torch.cuda.synchronize()
    native_first_call_ms = (time.perf_counter() - start) * 1000
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.02 if dtype == torch.bfloat16 else 0.005,
        rtol=0.02 if dtype == torch.bfloat16 else 0.005,
    )
    mmq_first_call_ms = None
    mmq_relative_rms = None
    if "mmq" in methods:
        torch.cuda.synchronize()
        start = time.perf_counter()
        candidate = pq2_mmq(x, w)
        torch.cuda.synchronize()
        mmq_first_call_ms = (time.perf_counter() - start) * 1000
        mmq_relative_rms = (
            (
                (candidate.float() - expected.float()).square().sum()
                / expected.float().square().sum()
            )
            .sqrt()
            .item()
        )
        if not math.isfinite(mmq_relative_rms) or mmq_relative_rms > 0.03:
            raise AssertionError(f"MMQ relative RMS {mmq_relative_rms} exceeds 0.03")
    int_gemv_first_call_ms = None
    int_gemv_relative_rms = None
    if "int_gemv" in methods:
        torch.cuda.synchronize()
        start = time.perf_counter()
        candidate = pq2_int_gemv(x, w)
        torch.cuda.synchronize()
        int_gemv_first_call_ms = (time.perf_counter() - start) * 1000
        int_gemv_relative_rms = (
            (
                (candidate.float() - expected.float()).square().sum()
                / expected.float().square().sum()
            )
            .sqrt()
            .item()
        )
        if not math.isfinite(int_gemv_relative_rms) or int_gemv_relative_rms > 0.03:
            raise AssertionError(
                f"Integer GEMV relative RMS {int_gemv_relative_rms} exceeds 0.03"
            )
    gemv_first_call_ms = None
    if "batched_gemv" in methods:
        torch.cuda.synchronize()
        start = time.perf_counter()
        candidate = pq2_batched_gemv(x, w)
        torch.cuda.synchronize()
        gemv_first_call_ms = (time.perf_counter() - start) * 1000
        torch.testing.assert_close(
            candidate,
            expected,
            atol=0.02 if dtype == torch.bfloat16 else 0.005,
            rtol=0.02 if dtype == torch.bfloat16 else 0.005,
        )
    del expected, actual
    functions = {
        "int_gemv": lambda: pq2_int_gemv(x, w),
        "batched_gemv": lambda: pq2_batched_gemv(x, w),
        "native": lambda: pq2_matmul(x, w),
        "mmq": lambda: pq2_mmq(x, w),
        "dequant_dense": lambda: x @ dequantize().T,
        "dense": lambda: x @ dense.T,
    }
    rows = []
    for method in methods:
        fn = functions[method]
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - start) * 1000
        # Warm compilation and libraries before graph capture/timing.
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        temporary_bytes = torch.cuda.max_memory_allocated() - allocated
        latency_ms = do_bench_cudagraph(fn, rep=rep)
        rows.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "dtype": str(dtype),
                "layout": layout,
                "method": method,
                "latency_ms": latency_ms,
                "relative_rms_vs_float": mmq_relative_rms
                if method == "mmq"
                else int_gemv_relative_rms
                if method == "int_gemv"
                else None,
                "packed_weight_GB_s": n * stride / (latency_ms * 1e6)
                if method in ("native", "mmq", "batched_gemv", "int_gemv")
                else None,
                "temporary_bytes_including_output": temporary_bytes,
                "first_call_ms": native_first_call_ms
                if method == "native"
                else mmq_first_call_ms
                if method == "mmq"
                else gemv_first_call_ms
                if method == "batched_gemv"
                else int_gemv_first_call_ms
                if method == "int_gemv"
                else first_call_ms,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projections", nargs="+", choices=PROJECTIONS, default=list(PROJECTIONS)
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=list(TOKENS),
        help="Include additional CUDA graph padded sizes here",
    )
    parser.add_argument(
        "--dtypes", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"]
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("contiguous", "strided"),
        default=["contiguous", "strided"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("native", "mmq", "batched_gemv", "int_gemv", "dequant_dense", "dense"),
        default=["native", "dequant_dense", "dense"],
    )
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens) <= 0 or args.rep_ms <= 0:
        parser.error("tokens and rep-ms must be positive")
    if {"batched_gemv", "int_gemv"}.intersection(args.methods) and max(
        args.tokens
    ) > 16:
        parser.error("decode GEMV candidates require token counts <= 16")
    torch.manual_seed(7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "timing": "warmed CUDA graph replay; dense validation precedes timing",
    }
    with args.output.open("w", newline="") as stream:
        writer = None
        for name in args.projections:
            n, k = PROJECTIONS[name]
            for dtype_name in args.dtypes:
                dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
                for layout in args.layouts:
                    for m in args.tokens:
                        for row in benchmark(
                            m, n, k, dtype, layout, args.methods, args.rep_ms
                        ):
                            row = {"projection": name, **row}
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=row)
                                writer.writeheader()
                            writer.writerow(row)
                            stream.flush()
                            print(row, flush=True)
    metadata["compiled_resources"] = kernel_resources()
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
