# SPDX-License-Identifier: Apache-2.0
"""Run with PYTHONPATH=. python benchmarks/benchmark_prism.py on an idle GPU."""

import torch
from gguf import GGMLQuantizationType
from triton.testing import do_bench

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.hadamard import fwht_blockwise
from vllm_gguf_plugin.triton.prism import hadamard, pq2_matmul


def benchmark(m):
    n, k = 4096, 5120
    w = torch.randint(0, 256, (n, k // 128, 34), device="cuda", dtype=torch.uint8)
    scales = torch.ones(n, k // 128, device="cuda", dtype=torch.float16) * 0.01
    w[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
    w = w.reshape(n, -1)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)

    def fallback():
        dense = ops.ggml_dequantize(w, int(GGMLQuantizationType.PQ2_0), n, k, x.dtype)
        return x @ dense.T

    torch.testing.assert_close(
        pq2_matmul(x, w),
        (
            x.float()
            @ ops.ggml_dequantize(w, int(GGMLQuantizationType.PQ2_0), n, k, x.dtype)
            .float()
            .T
        ).half(),
        atol=0.005,
        rtol=0.005,
    )
    print(
        m,
        "PQ2 native ms",
        do_bench(lambda: pq2_matmul(x, w)),
        "fallback ms",
        do_bench(fallback),
        flush=True,
    )
    print(
        m,
        "FWHT fused ms",
        do_bench(lambda: hadamard(x, None, 1024)),
        "eager ms",
        do_bench(lambda: fwht_blockwise(x, 1024)),
        flush=True,
    )


if __name__ == "__main__":
    for tokens in (1, 128):
        benchmark(tokens)
