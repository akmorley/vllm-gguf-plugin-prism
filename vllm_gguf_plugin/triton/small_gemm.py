# SPDX-License-Identifier: Apache-2.0
"""Split-K Triton GEMM for small unquantized (BF16 / FP16) projections at decode batch sizes.

y[M, N] = x[M, K] @ w[N, K].T for M <= 16 and small N, e.g. the 96-output Qwen3.5 ``in_proj_ba`` (ssm_alpha + ssm_beta,
K = 5120) that runs once per linear-attention layer. cuBLAS picks an 8-CTA kernel without split-K for it at M = 2..4
(~30 us per call on an RTX 3090 Ti, ~30 GB/s); this kernel splits K over (N / BN) x (K / BK) programs and sums the FP32
partials in fixed order (deterministic): ~4-5 us at every M <= 16. On in the `tuned` kernel set (rounding-level
difference from cuBLAS); GGUF_SMALL_GEMM_TRITON=0 disables.
"""
import torch
import triton
import triton.language as tl

from . import pq2_config as _config

ENABLED = _config.flag("GGUF_SMALL_GEMM_TRITON", "numerics_class")
_BM, _BN, _BK = 16, 32, 256
MAX_M, MAX_N = 16, 1024


@triton.jit
def _small_gemm_splitk(X, W, P, M, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr):
    # P[split, M, N] = X[:, k slice] @ W[n tile, k slice].T in FP32; one program per (N tile, K split)
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    m = tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = pk * BK + tl.arange(0, BK)
    x = tl.load(X + m[:, None] * K + k[None, :], mask=m[:, None] < M, other=0.0)
    w = tl.load(W + n[:, None] * K + k[None, :], mask=n[:, None] < N, other=0.0)
    acc = tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    tl.store(P + pk * M * N + m[:, None] * N + n[None, :], acc, mask=(m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce_splits_cast(P, Y, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    # fixed summation order over the K splits, then one rounding to the output dtype
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < MN
    acc = tl.load(P + i, ok, other=0.0)
    for s in tl.static_range(1, SPLIT):
        acc += tl.load(P + s * MN + i, ok, other=0.0)
    tl.store(Y + i, acc.to(Y.dtype.element_ty), ok)


def eligible(x, weight):
    return (ENABLED and x.is_cuda and x.dim() == 2 and 1 <= x.shape[0] <= MAX_M and weight.dim() == 2
            and weight.shape[0] <= MAX_N and x.shape[1] == weight.shape[1] and x.shape[1] % _BK == 0
            and x.dtype in (torch.bfloat16, torch.float16) and weight.dtype == x.dtype and weight.is_contiguous())


def small_gemm(x, weight):
    """x @ weight.T for eligible() inputs."""
    x = x.contiguous()
    m, k = x.shape
    n = weight.shape[0]
    split = k // _BK
    p = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _small_gemm_splitk[(triton.cdiv(n, _BN), split)](x, weight, p, m, n, k, _BM, _BN, _BK, num_warps=4)
    _reduce_splits_cast[(triton.cdiv(m * n, 1024),)](p, y, m * n, split, 1024)
    return y
