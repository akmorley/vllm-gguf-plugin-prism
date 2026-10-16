# SPDX-License-Identifier: Apache-2.0
"""Experimental PQ2 x Q8 prefill; serving requires GGUF_PQ2_MMQ=1.

Activation groups of 32, 64 or 128 values have FP32 scales and signed INT8 values.
Integer accumulation is reset at every PQ2 scale boundary. This implementation
is independent of llama.cpp and does not promise identical Q8 rounding/layout.
"""

import torch
import triton
import triton.language as tl

from .pq2_layout import _pq2_codes, _pq2_scale, validate_prepared


@triton.jit
def _quantize(X, Q, S, K: tl.constexpr, GROUP: tl.constexpr = 128):
    block = tl.program_id(0)
    i = tl.arange(0, GROUP)
    a = tl.load(X + block * GROUP + i).to(tl.float32)
    scale = tl.max(tl.abs(a), 0) / 127.0
    normalized = tl.div_rn(a, tl.where(scale > 0, scale, 1.0))
    # Round nearest, ties to even, then clamp before the narrowing conversion.
    q = tl.minimum(
        127.0, tl.maximum(-127.0, tl.extra.cuda.libdevice.nearbyint(normalized))
    )
    tl.store(Q + block * GROUP + i, q.to(tl.int8))
    tl.store(S + block, scale)


@triton.jit(do_not_specialize=["M"])
def _mmq(
    Q,
    S,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PREPARED: tl.constexpr = False,
    GROUP: tl.constexpr = 128,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, GROUP)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(K // GROUP):
        a = tl.load(
            Q + m[:, None] * K + block * GROUP + k[None, :], m[:, None] < M, other=0
        )
        weight_block = block // (128 // GROUP)
        ws = _pq2_scale(W, n, weight_block, N, K, STRIDE, n < N, PREPARED)
        # Load each packed byte once; subgroups retain the PQ2 weight scale.
        byte = tl.arange(0, GROUP // 4) + (block % (128 // GROUP)) * (GROUP // 4)
        packed = _pq2_codes(
            W,
            n[None, :],
            weight_block,
            byte[:, None],
            K,
            STRIDE,
            n[None, :] < N,
            PREPARED,
        )
        shifts = 2 * tl.arange(0, 4)
        codes = ((packed.to(tl.int32)[:, None, :] >> shifts[None, :, None]) & 3) - 1
        b = tl.reshape(codes.to(tl.int8), (GROUP, BN))
        partial = tl.dot(a, b).to(tl.float32)
        xs = tl.load(S + m * (K // GROUP) + block, m < M, other=0)
        acc += partial * xs[:, None] * ws[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def pq2_mmq(x, weight, *, prepared=False, activation_group=128):
    """Quantize activations once and reuse across all output tiles.

    Additional activation approximation requires model-quality validation before
    default serving adoption. Temporary storage is M*K bytes plus M*K/group FP32 scales.
    """
    if not isinstance(activation_group, int) or activation_group not in (32, 64, 128):
        raise ValueError("MMQ activation group must be 32, 64 or 128")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("PQ2 MMQ requires two matrices")
    m, k = x.shape
    n = weight.shape[0]
    if k == 0 or k % 128 or weight.shape[1] != k // 128 * 34:
        raise ValueError("Invalid PQ2 packed matrix shape")
    if not x.is_cuda or weight.device != x.device:
        raise ValueError("PQ2 MMQ requires inputs on the same CUDA device")
    if (
        x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or weight.dtype != torch.uint8
    ):
        raise ValueError("PQ2 MMQ requires floating activations and uint8 weights")
    x = x.contiguous()
    if prepared:
        validate_prepared(weight)
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m and n:
        q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        scales = torch.empty(
            (m, k // activation_group), dtype=torch.float32, device=x.device
        )
        _quantize[(m * (k // activation_group),)](x, q, scales, k, activation_group)
        _mmq[(triton.cdiv(m, 64), triton.cdiv(n, 128))](
            q,
            scales,
            weight,
            y,
            m,
            n,
            k,
            weight.stride(0),
            64,
            128,
            prepared,
            activation_group,
        )
    return y
