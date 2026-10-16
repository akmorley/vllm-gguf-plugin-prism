# SPDX-License-Identifier: Apache-2.0
"""Experimental PQ2 x Q8 prefill; serving requires GGUF_PQ2_MMQ=1.

Activation groups of 32, 64 or 128 values have FP32 scales and signed INT8 values.
Integer accumulation is reset at every PQ2 scale boundary. This implementation
is independent of llama.cpp and does not promise identical Q8 rounding/layout.
"""

import os

import torch
import triton
import triton.language as tl

from . import pq2_config as _config
from .pq2_layout import _pq2_codes, _pq2_scale, validate_prepared

# Shape-adaptive tiles; bit-exact with the fixed 64x128 tile. GGUF_PQ2_MMQ_TUNED_TILES=0 reverts.
_TUNED_TILES = _config.flag("GGUF_PQ2_MMQ_TUNED_TILES", "exact_class")
# Batched activation quantize (bit-exact with the one-group-per-program kernel).
_QUANT_ROWS = _config.flag("GGUF_PQ2_PREFILL_ROWS", "exact_class")
# Hand-written INT8 tensor-core prefill kernel (prepared weights, SM86, group-128): bit-exact with the
# Triton kernel and ~15% faster sustained at a 300 W cap on wide shapes, ~35% on the N = 5120 ones.
_MMA_PREFILL = _config.flag("GGUF_PQ2_MMA_PREFILL", "exact_class")


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


@triton.jit
def _quantize_rows(X, Q, S, G, GROUP: tl.constexpr, R: tl.constexpr):
    # Same arithmetic as _quantize with R groups per program (~2x faster at prefill sizes).
    r1 = tl.program_id(0) * R + tl.arange(0, R)
    r = r1[:, None]
    i = tl.arange(0, GROUP)[None, :]
    mask = r < G
    a = tl.load(X + r * GROUP + i, mask, other=0).to(tl.float32)
    scale = tl.max(tl.abs(a), 1) / 127.0
    normalized = tl.div_rn(a, tl.where(scale > 0, scale, 1.0)[:, None])
    q = tl.minimum(
        127.0, tl.maximum(-127.0, tl.extra.cuda.libdevice.nearbyint(normalized))
    )
    tl.store(Q + r * GROUP + i, q.to(tl.int8), mask)
    tl.store(S + r1, scale, r1 < G)


@triton.jit
def _quantize_prefill(X, Q, S, T, G, R: tl.constexpr):
    """Group-128 Q8 quantization (arithmetic identical to _quantize) into the prefill fragment layout.

    Element i = 4 * byte + s of a group is stored at tid * 32 + s * 8 + h * 4 + j with byte = 8 * tid +
    4 * h + j, so each MMA lane's B fragment is 32 contiguous bytes. Also writes the INT32 group sums.
    """
    r1 = tl.program_id(0) * R + tl.arange(0, R)
    r = r1[:, None]
    i = tl.arange(0, 128)[None, :]
    mask = r < G
    a = tl.load(X + r * 128 + i, mask, other=0).to(tl.float32)
    scale = tl.max(tl.abs(a), 1) / 127.0
    normalized = tl.div_rn(a, tl.where(scale > 0, scale, 1.0)[:, None])
    q = tl.minimum(
        127.0, tl.maximum(-127.0, tl.extra.cuda.libdevice.nearbyint(normalized))
    )
    qi = q.to(tl.int8)
    byte = i // 4
    off = byte % 8
    pos = (byte // 8) * 32 + (i % 4) * 8 + off
    tl.store(Q + r * 128 + pos, qi, mask)
    tl.store(S + r1, scale, r1 < G)
    tl.store(T + r1, tl.sum(qi.to(tl.int32), 1), r1 < G)


def _mma_prefill_available():
    try:
        from .. import _C_gguf  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops, "_C_gguf") and hasattr(torch.ops._C_gguf, "pq2_mma_prefill")


def _mma_prefill(x, weight, y, m, n, k):
    """INT8 tensor-core prefill on prepared planes; bit-exact with the Triton _mmq path."""
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    scales = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
    sums = torch.empty((m, k // 128), dtype=torch.int32, device=x.device)
    groups = m * (k // 128)
    _quantize_prefill[(triton.cdiv(groups, 8),)](x, q, scales, sums, groups, 8, num_warps=4)
    # 128x128 tile for wide outputs, 256x64 for the N = 5120 projections (measured, sustained at 300 W).
    torch.ops._C_gguf.pq2_mma_prefill(q, scales, sums, weight, y, 0 if n >= 14336 else 1)


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


def _tile(m, n, k, group):
    """(BM, BN, warps, stages) for the INT8 tile kernel; every choice is bit-exact with 64x128.

    Measured on RTX 3090 Ti, prepared weights, five model projection shapes: taller tiles
    amortize the weight-code unpack over more rows. 256x64/8 warps is ~1.3-1.45x faster than
    64x128 at M >= 512; 128x128/8 warps wins for wide outputs at 256 <= M < 512; 128x64 wins
    for narrow outputs (N = 5120) and for M < 256 where wide tiles under-fill the GPU.
    """
    if group != 128 or not _TUNED_TILES:
        return 64, 128, 4, 3
    if m >= 512:
        return 256, 64, 8, 2
    if m < 256 or (n <= 5120 and k <= 6144):
        return 128, 64, 4, 2
    return 128, 128, 8, 3


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
    if (
        m >= 64
        and n
        and _MMA_PREFILL
        and prepared
        and activation_group == 128
        and k % 256 == 0
        and x.dtype in (torch.bfloat16, torch.float16)
        and torch.cuda.get_device_capability(x.device) == (8, 6)
        and _mma_prefill_available()
    ):
        _mma_prefill(x, weight, y, m, n, k)
        return y
    if m and n:
        q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        scales = torch.empty(
            (m, k // activation_group), dtype=torch.float32, device=x.device
        )
        groups = m * (k // activation_group)
        if _QUANT_ROWS:
            _quantize_rows[(triton.cdiv(groups, 8),)](
                x, q, scales, groups, activation_group, 8, num_warps=4
            )
        else:
            _quantize[(groups,)](x, q, scales, k, activation_group)
        bm, bn, warps, stages = _tile(m, n, k, activation_group)
        _mmq[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            q,
            scales,
            weight,
            y,
            m,
            n,
            k,
            weight.stride(0),
            bm,
            bn,
            prepared,
            activation_group,
            num_warps=warps,
            num_stages=stages,
        )
    return y
