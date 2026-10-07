# SPDX-License-Identifier: Apache-2.0
"""Prism kernels: packed PQ2 multiplication and normalized block FWHT."""

import os

import torch
import triton
import triton.language as tl

from .pq2_layout import _pq2_codes, _pq2_scale, validate_prepared

# Read once before graph capture; opt-in while model/serving acceptance is pending.
_EXPERIMENTAL_BATCHED_GEMV = os.environ.get("GGUF_PQ2_BATCHED_GEMV", "0") == "1"
_EXPERIMENTAL_INT_GEMV = os.environ.get("GGUF_PQ2_INT_GEMV", "0") == "1"
_EXPERIMENTAL_INT_OUTPUT = os.environ.get("GGUF_PQ2_INT_OUTPUT", "1") == "1"
_EXPERIMENTAL_MMQ = os.environ.get("GGUF_PQ2_MMQ", "0") == "1"
_MMQ_ACTIVATION_GROUP = int(os.environ.get("GGUF_PQ2_MMQ_GROUP", "128"))
_BATCH8_FLOAT_OUTPUT_BM = int(os.environ.get("GGUF_PQ2_BATCH8_FLOAT_OUTPUT_BM", "8"))
# Bit-exact: smaller floating row tiles for (5120, 6144) at M = 5..16 (8 rows up to M = 8,
# 16 rows above), instead of padding to 64. Extends the batch-eight rule to verification sizes.
_SMALL_FLOAT_OUTPUT = os.environ.get("GGUF_PQ2_SMALL_FLOAT_OUTPUT", "1") == "1"
if _BATCH8_FLOAT_OUTPUT_BM not in (8, 16, 32, 64):
    raise ValueError("GGUF_PQ2_BATCH8_FLOAT_OUTPUT_BM must be 8, 16, 32 or 64")
if _MMQ_ACTIVATION_GROUP not in (32, 64, 128):
    raise ValueError("GGUF_PQ2_MMQ_GROUP must be 32, 64 or 128")
_BATCHED_GEMV_SHAPES = frozenset(
    {(34816, 5120), (5120, 17408), (16384, 5120), (14336, 5120), (248320, 5120)}
)


def _use_batched_gemv(x, n, k):
    return (
        _EXPERIMENTAL_BATCHED_GEMV
        and x.shape[0] == 4
        and (n, k) in _BATCHED_GEMV_SHAPES
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_cuda
        and torch.cuda.get_device_capability(x.device) == (8, 6)
    )


def _use_int_gemv(x, n, k):
    # Only measured batches/shapes: M=16 gains are too close to noise to select.
    return (
        _EXPERIMENTAL_INT_GEMV
        and x.shape[0] in (1, 2, 4, 5, 8)
        and ((n, k) in _BATCHED_GEMV_SHAPES
             or (_EXPERIMENTAL_INT_OUTPUT and (n, k) == (5120, 6144) and x.shape[0] == 1))
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_cuda
        and torch.cuda.get_device_capability(x.device) == (8, 6)
    )


@triton.jit
def _hadamard(
    X,
    S,
    Y,
    WIDTH: tl.constexpr,
    B: tl.constexpr,
    SIGNS: tl.constexpr,
    INVERSE: tl.constexpr,
):
    i = tl.arange(0, B)
    offset = tl.program_id(0) * B + i
    v = tl.load(X + offset).to(tl.float32)
    if SIGNS:
        signs = tl.load(S + offset % WIDTH).to(tl.float32)
        if not INVERSE:
            v *= signs
    for stage in tl.static_range(0, tl.constexpr(B.bit_length() - 1)):
        h = 1 << stage
        other = tl.gather(v, i ^ h, 0)
        v = tl.where((i & h) == 0, v + other, other - v)
    v *= B**-0.5
    if SIGNS and INVERSE:
        v *= signs
    tl.store(Y + offset, v)


def hadamard(x, signs, block_size, inverse=False):
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block size must be a power of two")
    if x.shape[-1] % block_size:
        raise ValueError("Activation width must be divisible by Hadamard block size")
    x = x.contiguous()
    y = torch.empty_like(x)
    if x.numel():
        _hadamard[(x.numel() // block_size,)](
            x,
            signs if signs is not None else x,
            y,
            x.shape[-1],
            block_size,
            signs is not None,
            inverse,
        )
    return y


@triton.jit(do_not_specialize=["M"])
def _pq2(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PREPARED: tl.constexpr = False,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        k = base * BK + kk
        a = tl.load(
            X + m[:, None] * K + k[None, :],
            (m[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        if PREPARED:
            tl.static_assert(BK == 128, "Prepared GEMM requires a scale-aligned K tile")
            byte = tl.arange(0, 32)
            packed = _pq2_codes(
                W, n[None, :], base, byte[:, None], K, STRIDE, n[None, :] < N, True
            )
            shifts = 2 * tl.arange(0, 4)
            code = ((packed.to(tl.int32)[:, None, :] >> shifts[None, :, None]) & 3) - 1
            code = tl.reshape(code, (BK, BN))
            scale = _pq2_scale(W, n, base, N, K, STRIDE, n < N, True)
            b = (code.to(tl.float32) * scale[None, :]).to(a.dtype)
        else:
            block = k // 128
            mask = (n[None, :] < N) & (k[:, None] < K)
            scale = _pq2_scale(W, n[None, :], block[:, None], N, K, STRIDE, mask, False)
            packed = _pq2_codes(
                W,
                n[None, :],
                block[:, None],
                (k[:, None] % 128) // 4,
                K,
                STRIDE,
                mask,
                False,
            )
            code = ((packed.to(tl.int32) >> (2 * (k[:, None] % 4))) & 3) - 1
            b = (code.to(tl.float32) * scale).to(a.dtype)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def pq2_matmul(x, weight, *, prepared=False):
    """Dequantize only a tile at a time, never materializing a dense weight."""
    m, k = x.shape
    n = weight.shape[0]
    if k % 128 or weight.shape[1] != k // 128 * 34:
        raise ValueError("Invalid PQ2 packed matrix shape")
    x = x.contiguous()
    if prepared:
        validate_prepared(weight)
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    if prepared:
        from . import pq2_small_mmq

        if pq2_small_mmq.eligible(x, weight, prepared):
            return pq2_small_mmq.pq2_small_mmq(x, weight)
    if _use_int_gemv(x, n, k):
        from .pq2_int_gemv import pq2_int_gemv

        if prepared:
            return pq2_int_gemv(x, weight, prepared=True)
        return pq2_int_gemv(x, weight)
    if _use_batched_gemv(x, n, k):
        from .pq2_gemv import pq2_batched_gemv

        if prepared:
            return pq2_batched_gemv(x, weight, prepared=True)
        return pq2_batched_gemv(x, weight)
    # M>=128 is the measured large-prefill range. Small/decode batches keep
    # their existing path; this default-off experiment changes activations.
    if (
        _EXPERIMENTAL_MMQ
        and m >= 128
        and (n, k) in _BATCHED_GEMV_SHAPES
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_cuda
        and torch.cuda.get_device_capability(x.device) == (8, 6)
    ):
        from .pq2_mmq import pq2_mmq

        if _MMQ_ACTIVATION_GROUP == 128:
            return pq2_mmq(x, weight, prepared=prepared)
        return pq2_mmq(
            x, weight, prepared=prepared, activation_group=_MMQ_ACTIVATION_GROUP
        )
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m and n and m <= 4 and k <= 32768:
        _pq2_gemv[(n, m)](
            x, weight, y, n, k, weight.stride(0), triton.next_power_of_2(k), prepared
        )
    elif m and n:
        small_output = (
            (n, k) == (5120, 6144)
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.is_cuda
            and torch.cuda.get_device_capability(x.device) == (8, 6)
        )
        if small_output and m == 8:
            bm = _BATCH8_FLOAT_OUTPUT_BM
        elif small_output and _SMALL_FLOAT_OUTPUT and 5 <= m <= 16:
            bm = 8 if m <= 8 else 16
        else:
            bm = 64
        _pq2[(triton.cdiv(m, bm), triton.cdiv(n, 64))](
            x,
            weight,
            y,
            m,
            n,
            k,
            weight.stride(0),
            bm,
            64,
            128,
            prepared,
            num_stages=2 if prepared and x.dtype == torch.float32 else 3,
        )
    return y


@triton.jit
def _pq2_gemv(
    X,
    W,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    B: tl.constexpr,
    PREPARED: tl.constexpr = False,
):
    row = tl.program_id(0)
    token = tl.program_id(1)
    k = tl.arange(0, B)
    scale = _pq2_scale(W, row, k // 128, N, K, STRIDE, k < K, PREPARED)
    packed = _pq2_codes(W, row, k // 128, (k % 128) // 4, K, STRIDE, k < K, PREPARED)
    code = ((packed.to(tl.int32) >> (2 * (k % 4))) & 3) - 1
    a = tl.load(X + token * K + k, k < K, other=0)
    # Match the activation dtype used by the dense fallback.
    b = (code.to(tl.float32) * scale).to(a.dtype).to(tl.float32)
    value = tl.sum(a.to(tl.float32) * b, 0)
    tl.store(Y + token * N + row, value)
