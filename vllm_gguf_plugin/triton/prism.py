# SPDX-License-Identifier: Apache-2.0
"""Prism kernels: packed PQ2 multiplication and normalized block FWHT."""

import torch
import triton
import triton.language as tl


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


@triton.jit
def _pq2(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
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
        block = k // 128
        address = n[None, :] * STRIDE + block[:, None] * 34
        lo = tl.load(W + address, (n[None, :] < N) & (k[:, None] < K), other=0).to(
            tl.uint16
        )
        hi = tl.load(W + address + 1, (n[None, :] < N) & (k[:, None] < K), other=0).to(
            tl.uint16
        )
        scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        packed = tl.load(
            W + address + 2 + (k[:, None] % 128) // 4,
            (n[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        code = ((packed.to(tl.int32) >> (2 * (k[:, None] % 4))) & 3) - 1
        b = (code.to(tl.float32) * scale).to(a.dtype)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def pq2_matmul(x, weight):
    """Dequantize only a tile at a time, never materializing a dense weight."""
    m, k = x.shape
    n = weight.shape[0]
    if k % 128 or weight.shape[1] != k // 128 * 34:
        raise ValueError("Invalid PQ2 packed matrix shape")
    x = x.contiguous()
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m and n and m <= 4 and k <= 32768:
        _pq2_gemv[(n, m)](
            x, weight, y, n, k, weight.stride(0), triton.next_power_of_2(k)
        )
    elif m and n:
        _pq2[(triton.cdiv(m, 64), triton.cdiv(n, 64))](
            x, weight, y, m, n, k, weight.stride(0), 64, 64, 128
        )
    return y


@triton.jit
def _pq2_gemv(
    X, W, Y, N: tl.constexpr, K: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    token = tl.program_id(1)
    k = tl.arange(0, B)
    address = row * STRIDE + (k // 128) * 34
    lo = tl.load(W + address, k < K, other=0).to(tl.uint16)
    hi = tl.load(W + address + 1, k < K, other=0).to(tl.uint16)
    scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    packed = tl.load(W + address + 2 + (k % 128) // 4, k < K, other=0)
    code = ((packed.to(tl.int32) >> (2 * (k % 4))) & 3) - 1
    a = tl.load(X + token * K + k, k < K, other=0)
    # Match the activation dtype used by the dense fallback.
    b = (code.to(tl.float32) * scale).to(a.dtype).to(tl.float32)
    value = tl.sum(a.to(tl.float32) * b, 0)
    tl.store(Y + token * N + row, value)
