# SPDX-License-Identifier: Apache-2.0
"""Experimental floating-point decode kernel with packed-weight batch reuse."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _batched_gemv(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    tokens = tl.arange(0, BT)
    offsets = tl.arange(0, BK)
    acc = tl.full((BT, BN, BK), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        k = base * BK + offsets
        address = rows[:, None] * STRIDE + (k[None, :] // 128) * 34
        mask = (rows[:, None] < N) & (k[None, :] < K)
        lo = tl.load(W + address, mask, other=0).to(tl.uint16)
        hi = tl.load(W + address + 1, mask, other=0).to(tl.uint16)
        scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        packed = tl.load(W + address + 2 + (k[None, :] % 128) // 4, mask, other=0).to(
            tl.int32
        )
        code = ((packed >> (2 * (k[None, :] % 4))) & 3) - 1
        a = tl.load(
            X + tokens[:, None] * K + k[None, :],
            (tokens[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = (code.to(tl.float32) * scale).to(a.dtype).to(tl.float32)
        acc += a.to(tl.float32)[:, None, :] * b[None, :, :]
    values = tl.sum(acc, 2)
    tl.store(
        Y + tokens[:, None] * N + rows[None, :],
        values,
        (tokens[:, None] < M) & (rows[None, :] < N),
    )


def pq2_batched_gemv(x, weight):
    """Reuse unpacked row tiles across up to sixteen tokens; no Q8 rounding."""
    m, k = x.shape
    n = weight.shape[0]
    if not 1 <= m <= 16:
        raise ValueError("Experimental batched GEMV requires 1 to 16 tokens")
    if k % 128 or weight.shape[1] != k // 128 * 34:
        raise ValueError("Invalid PQ2 packed matrix shape")
    x = x.contiguous()
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if n:
        _batched_gemv[(triton.cdiv(n, 4),)](
            x,
            weight,
            y,
            m,
            n,
            k,
            weight.stride(0),
            triton.next_power_of_2(m),
            4,
            256,
            num_warps=4,
        )
    return y
