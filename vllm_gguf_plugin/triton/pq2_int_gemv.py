# SPDX-License-Identifier: Apache-2.0
"""Experimental PQ2 x Q8 decode with packed signed INT8 DP4A operations.

Independent implementation; activation quantization matches our MMQ prototype,
not a claim of llama.cpp Q8_1 equivalence. Serving use remains opt-in.
"""

import torch
import triton
import triton.language as tl

from .pq2_layout import _pq2_codes, _pq2_scale, validate_prepared
from .pq2_mmq import _quantize


@triton.jit(do_not_specialize=["M"])
def _int_gemv(
    Q,
    S,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
    PREPARED: tl.constexpr = False,
):
    tokens = tl.arange(0, BT)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BC)
    groups = tl.arange(0, 32)
    acc = tl.full((BT, BN, BC), 0, tl.float32)
    for base in range(tl.cdiv(K // 128, BC)):
        block = base * BC + offsets
        valid = block < K // 128
        # Each INT32 activation word holds four signed Q8 values.
        q = tl.load(
            Q
            + tokens[:, None, None] * (K // 4)
            + block[None, :, None] * 32
            + groups[None, None, :],
            (tokens[:, None, None] < M) & valid[None, :, None],
            other=0,
        )
        mask = (rows[:, None] < N) & valid[None, :]
        ws = _pq2_scale(W, rows[:, None], block[None, :], N, K, STRIDE, mask, PREPARED)
        packed = _pq2_codes(
            W,
            rows[:, None, None],
            block[None, :, None],
            groups[None, None, :],
            K,
            STRIDE,
            mask[:, :, None],
            PREPARED,
        ).to(tl.int32)
        # Spread four 2-bit codes into bytes before converting 0/1/2/3 to
        # signed -1/0/1/2. Each byte + 127 is <= 130, so no carry crosses
        # a byte boundary. Unsigned arithmetic also handles the high byte.
        spread = (packed & 3) | ((packed << 6) & 0x300)
        spread |= ((packed << 12) & 0x30000) | ((packed << 18) & 0x3000000)
        w = ((spread.to(tl.uint32) + 0x7F7F7F7F) ^ 0x80808080).to(tl.int32)
        dots = tl.inline_asm_elementwise(
            "dp4a.s32.s32 $0, $1, $2, 0;",
            constraints="=r,r,r",
            args=[q[:, None, :, :], w[None, :, :, :]],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        # Integer reduction stops at each independently scaled 128-wide block.
        partial = tl.sum(dots, 3).to(tl.float32)
        xs = tl.load(
            S + tokens[:, None] * (K // 128) + block[None, :],
            (tokens[:, None] < M) & valid[None, :],
            other=0,
        )
        acc += partial * xs[:, None, :] * ws[None, :, :]
    values = tl.sum(acc, 2)
    tl.store(
        Y + tokens[:, None] * N + rows[None, :],
        values,
        (tokens[:, None] < M) & (rows[None, :] < N),
    )


def pq2_int_gemv(x, weight, *, prepared=False):
    """Quantize once, then reuse packed words across up to sixteen tokens.

    Scratch: M*K INT8 bytes and M*K/128 FP32 scales. Activation approximation
    requires model/quality acceptance before default serving adoption.
    """
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("PQ2 integer GEMV requires two matrices")
    m, k = x.shape
    n = weight.shape[0]
    if not 1 <= m <= 16:
        raise ValueError("PQ2 integer GEMV requires 1 to 16 tokens")
    if k == 0 or k % 128 or weight.shape[1] != k // 128 * 34:
        raise ValueError("Invalid PQ2 packed matrix shape")
    if not x.is_cuda or weight.device != x.device:
        raise ValueError("PQ2 integer GEMV requires inputs on the same CUDA device")
    if (
        x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or weight.dtype != torch.uint8
    ):
        raise ValueError(
            "PQ2 integer GEMV requires floating activations and uint8 weights"
        )
    x = x.contiguous()
    if prepared:
        validate_prepared(weight)
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if n:
        q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        scales = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
        _quantize[(m * (k // 128),)](x, q, scales, k)
        _int_gemv[(triton.cdiv(n, 16),)](
            q.view(torch.int32),
            scales,
            weight,
            y,
            m,
            n,
            k,
            weight.stride(0),
            triton.next_power_of_2(m),
            16,
            4,
            prepared,
            num_warps=4,
        )
    return y
