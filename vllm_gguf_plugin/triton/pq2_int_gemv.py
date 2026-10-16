# SPDX-License-Identifier: Apache-2.0
"""Experimental PQ2 x Q8 decode with packed signed INT8 DP4A operations.

Independent implementation; activation quantization matches our MMQ prototype,
not a claim of llama.cpp Q8_1 equivalence. Serving use remains opt-in.
"""

import os

import torch
import triton
import triton.language as tl

from .pq2_layout import _pq2_codes, _pq2_scale, validate_prepared
from .pq2_mmq import _quantize

_SINGLE_GEMV = os.environ.get("GGUF_PQ2_SINGLE_GEMV", "1") == "1"
_BATCH8_CHAINED = os.environ.get("GGUF_PQ2_BATCH8_CHAINED", "1") == "1"


@triton.jit
def _decode_codes(packed):
    # PRMT lookup: four 2-bit codes -> four signed bytes (0 -> -1, 1 -> 0, 2 -> 1, 3 -> 2).
    selector = (packed & 3) | ((packed >> 2 & 3) << 4)
    selector |= ((packed >> 4 & 3) << 8) | ((packed >> 6 & 3) << 12)
    return tl.inline_asm_elementwise(
        "prmt.b32 $0, 0x020100ff, 0x020100ff, $1;",
        constraints="=r,r", args=[selector], dtype=tl.int32,
        is_pure=True, pack=1,
    )


@triton.jit
def _dot_groups(Q, W, tokens, rows, block, M, N: tl.constexpr, K: tl.constexpr,
                STRIDE: tl.constexpr, PREPARED: tl.constexpr, CHAINED: tl.constexpr):
    valid = block < K // 128
    mask = (rows[:, None] < N) & valid[None, :]
    words = tl.arange(0, 4 if CHAINED else 32)
    dots = tl.full((tokens.shape[0], rows.shape[0], block.shape[0], words.shape[0]), 0, tl.int32)
    for j in tl.static_range(8 if CHAINED else 1):
        groups = words + j * 4
        q = tl.load(Q + tokens[:, None, None] * (K // 4)
                    + block[None, :, None] * 32 + groups[None, None, :],
                    (tokens[:, None, None] < M) & valid[None, :, None], other=0)
        packed = _pq2_codes(W, rows[:, None, None], block[None, :, None],
                            groups[None, None, :], K, STRIDE,
                            mask[:, :, None], PREPARED).to(tl.int32)
        w = _decode_codes(packed)
        dots = tl.inline_asm_elementwise(
            "dp4a.s32.s32 $0, $1, $2, $3;", constraints="=r,r,r,r",
            args=[q[:, None, :, :], w[None, :, :, :], dots],
            dtype=tl.int32, is_pure=True, pack=1,
        )
    return tl.sum(dots, 3).to(tl.float32)


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
    CHAINED: tl.constexpr = False,
):
    tokens = tl.arange(0, BT)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BC)
    acc = tl.full((BT, BN, BC), 0, tl.float32)
    for base in range(tl.cdiv(K // 128, BC)):
        block = base * BC + offsets
        valid = block < K // 128
        mask = (rows[:, None] < N) & valid[None, :]
        ws = _pq2_scale(W, rows[:, None], block[None, :], N, K, STRIDE, mask, PREPARED)
        partial = _dot_groups(Q, W, tokens, rows, block, M, N, K, STRIDE,
                              PREPARED, CHAINED)
        xs = tl.load(
            S + tokens[:, None] * (K // 128) + block[None, :],
            (tokens[:, None] < M) & valid[None, :],
            other=0,
        )
        acc += partial * xs[:, None, :] * ws[None, :, :]
    tl.store(
        Y + tokens[:, None] * N + rows[None, :],
        tl.sum(acc, 2),
        (tokens[:, None] < M) & (rows[None, :] < N),
    )


# Single-token (M = 1) tiles per (N, K): rows per program, 128-groups per iteration.
_SINGLE_TILES = {
    (34816, 5120): (32, 8),
    (5120, 17408): (16, 16),
    (5120, 6144): (16, 16),
    (16384, 5120): (16, 8),
    (14336, 5120): (16, 8),
    (248320, 5120): (64, 4),
}


def pq2_int_gemv(x, weight, *, prepared=False, chained=None, quantized=None):
    """Quantize once, then reuse packed words across up to sixteen tokens.

    Scratch: M*K INT8 bytes and M*K/128 FP32 scales. Activation approximation
    requires model/quality acceptance before default serving adoption.
    """
    if chained is None:
        chained = (
            _BATCH8_CHAINED and x.ndim == 2 and x.shape[0] == 8 and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and torch.cuda.get_device_capability(x.device) == (8, 6)
        )
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
    bn, bc = _SINGLE_TILES.get((n, k), (16, 4)) if m == 1 else (16, 4)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if n:
        if quantized is None:
            q = torch.empty((m, k), dtype=torch.int8, device=x.device)
            scales = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
            _quantize[(m * (k // 128),)](x, q, scales, k)
        else:
            q, scales = quantized
            if (q.shape != (m, k) or q.dtype != torch.int8 or q.device != x.device
                    or not q.is_contiguous() or scales.shape != (m, k // 128)
                    or scales.dtype != torch.float32 or scales.device != x.device
                    or not scales.is_contiguous()):
                raise ValueError("Invalid prequantized activations")
        if (_SINGLE_GEMV and m == 1 and prepared and not chained
                and x.dtype in (torch.bfloat16, torch.float16)
                and torch.cuda.get_device_capability(x.device) == (8, 6)
                and (n, k) in _SINGLE_TILES):
            from .pq2_single_gemv import launch_single
            launch_single(q, scales, weight, y, n, k, bn, bc)
            return y
        _int_gemv[(triton.cdiv(n, bn),)](
            q.view(torch.int32),
            scales,
            weight,
            y,
            m,
            n,
            k,
            weight.stride(0),
            triton.next_power_of_2(m),
            bn,
            bc,
            prepared,
            chained,
            num_warps=4,
        )
    return y
