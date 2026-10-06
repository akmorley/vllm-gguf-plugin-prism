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

_VARIANT = os.environ.get("GGUF_PQ2_INT_GEMV_VARIANT", "shape-tuned-v2")
_DECODE = os.environ.get("GGUF_PQ2_INT_GEMV_DECODE", "prmt")
_CHAINED = os.environ.get("GGUF_PQ2_INT_GEMV_CHAINED", "0") == "1"
_SINGLE_GEMV = os.environ.get("GGUF_PQ2_SINGLE_GEMV", "1") == "1"
_BATCH8_CHAINED = os.environ.get("GGUF_PQ2_BATCH8_CHAINED", "1") == "1"


@triton.jit
def _decode_codes(packed, PRMT: tl.constexpr):
    if PRMT:
        selector = (packed & 3) | ((packed >> 2 & 3) << 4)
        selector |= ((packed >> 4 & 3) << 8) | ((packed >> 6 & 3) << 12)
        return tl.inline_asm_elementwise(
            "prmt.b32 $0, 0x020100ff, 0x020100ff, $1;",
            constraints="=r,r", args=[selector], dtype=tl.int32,
            is_pure=True, pack=1,
        )
    spread = (packed & 3) | ((packed << 6) & 0x300)
    spread |= ((packed << 12) & 0x30000) | ((packed << 18) & 0x3000000)
    return ((spread.to(tl.uint32) + 0x7F7F7F7F) ^ 0x80808080).to(tl.int32)


@triton.jit
def _dot_groups(Q, W, tokens, rows, block, M, N: tl.constexpr, K: tl.constexpr,
                STRIDE: tl.constexpr, PREPARED: tl.constexpr,
                PRMT: tl.constexpr, CHAINED: tl.constexpr):
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
        w = _decode_codes(packed, PRMT)
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
    PRMT: tl.constexpr = False,
    CHAINED: tl.constexpr = False,
    GATED: tl.constexpr = False,
):
    tokens = tl.arange(0, BT)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BC)
    acc = tl.full((BT, BN, BC), 0, tl.float32)
    if GATED:
        acc_up = tl.full((BT, BN, BC), 0, tl.float32)
    for base in range(tl.cdiv(K // 128, BC)):
        block = base * BC + offsets
        valid = block < K // 128
        mask = (rows[:, None] < (N // 2 if GATED else N)) & valid[None, :]
        ws = _pq2_scale(W, rows[:, None], block[None, :], N, K, STRIDE, mask, PREPARED)
        partial = _dot_groups(Q, W, tokens, rows, block, M, N, K, STRIDE,
                              PREPARED, PRMT, CHAINED)
        xs = tl.load(
            S + tokens[:, None] * (K // 128) + block[None, :],
            (tokens[:, None] < M) & valid[None, :],
            other=0,
        )
        acc += partial * xs[:, None, :] * ws[None, :, :]
        if GATED:
            up_rows = rows + N // 2
            up_ws = _pq2_scale(W, up_rows[:, None], block[None, :], N, K, STRIDE, mask, PREPARED)
            up_partial = _dot_groups(Q, W, tokens, up_rows, block, M, N, K, STRIDE,
                                     PREPARED, PRMT, CHAINED)
            acc_up += up_partial * xs[:, None, :] * up_ws[None, :, :]
    values = tl.sum(acc, 2)
    output_n = N // 2 if GATED else N
    if GATED:
        # Preserve the separately materialized projection's dtype conversion.
        gate = values.to(Y.dtype.element_ty).to(tl.float32)
        up = tl.sum(acc_up, 2).to(Y.dtype.element_ty).to(tl.float32)
        values = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(
        Y + tokens[:, None] * output_n + rows[None, :],
        values,
        (tokens[:, None] < M) & (rows[None, :] < output_n),
    )


def pq2_int_gemv(x, weight, *, prepared=False, variant=None, decode=None,
                 chained=None, gated=False, quantized=None):
    """Quantize once, then reuse packed words across up to sixteen tokens.

    Scratch: M*K INT8 bytes and M*K/128 FP32 scales. Activation approximation
    requires model/quality acceptance before default serving adoption.
    """
    variant = _VARIANT if variant is None else variant
    if variant not in ("default", "row1-k32", "shape-tuned", "shape-tuned-v2"):
        raise ValueError("Integer GEMV variant must be default, row1-k32, shape-tuned or shape-tuned-v2")
    decode = _DECODE if decode is None else decode
    if chained is None:
        chained = _CHAINED or (
            _BATCH8_CHAINED and x.ndim == 2 and x.shape[0] == 8 and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and torch.cuda.get_device_capability(x.device) == (8, 6)
        )
    if decode not in ("shift", "prmt"):
        raise ValueError("Integer GEMV decode must be shift or prmt")
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
    # The single-token experiment trades row reuse for K parallelism.
    bn, bc = (1, 32) if variant == "row1-k32" and m == 1 else (16, 4)
    if variant in ("shape-tuned", "shape-tuned-v2") and m == 1:
        if (n, k) == (34816, 5120):
            bn, bc = 32, 8
        elif (n, k) in ((5120, 17408), (5120, 6144)):
            bn, bc = 16, 16
    if variant == "shape-tuned-v2" and m == 1:
        if (n, k) in ((16384, 5120), (14336, 5120)):
            bn, bc = 16, 8
        elif (n, k) == (248320, 5120):
            bn, bc = 64, 4
    if gated and n % 2:
        raise ValueError("Fused gate requires an even number of weight rows")
    output_n = n // 2 if gated else n
    y = torch.empty((m, output_n), dtype=x.dtype, device=x.device)
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
        if (_SINGLE_GEMV and m == 1 and prepared and decode == "prmt"
                and not chained and not gated
                and x.dtype in (torch.bfloat16, torch.float16)
                and torch.cuda.get_device_capability(x.device) == (8, 6)
                and (n, k) in ((34816, 5120), (5120, 17408), (5120, 6144),
                               (16384, 5120), (14336, 5120), (248320, 5120))):
            from .pq2_single_gemv import launch_single
            launch_single(q, scales, weight, y, n, k, bn, bc)
            return y
        _int_gemv[(triton.cdiv(output_n, bn),)](
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
            decode == "prmt",
            chained,
            gated,
            num_warps=4,
        )
    return y
