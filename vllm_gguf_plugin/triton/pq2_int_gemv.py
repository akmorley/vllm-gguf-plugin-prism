# SPDX-License-Identifier: Apache-2.0
"""Experimental PQ2 x Q8 decode with packed signed INT8 DP4A operations.

Independent implementation; activation quantization matches our MMQ prototype,
not a claim of llama.cpp Q8_1 equivalence. Serving use remains opt-in.
"""

import os

import torch
import triton
import triton.language as tl

from . import pq2_config as _config
from . import pq2_tables as _tables
from .pq2_layout import _pq2_codes, _pq2_scale, validate_prepared
from .pq2_mmq import _quantize

_SINGLE_GEMV = os.environ.get("GGUF_PQ2_SINGLE_GEMV", "1") == "1"
# Single-token hand-written CUDA kernel (csrc/pq2/pq2_gemv_m1.cu): DP4A on the raw 2-bit planes, no PRMT decode. Same
# integer products as the Triton kernel; FP32 group accumulation order differs (relative RMS ~1e-4, deterministic).
# (N, K) -> kernel config. GGUF_PQ2_M1_CUDA=0 restores the Triton `_single` path.
_M1_CUDA = _config.flag("GGUF_PQ2_M1_CUDA", "numerics_class")
_M1_CFG = _tables.SM86.m1_cfg  # per-architecture tables: pq2_tables.py
# Triton port of the same plane-DP4A algorithm (pq2_single_gemv._single_planes), preferred over the CUDA kernel for all six
# shapes: -4.3% matmul time per step in serving (nsys), +2.9% tok/s at 1 request, GSM8K answers unchanged. In the `tuned`
# kernel set; GGUF_PQ2_M1_TRITON=0 disables. Same integer products; FP32 group order differs (relative RMS <= 1.6e-4 vs CUDA).
_M1_TRITON = _config.flag("GGUF_PQ2_M1_TRITON", "numerics_class")
_M1_TRITON_CFG = _tables.SM86.m1_triton


def _m1_available():
    try:
        from .. import _C_gguf  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops, "_C_gguf") and hasattr(torch.ops._C_gguf, "pq2_gemv_m1")


def _gemv_m1(x, weight, cfg):
    from .pq2_small_mmq import _quantize_planes_sum

    k = x.shape[1]
    n = weight.shape[0]
    groups = k // 128
    q = torch.empty((1, k), dtype=torch.int8, device=x.device)
    s = torch.empty((1, groups), dtype=torch.float32, device=x.device)
    t = torch.empty((1, groups), dtype=torch.int32, device=x.device)
    _quantize_planes_sum[(groups,)](x, q, s, t, k)
    y = torch.empty((1, n), dtype=x.dtype, device=x.device)
    torch.ops._C_gguf.pq2_gemv_m1(q, s, t, weight, y, cfg)
    return y


def _gemv_m1_triton(x, weight, cfg):
    from .pq2_single_gemv import launch_planes
    from .pq2_small_mmq import _quantize_planes_sum

    k = x.shape[1]
    n = weight.shape[0]
    groups = k // 128
    q = torch.empty((1, k), dtype=torch.int8, device=x.device)
    s = torch.empty((1, groups), dtype=torch.float32, device=x.device)
    t = torch.empty((1, groups), dtype=torch.int32, device=x.device)
    _quantize_planes_sum[(groups,)](x, q, s, t, k)
    y = torch.empty((1, n), dtype=x.dtype, device=x.device)
    launch_planes(q, s, t, weight, y, n, k, *cfg)
    return y


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
    if (_M1_TRITON and m == 1 and prepared and quantized is None and not chained and n
            and x.dtype in (torch.bfloat16, torch.float16) and (n, k) in _M1_TRITON_CFG and x.is_cuda
            and torch.cuda.get_device_capability(x.device) == (8, 6)):
        return _gemv_m1_triton(x, weight, _M1_TRITON_CFG[(n, k)])
    if (_M1_CUDA and m == 1 and prepared and quantized is None and not chained and n
            and x.dtype in (torch.bfloat16, torch.float16) and k % 512 == 0
            and (n, k) in _M1_CFG and x.is_cuda
            and torch.cuda.get_device_capability(x.device) == (8, 6) and _m1_available()):
        return _gemv_m1(x, weight, _M1_CFG[(n, k)])
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
