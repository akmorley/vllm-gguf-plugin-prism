# SPDX-License-Identifier: Apache-2.0
"""One-allocation PQ2 code/scale planes; the shape stays (N, K/128*34)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pq2_scale(
    W,
    row,
    block,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    mask,
    PREPARED: tl.constexpr,
):
    if PREPARED:
        # All code bytes precede all FP16 scales in the same allocation.
        half = W.to(tl.pointer_type(tl.float16))
        return tl.load(
            half + N * (K // 8) + row * (K // 128) + block, mask, other=0
        ).to(tl.float32)
    address = row * STRIDE + block * 34
    lo = tl.load(W + address, mask, other=0).to(tl.uint16)
    hi = tl.load(W + address + 1, mask, other=0).to(tl.uint16)
    return (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _pq2_codes(
    W,
    row,
    block,
    byte,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    mask,
    PREPARED: tl.constexpr,
):
    if PREPARED:
        address = row * (K // 4) + block * 32 + byte
    else:
        address = row * STRIDE + block * 34 + 2 + byte
    return tl.load(W + address, mask, other=0)


@triton.jit
def _prepare(
    W,
    P,
    N: tl.constexpr,
    BLOCKS: tl.constexpr,
    STRIDE: tl.constexpr,
    INNER: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    code_bytes = N * BLOCKS * 32
    total = N * BLOCKS * 34
    is_code = i < code_bytes
    code_index = i
    scale_index = i - code_bytes
    row = tl.where(is_code, code_index // (BLOCKS * 32), scale_index // (BLOCKS * 2))
    block = tl.where(is_code, (code_index // 32) % BLOCKS, (scale_index // 2) % BLOCKS)
    byte = tl.where(is_code, 2 + code_index % 32, scale_index % 2)
    value = tl.load(W + row * STRIDE + (block * 34 + byte) * INNER, i < total, other=0)
    tl.store(P + i, value, i < total)


def prepare_pq2_layout(weight):
    """Reorder raw blocks once, retaining exactly one packed allocation.

    The caller replaces its parameter data and releases source references.
    Preparation is forbidden during graph capture and is never a matmul step.
    """
    if weight.ndim != 2 or weight.dtype != torch.uint8 or not weight.is_cuda:
        raise ValueError("PQ2 preparation requires a CUDA uint8 matrix")
    if not weight.shape[1] or weight.shape[1] % 34:
        raise ValueError("Invalid PQ2 packed matrix shape")
    with torch.cuda.device(weight.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare PQ2 weights before CUDA graph capture")
    n, width = weight.shape
    result = torch.empty((n, width), dtype=torch.uint8, device=weight.device)
    if n:
        _prepare[(triton.cdiv(result.numel(), 512),)](
            weight, result, n, width // 34, weight.stride(0), weight.stride(1), 512
        )
    return result


def validate_prepared(weight):
    if (
        weight.ndim != 2
        or weight.dtype != torch.uint8
        or not weight.is_cuda
        or not weight.is_contiguous()
        or weight.storage_offset() != 0
        or weight.untyped_storage().nbytes() != weight.numel()
    ):
        raise ValueError(
            "Prepared PQ2 planes require a complete contiguous CUDA allocation"
        )


@triton.jit
def _embedding(IDS, W, Y, N: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    k = tl.program_id(1) * B + tl.arange(0, B)
    row = tl.load(IDS + token)
    valid_row = (row >= 0) & (row < N)
    tl.device_assert(valid_row, "PQ2 embedding index out of range")
    mask = (k < K) & valid_row
    scale = _pq2_scale(W, row, k // 128, N, K, 0, mask, True)
    packed = _pq2_codes(W, row, k // 128, (k % 128) // 4, K, 0, mask, True)
    code = ((packed.to(tl.int32) >> (2 * (k % 4))) & 3) - 1
    tl.store(Y + token * K + k, code.to(tl.float32) * scale, mask)


def pq2_prepared_embedding(indices, weight, hidden_size, dtype):
    validate_prepared(weight)
    if indices.device != weight.device or indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "PQ2 embedding indices must be CUDA integers on the weight device"
        )
    if hidden_size % 128 or weight.shape[1] != hidden_size // 128 * 34:
        raise ValueError("Invalid prepared PQ2 embedding width")
    indices = indices.contiguous()
    out = torch.empty((*indices.shape, hidden_size), dtype=dtype, device=indices.device)
    if indices.numel():
        _embedding[(indices.numel(), triton.cdiv(hidden_size, 256))](
            indices,
            weight,
            out,
            weight.shape[0],
            hidden_size,
            256,
            debug=True,
        )
    return out
