# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only PQ2 loading/unpacking experiments; no serving dispatch."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .pq2_mmq import _quantize


@triton.jit
def _expand_byte(p, FAST: tl.constexpr):
    if FAST:
        # Each byte is 0..3. Adding 127 cannot carry between bytes; XOR maps
        # 127/128/129/130 to signed -1/0/1/2 without four separate subtractions.
        spread = (p & 3) | ((p << 6) & 0x300)
        spread |= ((p << 12) & 0x30000) | ((p << 18) & 0x3000000)
        word = ((spread.to(tl.uint32) + 0x7F7F7F7F) ^ 0x80808080).to(tl.int32)
    else:
        c0 = ((p & 3) - 1) & 255
        c1 = (((p >> 2) & 3) - 1) & 255
        c2 = (((p >> 4) & 3) - 1) & 255
        c3 = (((p >> 6) & 3) - 1) & 255
        word = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24)
    return word


@triton.jit(do_not_specialize=["M"])
def _packed_gemv(
    Q,
    S,
    W,
    WS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
    LOAD: tl.constexpr,
    SEPARATE: tl.constexpr,
    FAST: tl.constexpr,
    STAGES: tl.constexpr,
    UNROLL: tl.constexpr,
    CACHE: tl.constexpr,
    KEEP_LANES: tl.constexpr,
):
    tokens = tl.arange(0, BT)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BC)
    DOT_BYTES: tl.constexpr = 1 if KEEP_LANES else LOAD
    groups = tl.arange(0, 32 // DOT_BYTES)
    acc = tl.full((BT, BN, BC), 0, tl.float32)
    for base in tl.range(
        tl.cdiv(K // 128, BC), num_stages=STAGES, loop_unroll_factor=UNROLL
    ):
        block = base * BC + offsets
        valid = block < K // 128
        mask = (rows[:, None] < N) & valid[None, :]
        if SEPARATE:
            ws = tl.load(
                WS + rows[:, None] * (K // 128) + block[None, :], mask, other=0
            ).to(tl.float32)
            address = rows[:, None] * STRIDE + block[None, :] * (32 // LOAD)
            code_offset: tl.constexpr = 0
        else:
            address = rows[:, None] * STRIDE + block[None, :] * (34 // LOAD)
            if LOAD == 1:
                lo = tl.load(W + address, mask, other=0).to(tl.uint16)
                hi = tl.load(W + address + 1, mask, other=0).to(tl.uint16)
                ws = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            else:
                ws = (
                    tl.load(W + address, mask, other=0)
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
            code_offset: tl.constexpr = 2 // LOAD
        indices = groups // LOAD if KEEP_LANES else groups
        packed = tl.load(
            W + address[:, :, None] + code_offset + indices[None, None, :],
            mask[:, :, None],
            other=0,
            cache_modifier=CACHE,
        ).to(tl.uint32)
        dots = tl.full((BT, BN, BC, 32 // DOT_BYTES), 0, tl.int32)
        for byte in tl.static_range(DOT_BYTES):
            shift = 8 * (groups[None, None, :] % LOAD) if KEEP_LANES else 8 * byte
            p = ((packed >> shift) & 255).to(tl.int32)
            word = _expand_byte(p, FAST)
            q = tl.load(
                Q
                + tokens[:, None, None] * (K // 4)
                + block[None, :, None] * 32
                + groups[None, None, :] * DOT_BYTES
                + byte,
                (tokens[:, None, None] < M) & valid[None, :, None],
                other=0,
            )
            dots += tl.inline_asm_elementwise(
                "dp4a.s32.s32 $0, $1, $2, 0;",
                constraints="=r,r,r",
                args=[q[:, None, :, :], word[None]],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
        partial = tl.sum(dots, 3).to(tl.float32)
        xs = tl.load(
            S + tokens[:, None] * (K // 128) + block[None, :],
            (tokens[:, None] < M) & valid[None, :],
            other=0,
        )
        acc += partial * xs[:, None, :] * ws[None]
    tl.store(
        Y + tokens[:, None] * N + rows[None, :],
        tl.sum(acc, 2),
        (tokens[:, None] < M) & (rows[None, :] < N),
    )


@dataclass(frozen=True)
class PreparedPQ2:
    codes: torch.Tensor
    scales: torch.Tensor
    k: int
    load_bytes: int
    separate: bool

    @property
    def storage_bytes(self):
        """Logical payload, excluding raw row-stride padding/allocator overhead."""
        return self.codes.numel() * self.codes.element_size() + (
            self.scales.numel() * self.scales.element_size() if self.separate else 0
        )


def prepare_pq2(weight, *, load_bytes=1, separate=False):
    """Prepare reusable buffers. Separate layout allocates a new packed copy.

    Raw 16-bit loads require an even base offset and row stride. No dense
    dequantization; separate buffers still use exactly 34 bytes per block.
    """
    if (
        weight.ndim != 2
        or weight.dtype != torch.uint8
        or not weight.is_cuda
        or weight.shape[1] == 0
        or weight.shape[1] % 34
    ):
        raise ValueError("Expected a CUDA uint8 PQ2 matrix")
    if load_bytes not in (1, 2, 4) or (not separate and load_bytes == 4):
        raise ValueError("Raw PQ2 supports 1/2-byte loads; separate supports 1/2/4")
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    n, width = weight.shape
    k = width // 34 * 128
    if separate:
        blocks = weight.view(n, width // 34, 34)
        scales = (
            blocks[:, :, :2]
            .clone(memory_format=torch.contiguous_format)
            .view(torch.float16)
            .reshape(n, width // 34)
        )
        codes = (
            blocks[:, :, 2:]
            .clone(memory_format=torch.contiguous_format)
            .reshape(n, width // 34 * 32)
        )
    else:
        codes = weight
        scales = weight  # Unused pointer for this specialization; no extra storage.
        if load_bytes == 2 and (weight.data_ptr() % 2 or weight.stride(0) % 2):
            raise ValueError("Raw 16-bit loads require an even base and row stride")
    dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[load_bytes]
    return PreparedPQ2(codes.view(dtype), scales, k, load_bytes, separate)


def launch_packed(
    q,
    scales,
    prepared,
    output,
    m,
    *,
    tile=(16, 4, 4),
    fast_pack=False,
    stages=1,
    unroll=1,
    cache="",
    keep_lanes=False,
):
    bn, bc, warps = tile
    n = prepared.codes.shape[0]
    return _packed_gemv[(triton.cdiv(n, bn),)](
        q.view(torch.int32),
        scales,
        prepared.codes,
        prepared.scales,
        output,
        m,
        n,
        prepared.k,
        prepared.codes.stride(0),
        triton.next_power_of_2(m),
        bn,
        bc,
        prepared.load_bytes,
        prepared.separate,
        fast_pack,
        stages,
        unroll,
        cache,
        keep_lanes,
        num_warps=warps,
    )


def packed_gemv(x, prepared, **config):
    if (
        x.ndim != 2
        or not 1 <= x.shape[0] <= 16
        or x.shape[1] != prepared.k
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not x.is_cuda
        or x.device != prepared.codes.device
    ):
        raise ValueError("Expected matching CUDA floating activations, M=1..16")
    x = x.contiguous()
    m, k = x.shape
    output = torch.empty((m, prepared.codes.shape[0]), dtype=x.dtype, device=x.device)
    if output.numel():
        q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        scales = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
        _quantize[(m * (k // 128),)](x, q, scales, k)
        launch_packed(q, scales, prepared, output, m, **config)
    return output
