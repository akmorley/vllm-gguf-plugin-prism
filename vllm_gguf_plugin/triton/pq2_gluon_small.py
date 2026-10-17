# SPDX-License-Identifier: Apache-2.0
"""Gluon port of csrc/pq2/pq2_mma_small.cu: direct-fragment INT8 tensor-core MMA on the raw 2-bit planes (M <= 8).

Each lane loads its 8 contiguous code bytes per row per 128-weight group (LDG.64) as two int32 words in the layout
`words`, whose byte view is the m16n8k32 A-operand layout up to a register renaming (no data movement, no shared
memory). Plane s of a packed register is (w >> 2s) & 0x03030303, one MMA k-step, computed on the packed 32-bit word by
inline PTX. Activations use the CUDA kernel's fragment order (pq2_small_mmq._quantize_fragments) and the epilogue is
the CUDA kernel's, so at split 1 the output is bit-identical to pq2_mma_small (any row tiling).
The next U groups' loads are issued before the current U groups are computed (loop-carried register prefetch,
last block peeled): ptxas otherwise interleaves them with the MMAs and too few DRAM requests stay in flight. split > 1 writes FP32
partials that pq2_small_mmq._reduce_splits sums in fixed order (deterministic).
"""
import functools
import math

import torch
import triton

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as ttgl
    from triton.experimental.gluon.language.nvidia.ampere import mma_v2
    AVAILABLE = True
except ImportError:  # Triton without Gluon's Ampere ops
    AVAILABLE = False

from .pq2_small_mmq import _quantize_fragments, _reduce_splits


if AVAILABLE:
    @functools.cache
    def layouts(warps, rt):
        bm = 16 * warps * rt
        mma = ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[warps, 1], instr_shape=[16, 8])
        wb = [[1 << i, 0, 0] for i in (4, 5, 6, 7) if (1 << i) < 16 * warps]
        rb = [[16 * warps << i, 0, 0] for i in range(int(math.log2(rt)))]
        words = ttgl.DistributedLinearLayout(
            reg_bases=[[0, 1, 0], [8, 0, 0]] + rb,
            lane_bases=[[0, 0, 1], [0, 0, 2], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
            warp_bases=wb, block_bases=[], shape=[bm, 2, 4])
        return bm, mma, words


    @gluon.jit
    def _bytes(p):
        b0 = (p & 0xFF).to(ttgl.int8)
        b1 = ((p >> 8) & 0xFF).to(ttgl.int8)
        b2 = ((p >> 16) & 0xFF).to(ttgl.int8)
        b3 = ((p >> 24) & 0xFF).to(ttgl.int8)
        x = ttgl.join(ttgl.join(b0, b2), ttgl.join(b1, b3))
        return ttgl.reshape(x, [x.shape[0], 32])


    @gluon.jit
    def _plane(a, S: ttgl.constexpr):
        # plane S of four packed code bytes in one 32-bit register: (w >> 2S) & 0x03030303 (pack=4: no byte split)
        if S == 0:
            return ttgl.inline_asm_elementwise("and.b32 $0, $1, 0x03030303;", "=r,r", [a], dtype=ttgl.int8,
                                               is_pure=True, pack=4)
        elif S == 1:
            return ttgl.inline_asm_elementwise("{ .reg .b32 t; shr.b32 t, $1, 2; and.b32 $0, t, 0x03030303; }", "=r,r",
                                               [a], dtype=ttgl.int8, is_pure=True, pack=4)
        elif S == 2:
            return ttgl.inline_asm_elementwise("{ .reg .b32 t; shr.b32 t, $1, 4; and.b32 $0, t, 0x03030303; }", "=r,r",
                                               [a], dtype=ttgl.int8, is_pure=True, pack=4)
        else:
            return ttgl.inline_asm_elementwise("{ .reg .b32 t; shr.b32 t, $1, 6; and.b32 $0, t, 0x03030303; }", "=r,r",
                                               [a], dtype=ttgl.int8, is_pure=True, pack=4)


    @gluon.jit
    def _load_block(wp, qp, qmask, g0, U: ttgl.constexpr):
        wt = ()
        bt = ()
        for u in ttgl.static_range(U):
            g = g0 + u
            wt = wt + (ttgl.load(wp + g * 8),)
            for s in ttgl.static_range(4):
                bt = bt + (ttgl.load(qp + g * 128 + s * 32, mask=qmask, other=0),)
        return wt, bt


    @gluon.jit
    def _consume(acc, wt, bt, g0, U: ttgl.constexpr, WS, S, T, rcc, ncc, n_ok, G: ttgl.constexpr, MMA: ttgl.constexpr,
                 A_L: ttgl.constexpr, TRIVIAL: ttgl.constexpr):
        for u in ttgl.static_range(U):
            g = g0 + u
            wb = ttgl.convert_layout(_bytes(wt[u]), A_L, assert_trivial=TRIVIAL)
            c = ttgl.zeros(acc.shape, ttgl.int32, layout=MMA)
            for s in ttgl.static_range(4):
                c = mma_v2(_plane(wb, s), bt[u * 4 + s], c)
            ws = ttgl.load(WS + rcc * G + g).to(ttgl.float32)
            xs = ttgl.load(S + ncc * G + g, mask=n_ok, other=0.0)
            qs = ttgl.load(T + ncc * G + g, mask=n_ok, other=0)
            acc += (c - qs[None, :]).to(ttgl.float32) * ws[:, None] * xs[None, :]
        return acc


    @gluon.jit
    def _gl_small(Q, S, T, W, WS, Y, M, N: ttgl.constexpr, K: ttgl.constexpr, BM: ttgl.constexpr, U: ttgl.constexpr,
                  MMA: ttgl.constexpr, WORDS: ttgl.constexpr, TRIVIAL: ttgl.constexpr,
                  SPLIT: ttgl.constexpr = 1):
        G: ttgl.constexpr = K // 128
        A_L: ttgl.constexpr = ttgl.DotOperandLayout(parent=MMA, operand_index=0, k_width=4)
        B_L: ttgl.constexpr = ttgl.DotOperandLayout(parent=MMA, operand_index=1, k_width=4)
        row0 = ttgl.program_id(0) * BM
        r = row0 + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, ttgl.SliceLayout(2, WORDS)))
        r = ttgl.minimum(r, N - 1)
        h = ttgl.arange(0, 2, layout=ttgl.SliceLayout(0, ttgl.SliceLayout(2, WORDS)))
        t = ttgl.arange(0, 4, layout=ttgl.SliceLayout(0, ttgl.SliceLayout(1, WORDS)))
        wp = W + r[:, None, None] * (K // 16) + 2 * t[None, None, :] + h[None, :, None]
        kb = ttgl.arange(0, 32, layout=ttgl.SliceLayout(1, B_L))
        nb = ttgl.arange(0, 8, layout=ttgl.SliceLayout(0, B_L))
        qp = Q + nb[None, :] * K + kb[:, None]
        qmask = (nb < M)[None, :] & (kb >= 0)[:, None]
        rc = row0 + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, MMA))
        nc = ttgl.arange(0, 8, layout=ttgl.SliceLayout(0, MMA))
        rcc = ttgl.minimum(rc, N - 1)
        n_ok = nc < M
        ncc = ttgl.where(n_ok, nc, 0)
        acc = ttgl.zeros([BM, 8], ttgl.float32, layout=MMA)
        PER: ttgl.constexpr = G // SPLIT
        first = ttgl.program_id(1) * PER
        # register prefetch: the next U groups are loaded before the current U groups are computed (loop-carried);
        # the last block is peeled so no prefetch index needs clamping (keeps address strength reduction)
        wt, bt = _load_block(wp, qp, qmask, first, U)
        for g0 in range(first, first + PER - U, U):
            nwt, nbt = _load_block(wp, qp, qmask, g0 + U, U)
            acc = _consume(acc, wt, bt, g0, U, WS, S, T, rcc, ncc, n_ok, G, MMA, A_L, TRIVIAL)
            wt = nwt
            bt = nbt
        acc = _consume(acc, wt, bt, first + PER - U, U, WS, S, T, rcc, ncc, n_ok, G, MMA, A_L, TRIVIAL)
        # SPLIT > 1: Y is an FP32 (SPLIT, M, N) workspace, summed in fixed order by pq2_small_mmq._reduce_splits
        ttgl.store(Y + ttgl.program_id(1) * M * N + nc[None, :] * N + rc[:, None], acc, mask=n_ok[None, :] & (rc < N)[:, None])


def pq2_gluon_small(x, weight, warps, rt, u, split=1):
    """x: (M <= 8, K) floating, already rotated; weight: prepared PQ2 planes."""
    m, k = x.shape
    n = weight.shape[0]
    G = k // 128
    if not 1 <= m <= 8 or k % 128 or weight.shape[1] != G * 34 or G % split or (G // split) % u:
        raise ValueError("Gluon PQ2 kernel requires 1 to 8 tokens, a PQ2 matrix and a K split dividing K/128")
    x = x.contiguous()
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m, G), dtype=torch.float32, device=x.device)
    t = torch.empty((m, G), dtype=torch.int32, device=x.device)
    _quantize_fragments[(m * G,)](x, q, s, t, k)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    flat = weight.reshape(-1)
    bm, mma, words = layouts(warps, rt)
    target = y if split == 1 else torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    _gl_small[(triton.cdiv(n, bm), split)](q, s, t, flat[: n * k // 4].view(torch.int32),
                                            flat[n * k // 4:].view(torch.float16), target, m, n, k, bm, u, mma, words,
                                            rt == 1, split, num_warps=warps)
    if split > 1:
        _reduce_splits[(triton.cdiv(m * n, 1024),)](target, y, m * n, split, 1024)
    return y
