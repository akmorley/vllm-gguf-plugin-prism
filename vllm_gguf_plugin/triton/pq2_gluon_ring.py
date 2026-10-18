# SPDX-License-Identifier: Apache-2.0
"""Gluon port of csrc/pq2/pq2_mma_ring.cu: small-batch (M <= 16) PQ2 x Q8 projection through a cp.async ring.

Each of the ST shared-memory stages holds SG 128-weight groups of BM weight rows (SG * 32 contiguous code bytes per row,
so every DRAM request is a >= 64 B row segment), their FP16 scales, and the 8- or 16-token activation tile (fragment order of
pq2_small_mmq._quantize_fragments) with its scales and sums. Fragments are read from shared memory in the register
layouts of pq2_gluon_small (weight words -> INT8 A operand by a register renaming; planes by inline PTX), INT8
mma.sync, epilogue (c - qsum) * ws * xs as the CUDA kernels: bit-identical to pq2_mma_small at split 1 and to
pq2_mma_ring at the same split. split > 1 writes FP32 partials summed in fixed order by pq2_small_mmq._reduce_splits.
Shared buffers use XOR swizzles (16 B chunks ^ row % 8), conflict-free for the fragment reads; Triton's padded shared
layouts do not lower for these loads.
"""
import functools
import math

import torch
import triton

from . import pq2_gluon_small as _small

AVAILABLE = _small.AVAILABLE
if AVAILABLE:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as ttgl
    from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
    from triton.experimental.gluon.language.nvidia.ampere import mma_v2

    from .pq2_gluon_small import _bytes, _plane, layouts as small_layouts

from .pq2_small_mmq import _quantize_fragments, _reduce_splits


if AVAILABLE:
    @functools.cache
    def ring_layouts(warps, rt, sg, st):
        bm, mma, words = small_layouts(warps, rt)
        # the same lane ownership as `words`, on the 2-D [BM, 8] word slice (word j = 2t + h: h = bit 0, t = bits 1-2)
        words2d = ttgl.DistributedLinearLayout(
            reg_bases=[[0, 1], [8, 0]] + [[16 * warps << i, 0] for i in range(int(math.log2(rt)))],
            lane_bases=[[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
            warp_bases=[[1 << i, 0] for i in (4, 5, 6, 7) if (1 << i) < 16 * warps], block_bases=[], shape=[bm, 8])
        # (layouts describe one ring slot; the allocations add the leading stage dimension)
        # XOR swizzles chosen by bank_sim.py (Triton's padded shared layouts do not lower for these NVIDIA loads):
        # weight words [BM, SG*8] int32: 16 B chunks XORed with row % 8 -> conflict-free LDS.64 A-fragment reads (plain: 8-way)
        w_smem = ttgl.SwizzledSharedLayout(4, 1, min(8, 2 * sg), [1, 0])
        # activations [8, SG*128] int8: 16 B chunks XORed with token row -> conflict-free B-fragment reads (plain: 8-way)
        q_smem = ttgl.SwizzledSharedLayout(16, 1, 8, [1, 0])
        plain = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        # global -> shared copy layouts (16 B per thread for codes and activations)
        tw = min(32, sg * 2)
        w_copy = ttgl.BlockedLayout([1, 4], [32 // tw, tw], [warps, 1], [1, 0])
        tq = min(32, sg * 8)
        q_copy = ttgl.BlockedLayout([1, 16], [32 // tq, tq], [warps, 1], [1, 0])
        s_copy = ttgl.BlockedLayout([1, sg], [32, 1], [warps, 1], [1, 0])
        st_copy = ttgl.BlockedLayout([1, 1], [sg, 32 // sg], [1, warps], [1, 0])
        return bm, mma, words2d, w_smem, q_smem, plain, w_copy, q_copy, s_copy, st_copy


    @gluon.jit
    def _issue(Wg, WSg, Qg, Sg, Tg, w_s, ws_s, q_s, xs_s, qs_s, slot, g0, row0, M, N: ttgl.constexpr, K: ttgl.constexpr, BT: ttgl.constexpr,
               BM: ttgl.constexpr, SG: ttgl.constexpr, W_COPY: ttgl.constexpr, Q_COPY: ttgl.constexpr,
               S_COPY: ttgl.constexpr, ST_COPY: ttgl.constexpr):
        G: ttgl.constexpr = K // 128
        # weight code words: rows clamped to N - 1 (as the CUDA kernel); SG*8 int32 words per row from group g0
        r = ttgl.minimum(row0 + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, W_COPY)), N - 1)
        c = ttgl.arange(0, SG * 8, layout=ttgl.SliceLayout(0, W_COPY))
        cp.async_copy_global_to_shared(w_s.index(slot), Wg + r[:, None] * (K // 16) + g0 * 8 + c[None, :])
        rs = ttgl.minimum(row0 + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, S_COPY)), N - 1)
        cs = ttgl.arange(0, SG, layout=ttgl.SliceLayout(0, S_COPY))
        cp.async_copy_global_to_shared(ws_s.index(slot), WSg + rs[:, None] * G + g0 + cs[None, :])
        # activations: 8 token rows (clamped to M - 1), SG*128 bytes from group g0
        t = ttgl.minimum(ttgl.arange(0, BT, layout=ttgl.SliceLayout(1, Q_COPY)), M - 1)
        cq = ttgl.arange(0, SG * 128, layout=ttgl.SliceLayout(0, Q_COPY))
        cp.async_copy_global_to_shared(q_s.index(slot), Qg + t[:, None] * K + g0 * 128 + cq[None, :])
        # activation scales / sums stored transposed, [SG, 8], so a group's 8 values are one row
        gg = ttgl.arange(0, SG, layout=ttgl.SliceLayout(1, ST_COPY))
        ts = ttgl.minimum(ttgl.arange(0, BT, layout=ttgl.SliceLayout(0, ST_COPY)), M - 1)
        cp.async_copy_global_to_shared(xs_s.index(slot), Sg + ts[None, :] * G + g0 + gg[:, None])
        cp.async_copy_global_to_shared(qs_s.index(slot), Tg + ts[None, :] * G + g0 + gg[:, None])


    @gluon.jit
    def _gl_ring(Q, S, T, W, WS, Y, M, N: ttgl.constexpr, K: ttgl.constexpr, BM: ttgl.constexpr, SG: ttgl.constexpr,
                 ST: ttgl.constexpr, SPLIT: ttgl.constexpr, MMA: ttgl.constexpr, WORDS: ttgl.constexpr,
                 W_SMEM: ttgl.constexpr, Q_SMEM: ttgl.constexpr, PLAIN: ttgl.constexpr, W_COPY: ttgl.constexpr,
                 Q_COPY: ttgl.constexpr, S_COPY: ttgl.constexpr, ST_COPY: ttgl.constexpr, TRIVIAL: ttgl.constexpr,
                 BT: ttgl.constexpr = 8):
        G: ttgl.constexpr = K // 128
        PER: ttgl.constexpr = G // SPLIT
        ITERS: ttgl.constexpr = PER // SG
        A_L: ttgl.constexpr = ttgl.DotOperandLayout(parent=MMA, operand_index=0, k_width=4)
        B_L: ttgl.constexpr = ttgl.DotOperandLayout(parent=MMA, operand_index=1, k_width=4)
        row0 = ttgl.program_id(0) * BM
        first = ttgl.program_id(1) * PER
        w_s = ttgl.allocate_shared_memory(ttgl.int32, [ST, BM, SG * 8], W_SMEM)
        ws_s = ttgl.allocate_shared_memory(ttgl.float16, [ST, BM, SG], PLAIN)
        q_s = ttgl.allocate_shared_memory(ttgl.int8, [ST, BT, SG * 128], Q_SMEM)
        xs_s = ttgl.allocate_shared_memory(ttgl.float32, [ST, SG, BT], PLAIN)
        qs_s = ttgl.allocate_shared_memory(ttgl.int32, [ST, SG, BT], PLAIN)
        for s in ttgl.static_range(ST - 1):
            if s < ITERS:
                _issue(W, WS, Q, S, T, w_s, ws_s, q_s, xs_s, qs_s, s, first + s * SG, row0, M, N, K, BT, BM, SG,
                       W_COPY, Q_COPY, S_COPY, ST_COPY)
            cp.commit_group()
        acc = ttgl.zeros([BM, BT], ttgl.float32, layout=MMA)
        for it in range(ITERS):
            cp.wait_group(ST - 2)
            ttgl.barrier()
            nit = it + ST - 1
            if nit < ITERS:
                _issue(W, WS, Q, S, T, w_s, ws_s, q_s, xs_s, qs_s, nit % ST, first + nit * SG, row0, M, N, K, BT, BM, SG,
                       W_COPY, Q_COPY, S_COPY, ST_COPY)
            cp.commit_group()
            slot = it % ST
            wst = w_s.index(slot)
            qst = q_s.index(slot)
            for gl in ttgl.static_range(SG):
                w = wst.slice(gl * 8, 8, dim=1).load(WORDS)                           # [BM, 8] words, register view:
                w = ttgl.permute(ttgl.reshape(w, [BM, 4, 2]), [0, 2, 1])               # [BM, 2 (h), 4 (t)]
                wb = ttgl.convert_layout(_bytes(w), A_L, assert_trivial=TRIVIAL)
                c = ttgl.zeros([BM, BT], ttgl.int32, layout=MMA)
                for s in ttgl.static_range(4):
                    b = qst.slice(gl * 128 + s * 32, 32, dim=1).permute([1, 0]).load(B_L)
                    c = mma_v2(_plane(wb, s), b, c)
                # [BM, 1] / [1, 8] slices loaded in the MMA layout, reduced over the size-1 axis -> the epilogue's layouts
                ws = ttgl.sum(ws_s.index(slot).slice(gl, 1, dim=1).load(MMA).to(ttgl.float32), 1)
                xs = ttgl.sum(xs_s.index(slot).slice(gl, 1, dim=0).load(MMA), 0)
                qs = ttgl.sum(qs_s.index(slot).slice(gl, 1, dim=0).load(MMA), 0)
                acc += (c - qs[None, :]).to(ttgl.float32) * ws[:, None] * xs[None, :]
        cp.wait_group(0)
        rc = row0 + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, MMA))
        nc = ttgl.arange(0, BT, layout=ttgl.SliceLayout(0, MMA))
        ttgl.store(Y + ttgl.program_id(1) * M * N + nc[None, :] * N + rc[:, None], acc,
                   mask=(nc < M)[None, :] & (rc < N)[:, None])


def pq2_gluon_ring(x, weight, warps, rt, sg, st, split=1):
    """x: (M <= 16, K) floating, already rotated; weight: prepared PQ2 planes. Token tile 8 (M <= 8) or 16."""
    m, k = x.shape
    n = weight.shape[0]
    G = k // 128
    if not 1 <= m <= 16 or k % 128 or weight.shape[1] != G * 34 or sg < 2 or G % split or (G // split) % sg:
        raise ValueError("Gluon ring kernel requires 1 to 16 tokens, a PQ2 matrix, SG >= 2 and a K split / stage size dividing K/128")
    x = x.contiguous()
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m, G), dtype=torch.float32, device=x.device)
    t = torch.empty((m, G), dtype=torch.int32, device=x.device)
    _quantize_fragments[(m * G,)](x, q, s, t, k)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    flat = weight.reshape(-1)
    bm, mma, words, w_smem, q_smem, plain, w_copy, q_copy, s_copy, st_copy = ring_layouts(warps, rt, sg, st)
    target = y if split == 1 else torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    _gl_ring[(triton.cdiv(n, bm), split)](q, s, t, flat[: n * k // 4].view(torch.int32),
                                           flat[n * k // 4:].view(torch.float16), target, m, n, k, bm, sg, st, split,
                                           mma, words, w_smem, q_smem, plain, w_copy, q_copy, s_copy, st_copy, rt == 1, 8 if m <= 8 else 16,
                                           num_warps=warps)
    if split > 1:
        _reduce_splits[(triton.cdiv(m * n, 1024),)](target, y, m * n, split, 1024)
    return y
