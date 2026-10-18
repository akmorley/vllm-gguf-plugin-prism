# SPDX-License-Identifier: Apache-2.0
"""Gluon port of csrc/pq2/pq2_mma_prefill.cu (M >= 64): INT8 tensor-core MMA on the raw 2-bit planes.

CTA tile BM = 16*WR*RT weight rows x BT = 8*WT*NT tokens on a WR x WT warp grid, ST-stage cp.async ring of SG groups.
Packed code bytes are staged in shared memory and loaded straight into A-fragment registers (planes by inline PTX), the
structure of pq2_gluon_ring; Triton's tl.dot (pq2_mmq._mmq_planes) instead expands four int8 planes through shared
memory, which costs instructions and energy in power-capped prefill. Epilogue ((c - qsum) * xs) * ws per group: the CUDA
prefill kernel's and _mmq's order, so the output is bit-exact with both.
"""
import functools
import math

import torch
import triton
import triton.language as tl

from . import pq2_gluon_small as _small

AVAILABLE = _small.AVAILABLE
if AVAILABLE:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as ttgl
    from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
    from triton.experimental.gluon.language.nvidia.ampere import mma_v2

    from .pq2_gluon_small import _bytes, _plane


@triton.jit
def _quantize_fragments_rows(X, Q, S, T, G, R: tl.constexpr):
    """pq2_small_mmq._quantize_fragments (production group-128 Q8 arithmetic, m16n8k32 fragment order), R groups per
    program instead of one."""
    r1 = tl.program_id(0) * R + tl.arange(0, R)
    r = r1[:, None]
    i = tl.arange(0, 128)[None, :]
    mask = r < G
    a = tl.load(X + r * 128 + i, mask, other=0).to(tl.float32)
    scale = tl.max(tl.abs(a), 1) / 127.0
    normalized = tl.div_rn(a, tl.where(scale > 0, scale, 1.0)[:, None])
    q = tl.minimum(127.0, tl.maximum(-127.0, tl.extra.cuda.libdevice.nearbyint(normalized)))
    qi = q.to(tl.int8)
    byte = i // 4
    off = byte % 8
    slot = tl.where(off < 4, 4 * (byte // 8) + off, 16 + 4 * (byte // 8) + off - 4)
    tl.store(Q + r * 128 + (i % 4) * 32 + slot, qi, mask)
    tl.store(S + r1, scale, r1 < G)
    tl.store(T + r1, tl.sum(qi.to(tl.int32), 1), r1 < G)


if AVAILABLE:
    @functools.cache
    def prefill_layouts(wr, wt, rt, nt, sg):
        nw = wr * wt
        bm, bt = 16 * wr * rt, 8 * wt * nt
        mma = ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[wr, wt], instr_shape=[16, 8])
        # weight words [BM, 8] of one group with the A operand's lane ownership (word j = 2t + h); warp bit 0 is the token
        # direction (broadcast), the rest split the rows (printed A layout: warp_bases [[0,0],[16,0],[32,0]] for 4 x 2)
        wb = [[0, 0] for _ in range(int(math.log2(wt)))] + [[16 << i, 0] for i in range(int(math.log2(wr)))]
        words2d = ttgl.DistributedLinearLayout(
            reg_bases=[[0, 1], [8, 0]] + [[16 * wr << i, 0] for i in range(int(math.log2(rt)))],
            lane_bases=[[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp_bases=wb, block_bases=[], shape=[bm, 8])
        w_smem = ttgl.SwizzledSharedLayout(4, 1, min(8, 2 * sg), [1, 0])
        q_smem = ttgl.SwizzledSharedLayout(16, 1, 8, [1, 0])
        plain = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        tw = min(32, sg * 2)
        w_copy = ttgl.BlockedLayout([1, 4], [32 // tw, tw], [nw, 1], [1, 0])
        tq = min(32, sg * 8)
        q_copy = ttgl.BlockedLayout([1, 16], [32 // tq, tq], [nw, 1], [1, 0])
        s_copy = ttgl.BlockedLayout([1, sg], [32, 1], [nw, 1], [1, 0])
        st_copy = ttgl.BlockedLayout([1, 1], [sg, 32 // sg], [1, nw], [1, 0])
        return bm, bt, mma, words2d, w_smem, q_smem, plain, w_copy, q_copy, s_copy, st_copy


    @gluon.jit
    def _issue(Wg, WSg, Qg, Sg, Tg, w_s, ws_s, q_s, xs_s, qs_s, slot, g0, row0, tok0, M, N: ttgl.constexpr,
               K: ttgl.constexpr, BM: ttgl.constexpr, BT: ttgl.constexpr, SG: ttgl.constexpr, W_COPY: ttgl.constexpr,
               Q_COPY: ttgl.constexpr, S_COPY: ttgl.constexpr, ST_COPY: ttgl.constexpr):
        G: ttgl.constexpr = K // 128
        r = ttgl.minimum(row0 + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, W_COPY)), N - 1)
        c = ttgl.arange(0, SG * 8, layout=ttgl.SliceLayout(0, W_COPY))
        cp.async_copy_global_to_shared(w_s.index(slot), Wg + r[:, None] * (K // 16) + g0 * 8 + c[None, :])
        rs = ttgl.minimum(row0 + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, S_COPY)), N - 1)
        cs = ttgl.arange(0, SG, layout=ttgl.SliceLayout(0, S_COPY))
        cp.async_copy_global_to_shared(ws_s.index(slot), WSg + rs[:, None] * G + g0 + cs[None, :])
        t = ttgl.minimum(tok0 + ttgl.arange(0, BT, layout=ttgl.SliceLayout(1, Q_COPY)), M - 1)
        cq = ttgl.arange(0, SG * 128, layout=ttgl.SliceLayout(0, Q_COPY))
        cp.async_copy_global_to_shared(q_s.index(slot), Qg + t[:, None] * K + g0 * 128 + cq[None, :])
        gg = ttgl.arange(0, SG, layout=ttgl.SliceLayout(1, ST_COPY))
        ts = ttgl.minimum(tok0 + ttgl.arange(0, BT, layout=ttgl.SliceLayout(0, ST_COPY)), M - 1)
        cp.async_copy_global_to_shared(xs_s.index(slot), Sg + ts[None, :] * G + g0 + gg[:, None])
        cp.async_copy_global_to_shared(qs_s.index(slot), Tg + ts[None, :] * G + g0 + gg[:, None])


    @gluon.jit
    def _gl_prefill(Q, S, T, W, WS, Y, M, N: ttgl.constexpr, K: ttgl.constexpr, BM: ttgl.constexpr, BT: ttgl.constexpr,
                    SG: ttgl.constexpr, ST: ttgl.constexpr, MMA: ttgl.constexpr, WORDS: ttgl.constexpr,
                    W_SMEM: ttgl.constexpr, Q_SMEM: ttgl.constexpr, PLAIN: ttgl.constexpr, W_COPY: ttgl.constexpr,
                    Q_COPY: ttgl.constexpr, S_COPY: ttgl.constexpr, ST_COPY: ttgl.constexpr):
        G: ttgl.constexpr = K // 128
        ITERS: ttgl.constexpr = G // SG
        A_L: ttgl.constexpr = ttgl.DotOperandLayout(parent=MMA, operand_index=0, k_width=4)
        B_L: ttgl.constexpr = ttgl.DotOperandLayout(parent=MMA, operand_index=1, k_width=4)
        row0 = ttgl.program_id(0) * BM
        tok0 = ttgl.program_id(1) * BT
        w_s = ttgl.allocate_shared_memory(ttgl.int32, [ST, BM, SG * 8], W_SMEM)
        ws_s = ttgl.allocate_shared_memory(ttgl.float16, [ST, BM, SG], PLAIN)
        q_s = ttgl.allocate_shared_memory(ttgl.int8, [ST, BT, SG * 128], Q_SMEM)
        xs_s = ttgl.allocate_shared_memory(ttgl.float32, [ST, SG, BT], PLAIN)
        qs_s = ttgl.allocate_shared_memory(ttgl.int32, [ST, SG, BT], PLAIN)
        for s in ttgl.static_range(ST - 1):
            if s < ITERS:
                _issue(W, WS, Q, S, T, w_s, ws_s, q_s, xs_s, qs_s, s, s * SG, row0, tok0, M, N, K, BM, BT, SG,
                       W_COPY, Q_COPY, S_COPY, ST_COPY)
            cp.commit_group()
        acc = ttgl.zeros([BM, BT], ttgl.float32, layout=MMA)
        for it in range(ITERS):
            cp.wait_group(ST - 2)
            ttgl.barrier()
            nit = it + ST - 1
            if nit < ITERS:
                _issue(W, WS, Q, S, T, w_s, ws_s, q_s, xs_s, qs_s, nit % ST, nit * SG, row0, tok0, M, N, K, BM, BT, SG,
                       W_COPY, Q_COPY, S_COPY, ST_COPY)
            cp.commit_group()
            slot = it % ST
            wst = w_s.index(slot)
            qst = q_s.index(slot)
            for gl in ttgl.static_range(SG):
                w = wst.slice(gl * 8, 8, dim=1).load(WORDS)
                w = ttgl.permute(ttgl.reshape(w, [BM, 4, 2]), [0, 2, 1])
                wb = ttgl.convert_layout(_bytes(w), A_L)
                c = ttgl.zeros([BM, BT], ttgl.int32, layout=MMA)
                for s in ttgl.static_range(4):
                    b = qst.slice(gl * 128 + s * 32, 32, dim=1).permute([1, 0]).load(B_L)
                    c = mma_v2(_plane(wb, s), b, c)
                ws = ttgl.sum(ws_s.index(slot).slice(gl, 1, dim=1).load(MMA).to(ttgl.float32), 1)
                xs = ttgl.sum(xs_s.index(slot).slice(gl, 1, dim=0).load(MMA), 0)
                qs = ttgl.sum(qs_s.index(slot).slice(gl, 1, dim=0).load(MMA), 0)
                acc += (c - qs[None, :]).to(ttgl.float32) * xs[None, :] * ws[:, None]
        cp.wait_group(0)
        rc = row0 + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, MMA))
        nc = tok0 + ttgl.arange(0, BT, layout=ttgl.SliceLayout(0, MMA))
        ttgl.store(Y + nc[None, :] * N + rc[:, None], acc, mask=(nc < M)[None, :] & (rc < N)[:, None])


# Launch plans by (weight, tile, dtype, M % 16 == 0, M*G % 16 == 0): prefill runs outside CUDA graphs, and Triton's
# per-call argument binding and specialization (which hashes the layout constexprs), the weight-view slicing and the
# checks cost ~45 us of CPU per projection, enough to make short prompts launch-bound (a 128-token prompt was 12%
# slower than the CUDA kernel). A plan holds both compiled kernels, the weight views and the constant arguments, so a
# call only allocates its buffers and fires two direct launches.
_PLANS = {}


def _make_plan(x, weight, m, n, k, cfg, warm_cfgs):
    G = k // 128
    wr, wt, rt, nt, sg, st = cfg
    if k % 128 or weight.shape[1] != G * 34 or sg < 2 or G % sg:
        raise ValueError("Gluon prefill kernel requires a PQ2 matrix and SG >= 2 dividing K/128")
    flat = weight.reshape(-1)
    codes, scales = flat[: n * k // 4].view(torch.int32), flat[n * k // 4:].view(torch.float16)
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m, G), dtype=torch.float32, device=x.device)
    t = torch.empty((m, G), dtype=torch.int32, device=x.device)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    # compile (no launch) every tile this shape can use, for both M-divisibility specializations, so that no later
    # prefill chunk compiles on a live request
    kerns = {}
    for c in sorted(set(warm_cfgs) | {tuple(cfg)}):
        lay = prefill_layouts(*c[:5])
        consts = (lay[0], lay[1], c[4], c[5], *lay[2:])
        for mm in (m, m + 1 if m % 16 == 0 else m + 16 - m % 16):
            kerns[(c, mm % 16 == 0)] = _gl_prefill.warmup(q, s, t, codes, scales, y, mm, n, k, *consts,
                                                          num_warps=c[0] * c[1], grid=(1,))
    lay = prefill_layouts(wr, wt, rt, nt, sg)
    qkern = _quantize_fragments_rows.warmup(x, q, s, t, m * G, 8, num_warps=4, grid=(1,))
    return kerns[(tuple(cfg), m % 16 == 0)], qkern, codes, scales, lay[0], lay[1], (lay[0], lay[1], sg, st, *lay[2:])


def pq2_gluon_prefill(x, weight, cfg, warm_cfgs=()):
    """x: (M, K) contiguous, floating, already rotated; weight: prepared PQ2 planes; cfg = (WR, WT, RT, NT, SG, ST).

    warm_cfgs: every tile this shape can use, compiled on its first call."""
    m, k = x.shape
    n = weight.shape[0]
    G = k // 128
    key = (weight.data_ptr(), n, k, cfg, x.dtype, m % 16 == 0, (m * G) % 16 == 0)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS[key] = _make_plan(x, weight, m, n, k, tuple(cfg), warm_cfgs)
    kern, qkern, codes, scales, bm, bt, consts = plan
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m, G), dtype=torch.float32, device=x.device)
    t = torch.empty((m, G), dtype=torch.int32, device=x.device)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    qkern[((m * G + 7) // 8, 1, 1)](x, q, s, t, m * G, 8)
    kern[((n + bm - 1) // bm, (m + bt - 1) // bt, 1)](q, s, t, codes, scales, y, m, n, k, *consts)
    return y
