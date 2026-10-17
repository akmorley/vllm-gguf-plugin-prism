# SPDX-License-Identifier: Apache-2.0
"""Prepared PQ2 single-token DP4A specialization (SM86)."""
import torch
import triton
import triton.language as tl
from .pq2_int_gemv import _decode_codes

@triton.jit
def _load_tile(W, WS, rows, blocks, N: tl.constexpr, K: tl.constexpr):
    valid = (rows[:, None] < N) & (blocks[None, :] < K // 128)
    words = tl.arange(0, 32)
    address = rows[:, None, None] * (K // 4) + blocks[None, :, None] * 32 + words[None, None, :]
    scales = tl.load(WS + rows[:, None] * (K // 128) + blocks[None, :], valid, other=0).to(tl.float32)
    values = tl.load(W + address, valid[:, :, None], other=0).to(tl.int32)
    return values, scales


@triton.jit
def _consume(Q,S,packed,ws,blocks,K:tl.constexpr):
    words=tl.arange(0,32)
    q=tl.load(Q+blocks[:,None]*32+words[None,:],blocks[:,None]<K//128,other=0)
    w=_decode_codes(packed)
    dots=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, 0;',
          constraints='=r,r,r',args=[q[None,:,:],w],dtype=tl.int32,is_pure=True,pack=1)
    partial=tl.sum(dots,2).to(tl.float32)
    xs=tl.load(S+blocks,blocks<K//128,other=0)
    return partial*xs[None,:]*ws


@triton.jit
def _single(Q,S,W,WS,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BC:tl.constexpr):
    rows=tl.program_id(0)*BN+tl.arange(0,BN)
    offset=tl.arange(0,BC)
    acc=tl.full((BN,BC),0,tl.float32)
    for base in tl.range(0,tl.cdiv(K//128,BC),num_stages=1):
        blocks=base*BC+offset
        packed,ws=_load_tile(W,WS,rows,blocks,N,K)
        acc+=_consume(Q,S,packed,ws,blocks,K)
    tl.store(Y+rows,tl.sum(acc,1),rows<N)


def launch_single(q, scales, weight, y, n, k, bn, bc):
    flat=weight.reshape(-1)
    codes=flat[:n*k//4]
    weight_scales=flat[n*k//4:].view(torch.float16)
    return _single[(triton.cdiv(n,bn),)](
        q.view(torch.int32),scales,codes,weight_scales,y,n,k,bn,bc,num_warps=4)


@triton.jit
def _dp4a(a, b, c):
    return tl.inline_asm_elementwise("dp4a.s32.s32 $0, $1, $2, $3;", constraints="=r,r,r,r",
                                     args=[a, b, c], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _single_planes(Q, S, T, W, WS, Y, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr,
                   STAGES: tl.constexpr):
    """DP4A on the raw 2-bit planes (the algorithm of csrc/pq2/pq2_gemv_m1.cu): no PRMT decode.

    Activations are in plane order (_quantize_planes_sum), so as int32 words Q[g*32 + s*8 + j] pairs with plane s of
    weight word j, (w >> 2s) & 0x03030303; the ternary -1 offset is applied once per group as -T[g]. Integer group
    sums equal `_single`'s; FP32 accumulation order follows the (BN, BC) tile.
    """
    G: tl.constexpr = K // 128
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    offs = tl.arange(0, BC)
    j = tl.arange(0, 8)
    row_ok = rows < N
    acc = tl.full((BN, BC), 0, tl.float32)
    for base in tl.range(0, tl.cdiv(G, BC), num_stages=STAGES):
        g = base * BC + offs
        g_ok = g < G
        m2 = row_ok[:, None] & g_ok[None, :]
        w = tl.load(W + rows[:, None, None] * (K // 16) + g[None, :, None] * 8 + j[None, None, :],
                    m2[:, :, None], other=0)
        ws = tl.load(WS + rows[:, None] * G + g[None, :], m2, other=0).to(tl.float32)
        qb = Q + g[:, None] * 32 + j[None, :]
        q0 = tl.load(qb, g_ok[:, None], other=0)
        q1 = tl.load(qb + 8, g_ok[:, None], other=0)
        q2 = tl.load(qb + 16, g_ok[:, None], other=0)
        q3 = tl.load(qb + 24, g_ok[:, None], other=0)
        d = _dp4a(w & 0x03030303, q0[None, :, :], tl.zeros((BN, BC, 8), tl.int32))
        d = _dp4a((w >> 2) & 0x03030303, q1[None, :, :], d)
        d = _dp4a((w >> 4) & 0x03030303, q2[None, :, :], d)
        d = _dp4a((w >> 6) & 0x03030303, q3[None, :, :], d)
        dot = tl.sum(d, 2) - tl.load(T + g, g_ok, other=0)[None, :]
        xs = tl.load(S + g, g_ok, other=0)
        acc += dot.to(tl.float32) * xs[None, :] * ws
    tl.store(Y + rows, tl.sum(acc, 1), row_ok)


def launch_planes(q, scales, sums, weight, y, n, k, bn, bc, warps, stages):
    flat = weight.reshape(-1)
    codes = flat[: n * k // 4].view(torch.int32)
    weight_scales = flat[n * k // 4:].view(torch.float16)
    return _single_planes[(triton.cdiv(n, bn),)](
        q.view(torch.int32), scales, sums, codes, weight_scales, y, n, k, bn, bc, stages, num_warps=warps)
