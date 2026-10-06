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
    w=_decode_codes(packed,True)
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
