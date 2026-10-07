# SPDX-License-Identifier: Apache-2.0
"""Small-batch (M = 2..16) PQ2 x Q8 projection on INT8 tensor cores (SM86), swapped operands.

Weights are the MMA A operand (BN rows x 32 bytes per 2-bit plane); activations are the
B operand (32 x 16 tokens, padded). Activations use the production group-128 Q8 arithmetic
but are stored in plane order (k = byte * 4 + s -> s * 32 + byte), so each plane of a packed
byte multiplies a contiguous slice. Plane s is `byte & (3 << 2s)` = code * 4^s; the exact
INT32 product is shifted back, and the ternary -1 offset is applied once per group as -sum(q).
INT32 group sums equal the DP4A kernel's; only FP32 accumulation order differs.
Serving use is opt-in: GGUF_PQ2_SMALL_MMQ=1 (M = 2..16).
"""
import os

import torch
import triton
import triton.language as tl

# GGUF_PQ2_SMALL_MMQ=1 enables M = 2..16 (speculative verification, small batches).
_SMALL_MMQ = os.environ.get("GGUF_PQ2_SMALL_MMQ", "0") == "1"
# Opt-in: also route the (5120, 6144) output projections (M = 2..8) through the INT8 CUDA kernel.
# This is the Q8 treatment production already applies at M = 1 (GGUF_PQ2_INT_OUTPUT), but it
# changes the floating M >= 2 outputs by ~0.7% relative RMS.
_SMALL_MMQ_OUTPUT = os.environ.get("GGUF_PQ2_SMALL_MMQ_OUTPUT", "0") == "1"
OUTPUT_CUDA = (2, 8)  # (warps, groups in flight), RTX 3090 Ti microbenchmark, M = 2..8
_VERSION = "masked-planes-splitk-v2-m2-16+cuda-mma-v1"
_CALLS = 0  # Python-side launches (warmup and graph capture); replay does not re-enter.
# M -> (N, K) -> (rows per program, groups per iteration, warps, pipeline stages, K splits).
# Selected on real captured rows, RTX 3090: pq2-small-mmq-screen-20261007-masked (M = 8)
# and pq2-small-mmq-mscreen-20261007 (other M). M = 9..15 use the M = 16 table.
GEOMETRY_BY_M = {
    2: {
        (248320, 5120): (128, 2, 4, 3, 1),
    },
    3: {
        (5120, 17408): (64, 2, 4, 2, 4),
        (14336, 5120): (64, 2, 4, 3, 1),
        (16384, 5120): (64, 2, 4, 3, 1),
        (34816, 5120): (128, 2, 4, 2, 1),
        (248320, 5120): (128, 2, 4, 4, 1),
    },
    4: {
        (5120, 17408): (64, 2, 4, 2, 4),
        (14336, 5120): (64, 2, 4, 3, 1),
        (16384, 5120): (64, 2, 4, 3, 1),
        (34816, 5120): (128, 2, 4, 2, 1),
        (248320, 5120): (128, 2, 4, 4, 1),
    },
    5: {
        (5120, 17408): (64, 2, 4, 2, 4),
        (14336, 5120): (64, 2, 4, 3, 1),
        (16384, 5120): (64, 2, 4, 3, 1),
        (34816, 5120): (128, 2, 4, 2, 1),
        (248320, 5120): (128, 2, 4, 4, 1),
    },
    6: {
        (5120, 17408): (64, 2, 4, 2, 4),
        (14336, 5120): (64, 2, 4, 3, 1),
        (16384, 5120): (64, 2, 4, 3, 1),
        (34816, 5120): (128, 2, 4, 2, 1),
        (248320, 5120): (128, 2, 4, 4, 1),
    },
    7: {
        (5120, 17408): (64, 2, 4, 2, 4),
        (14336, 5120): (64, 2, 4, 3, 1),
        (16384, 5120): (64, 2, 4, 3, 1),
        (34816, 5120): (128, 2, 4, 2, 1),
        (248320, 5120): (128, 2, 4, 4, 1),
    },
    8: {
        (16384, 5120): (64, 2, 4, 3, 1),
        (34816, 5120): (128, 2, 4, 2, 1),
        (5120, 17408): (64, 2, 4, 2, 4),
        (14336, 5120): (64, 2, 4, 3, 1),
        (248320, 5120): (128, 2, 4, 4, 1),
    },
    16: {
        (5120, 17408): (64, 2, 4, 3, 2),
        (14336, 5120): (64, 2, 4, 3, 1),
        (16384, 5120): (128, 2, 4, 3, 1),
        (34816, 5120): (128, 2, 4, 2, 1),
        (248320, 5120): (128, 2, 4, 4, 1),
    },
}


def geometry_for(m, n, k):
    table = GEOMETRY_BY_M.get(16 if 8 < m <= 16 else m)
    return None if table is None else table.get((n, k))


# Direct-fragment CUDA MMA kernel (csrc/pq2/pq2_mma_small.cu): M -> (N, K) -> (warps, groups in
# flight). Preferred over the Triton kernel where present (M <= 8). Selected on real captured
# rows, RTX 3090: pq2-mma-small-screen-20261007.
CUDA_BY_M = {
    2: {
        (5120, 17408): (2, 8),
        (14336, 5120): (4, 4),
        (16384, 5120): (2, 4),
        (34816, 5120): (2, 8),
        (248320, 5120): (4, 8),
    },
    3: {
        (5120, 17408): (2, 8),
        (14336, 5120): (4, 4),
        (16384, 5120): (2, 4),
        (34816, 5120): (2, 8),
        (248320, 5120): (2, 8),
    },
    4: {
        (5120, 17408): (2, 8),
        (14336, 5120): (2, 4),
        (16384, 5120): (2, 4),
        (34816, 5120): (4, 8),
        (248320, 5120): (2, 8),
    },
    5: {
        (5120, 17408): (2, 8),
        (14336, 5120): (2, 4),
        (16384, 5120): (2, 4),
        (34816, 5120): (2, 8),
        (248320, 5120): (2, 8),
    },
    6: {
        (5120, 17408): (2, 8),
        (14336, 5120): (2, 4),
        (16384, 5120): (2, 4),
        (34816, 5120): (2, 8),
        (248320, 5120): (2, 8),
    },
    7: {
        (14336, 5120): (2, 4),
        (16384, 5120): (2, 4),
        (34816, 5120): (2, 8),
        (248320, 5120): (2, 8),
    },
    8: {
        (14336, 5120): (2, 4),
        (16384, 5120): (2, 4),
        (34816, 5120): (2, 8),
        (248320, 5120): (2, 8),
    },
}


def cuda_for(m, n, k):
    if not _cuda_available():
        return None
    if _SMALL_MMQ_OUTPUT and (n, k) == (5120, 6144) and 2 <= m <= 8:
        return OUTPUT_CUDA
    return CUDA_BY_M.get(m, {}).get((n, k))


def _cuda_available():
    try:
        from .. import _C_gguf  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops, "_C_gguf") and hasattr(torch.ops._C_gguf, "pq2_mma_small")


@triton.jit
def _quantize_planes_sum(X, Q, S, T, K: tl.constexpr):
    # As _quantize_planes, plus the integer sum of each group's codes (for the -1 offset).
    block = tl.program_id(0)
    i = tl.arange(0, 128)
    a = tl.load(X + block * 128 + i).to(tl.float32)
    scale = tl.max(tl.abs(a), 0) / 127.0
    normalized = tl.div_rn(a, tl.where(scale > 0, scale, 1.0))
    q = tl.minimum(127.0, tl.maximum(-127.0, tl.extra.cuda.libdevice.nearbyint(normalized)))
    tl.store(Q + block * 128 + (i % 4) * 32 + i // 4, q.to(tl.int8))
    tl.store(S + block, scale)
    tl.store(T + block, tl.sum(q.to(tl.int32), 0))


@triton.jit
def _quantize_fragments(X, Q, S, T, K: tl.constexpr):
    # Production group-128 Q8 arithmetic, stored at s*32 + slot(byte) to match the CUDA kernel's
    # m16n8k32 fragments: slot = 4*(byte//8) + byte%8 for byte%8 < 4, else 16 + 4*(byte//8) + byte%8 - 4.
    block = tl.program_id(0)
    i = tl.arange(0, 128)
    a = tl.load(X + block * 128 + i).to(tl.float32)
    scale = tl.max(tl.abs(a), 0) / 127.0
    normalized = tl.div_rn(a, tl.where(scale > 0, scale, 1.0))
    q = tl.minimum(127.0, tl.maximum(-127.0, tl.extra.cuda.libdevice.nearbyint(normalized)))
    byte = i // 4
    off = byte % 8
    slot = tl.where(off < 4, 4 * (byte // 8) + off, 16 + 4 * (byte // 8) + off - 4)
    tl.store(Q + block * 128 + (i % 4) * 32 + slot, q.to(tl.int8))
    tl.store(S + block, scale)
    tl.store(T + block, tl.sum(q.to(tl.int32), 0))


@triton.jit(do_not_specialize=["M"])
def _small_mmq_masked(Q, S, T, W, WS, Y, M, N: tl.constexpr, K: tl.constexpr,
                      BN: tl.constexpr, G: tl.constexpr, STAGES: tl.constexpr,
                      SPLIT: tl.constexpr = 1):
    """Codes stay in place: plane s is `byte & (3 << 2s)` = code * 4^s (plane 3 pre-shifted
    by 2 to stay below 128). The exact INT32 product is shifted back, and the ternary -1
    offset is applied once per group as -sum(q). Integer result equals the DP4A kernel's."""
    BT: tl.constexpr = 16
    GROUPS: tl.constexpr = K // 128
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    tokens = tl.arange(0, BT)
    byte = tl.arange(0, 32)
    row_ok = rows < N
    tok_ok = tokens < M
    acc = tl.zeros((BN, BT), tl.float32)
    PER: tl.constexpr = GROUPS // SPLIT
    first = tl.program_id(1) * PER
    for base in tl.range(first, first + PER, G, num_stages=STAGES):
        for j in tl.static_range(G):
            g = base + j
            packed = tl.load(W + rows[:, None] * (K // 4) + g * 32 + byte[None, :],
                             row_ok[:, None], other=0)
            q0 = tl.load(Q + tokens[None, :] * K + g * 128 + byte[:, None], tok_ok[None, :], other=0)
            q1 = tl.load(Q + tokens[None, :] * K + g * 128 + 32 + byte[:, None], tok_ok[None, :], other=0)
            q2 = tl.load(Q + tokens[None, :] * K + g * 128 + 64 + byte[:, None], tok_ok[None, :], other=0)
            q3 = tl.load(Q + tokens[None, :] * K + g * 128 + 96 + byte[:, None], tok_ok[None, :], other=0)
            part = tl.dot((packed & 0x03).to(tl.int8), q0)
            part += tl.dot((packed & 0x0C).to(tl.int8), q1) >> 2
            part += tl.dot((packed & 0x30).to(tl.int8), q2) >> 4
            part += tl.dot(((packed >> 2) & 0x30).to(tl.int8), q3) >> 4
            part -= tl.load(T + tokens * GROUPS + g, tok_ok, other=0)[None, :]
            ws = tl.load(WS + rows * GROUPS + g, row_ok, other=0).to(tl.float32)
            xs = tl.load(S + tokens * GROUPS + g, tok_ok, other=0)
            acc += part.to(tl.float32) * ws[:, None] * xs[None, :]
    # SPLIT == 1 stores the output directly; otherwise Y is an FP32 (SPLIT, M, N) workspace.
    out = Y + tl.program_id(1) * M * N
    tl.store(out + tokens[None, :] * N + rows[:, None], acc, row_ok[:, None] & tok_ok[None, :])


@triton.jit
def _reduce_splits(P, Y, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    # Fixed summation order over splits: deterministic output.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < MN
    acc = tl.load(P + i, ok, other=0)
    for s in tl.static_range(1, SPLIT):
        acc += tl.load(P + s * MN + i, ok, other=0)
    tl.store(Y + i, acc, ok)




def eligible(x, weight, prepared):
    """Prepared SM86 BF16/FP16 calls with a tuned (M, N, K) entry and an enabled switch."""
    if not prepared or x.ndim != 2 or not x.is_cuda:
        return False
    m = x.shape[0]
    enabled = _SMALL_MMQ and 2 <= m <= 16
    return (
        enabled
        and x.dtype in (torch.bfloat16, torch.float16)
        and (geometry_for(m, weight.shape[0], x.shape[1]) is not None
             or cuda_for(m, weight.shape[0], x.shape[1]) is not None)
        and torch.cuda.get_device_capability(x.device) == (8, 6)
    )


def pq2_small_mmq(x, weight, geometry=None):
    """x: (M <= 16, K) floating, already rotated; weight: prepared PQ2 planes."""
    global _CALLS
    _CALLS += 1
    m, k = x.shape
    n = weight.shape[0]
    cuda = None if geometry is not None else cuda_for(m, n, k)
    if cuda is not None:
        return _pq2_mma_small(x, weight, *cuda)
    bn, g, warps, stages, split = geometry or geometry_for(m, n, k)
    if not 1 <= m <= 16 or k % 128 or weight.shape[1] != k // 128 * 34:
        raise ValueError("PQ2 small MMQ requires 1 to 16 tokens and a PQ2 matrix")
    if (k // 128) % split or (k // 128 // split) % g:
        raise ValueError("Split and group count must divide K/128")
    x = x.contiguous()
    flat = weight.reshape(-1)
    codes, scales = flat[: n * k // 4], flat[n * k // 4:].view(torch.float16)
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
    t = torch.empty((m, k // 128), dtype=torch.int32, device=x.device)
    _quantize_planes_sum[(m * (k // 128),)](x, q, s, t, k)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    target = y if split == 1 else torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    _small_mmq_masked[(triton.cdiv(n, bn), split)](
        q, s, t, codes, scales, target, m, n, k, bn, g, stages, split, num_warps=warps
    )
    if split > 1:
        _reduce_splits[(triton.cdiv(m * n, 1024),)](target, y, m * n, split, 1024)
    return y


def _pq2_mma_small(x, weight, warps, unroll):
    m, k = x.shape
    n = weight.shape[0]
    if not 1 <= m <= 16 or k % 128 or weight.shape[1] != k // 128 * 34:
        raise ValueError("PQ2 MMA kernel requires 1 to 16 tokens and a PQ2 matrix")
    x = x.contiguous()
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
    t = torch.empty((m, k // 128), dtype=torch.int32, device=x.device)
    _quantize_fragments[(m * (k // 128),)](x, q, s, t, k)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    torch.ops._C_gguf.pq2_mma_small(q, s, t, weight, y, warps, unroll)
    return y
