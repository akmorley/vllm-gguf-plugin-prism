# SPDX-License-Identifier: Apache-2.0
"""Small-batch (M = 2..64) PQ2 x Q8 projection on INT8 tensor cores (SM86), swapped operands.

Weights are the MMA A operand (BN rows x 32 bytes per 2-bit plane); activations are the
B operand (32 x 16/32/64 tokens, padded). Activations use the production group-128 Q8 arithmetic
but are stored in plane order (k = byte * 4 + s -> s * 32 + byte), so each plane of a packed
byte multiplies a contiguous slice. Plane s is `byte & (3 << 2s)` = code * 4^s; the exact
INT32 product is shifted back, and the ternary -1 offset is applied once per group as -sum(q).
INT32 group sums equal the DP4A kernel's; only FP32 accumulation order differs.
On by default within the integer-decode path (prepared weights, SM86); set
GGUF_PQ2_SMALL_MMQ=0 to disable.
"""
import os

import torch
import triton
import triton.language as tl

from . import pq2_config as _config
from . import pq2_tables as _tables

# M = 2..64 (speculative verification, small batches). Default on; GGUF_PQ2_SMALL_MMQ=0 disables.
_SMALL_MMQ = os.environ.get("GGUF_PQ2_SMALL_MMQ", "1") == "1"
# Also route the (5120, 6144) output projections (M = 2..8) through the INT8 CUDA kernel.
# This is the Q8 treatment production already applies at M = 1 (GGUF_PQ2_INT_OUTPUT), but it
# changes the floating M >= 2 outputs by ~0.7% relative RMS. Default on (paired GSM8K/GPQA showed
# no difference: pq2-dflash-quality-20261007); GGUF_PQ2_SMALL_MMQ_OUTPUT=0 disables.
_SMALL_MMQ_OUTPUT = os.environ.get("GGUF_PQ2_SMALL_MMQ_OUTPUT", "1") == "1"
OUTPUT_CUDA = (2, 8)  # (warps, groups in flight), RTX 3090 Ti microbenchmark, M = 2..8
# Register-tiled CUDA variants (row tiles per warp > 1); GGUF_PQ2_SMALL_MMQ_RT=0 restores one tile per warp.
_CUDA_RT = _config.flag("GGUF_PQ2_SMALL_MMQ_RT", "exact_class")
_VERSION = "masked-planes-splitk-v2-m2-16+cuda-mma-v3-rt-ring"
_MAX_M = 64
_CALLS = 0  # Python-side launches (warmup and graph capture); replay does not re-enter.
# M -> (N, K) -> (rows per program, groups per iteration, warps, pipeline stages, K splits).
# Selected on real captured rows, RTX 3090: pq2-small-mmq-screen-20261007-masked (M = 8)
# and pq2-small-mmq-mscreen-20261007 (other M). Buckets: M = 9..16, 17..32, 33..64.
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
    32: {  # token tile 32: M = 17..32 (synthetic sweep, RTX 3090 Ti)
        (5120, 17408): (128, 2, 4, 2, 4),
        (14336, 5120): (128, 1, 4, 3, 1),
        (16384, 5120): (128, 1, 4, 2, 1),
        (34816, 5120): (128, 1, 4, 2, 1),
        (248320, 5120): (128, 1, 4, 3, 1),
    },
    64: {  # token tile 64: M = 33..64 (synthetic sweep, RTX 3090 Ti)
        (5120, 17408): (128, 2, 4, 2, 2),
        (14336, 5120): (128, 1, 4, 2, 1),
        (16384, 5120): (128, 1, 4, 2, 1),
        (34816, 5120): (128, 2, 4, 2, 1),
        (248320, 5120): (128, 2, 4, 2, 1),
    },
}


def _bucket(m):
    # Exact tables for M <= 8; above that, one table per token tile (16, 32 or 64 rows).
    return m if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 64


def _token_tile(m):
    return 16 if m <= 16 else 32 if m <= 32 else 64


def geometry_for(m, n, k):
    table = GEOMETRY_BY_M.get(_bucket(m))
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


# (warps, groups in flight, row tiles per warp): register-tiled variants of the CUDA kernel. Each warp
# handles RT 16-row tiles per group, loading the group's activation fragments once instead of RT
# times; per-element arithmetic (and so the output) is identical to RT = 1. Selected on RTX 3090 Ti
# (pq2-batched-decode-longctx-20261015); only entries >= 4% faster than the previous path are listed.
# Not listed: 5120 x 17408 and 5120 x 6144 (320 row tiles: too little parallelism to gain).
CUDA_RT_BY_M = _tables.SM86.rt_by_m  # per-architecture tables: pq2_tables.py


# cp.async-ring CUDA kernel (csrc/pq2/pq2_mma_ring.cu): weights and activations are staged in shared memory so every
# DRAM request is a >= 64 B row segment. The direct-fragment kernel above issues one 32 B piece per row per group,
# which limits it on the narrow projections (N = 5120) and at M >= 6 for the wide ones.
# M -> (N, K) -> (cfg, split). cfg selects the CTA tile in pq2_mma_ring.cu; split > 1 is split-K with a fused
# fixed-order reduction (deterministic). Entries in RING_EXACT sum in the same order as the path they replace, so the
# output is bit-identical; the rest differ at the FP32 rounding level (relative RMS ~1e-4) and are deterministic.
# Selected on RTX 3090 Ti against the paths above and confirmed on the captured real batch-8 decode data (257 projections:
# 13.70 -> 12.43 ms per step); in_proj and gate/up showed no gain on real data and stay on the direct-fragment kernel.
# (pq2-decode-kernel-design-20261015)
RING_BY_M = _tables.SM86.ring_by_m
RING_EXACT = _tables.SM86.ring_exact
_RING = _config.flag("GGUF_PQ2_SMALL_MMQ_RING", "exact_class")
# GGUF_PQ2_SMALL_MMQ_RING_EXACT=1 keeps only the entries in RING_EXACT (bit-identical outputs); the split-K `down`
# entries (the largest gain, FP32 rounding-level reordering) are then left on the previous path.
_RING_EXACT_ONLY = _config.flag("GGUF_PQ2_SMALL_MMQ_RING_EXACT", "exact_only_ring")
_RING_WS = {}
# Gluon ports of the direct-fragment kernel (pq2_gluon_small.py) and of the cp.async ring kernel (pq2_gluon_ring.py),
# preferred over every CUDA small-batch kernel where GLUON_BY_M / GLUON_RING_BY_M list them (tuned under the serving
# power proxy). Cells in GLUON_EXACT are bit-identical to production; the rest differ at FP32 order. In the `tuned` set
# (all cells) and the `exact` set (GLUON_EXACT cells only); GGUF_PQ2_SMALL_GLUON=0 disables.
_GLUON = _config.flag("GGUF_PQ2_SMALL_GLUON", "exact_class")
GLUON_BY_M = _tables.SM86.gluon_small
GLUON_RING_BY_M = _tables.SM86.gluon_ring
GLUON_EXACT = _tables.SM86.gluon_exact


def gluon_for(m, n, k):
    """("direct", cfg) for pq2_gluon_small, ("ring", cfg) for pq2_gluon_ring, or None."""
    if not _GLUON:
        return None
    from . import pq2_gluon_small
    if not pq2_gluon_small.AVAILABLE:
        return None
    if _config.KERNELS == "exact" and (m, n, k) not in GLUON_EXACT:
        return None
    cfg = GLUON_BY_M.get(m, {}).get((n, k))
    if cfg is not None:
        return "direct", cfg
    cfg = GLUON_RING_BY_M.get(m, {}).get((n, k))
    return None if cfg is None else ("ring", cfg)


def _ring_available():
    return (_cuda_available() and hasattr(torch.ops._C_gguf, "pq2_mma_ring"))


def ring_for(m, n, k):
    """(cfg, split) for the ring kernel, or None."""
    if not _RING or not RING_BY_M or not _ring_available():
        return None
    if _RING_EXACT_ONLY and (m, n, k) not in RING_EXACT:
        return None
    return RING_BY_M.get(m, {}).get((n, k))


def _ring_workspace(device, allocate=True):
    """Persistent split-K scratch: FP32 partials and zero-initialised self-resetting tile counters.

    One workspace per device, shared by every split-K launch, so all ring launches on a device must be ordered on one
    stream (true for the engine's forward pass and for CUDA-graph replay; two split-K launches overlapping on different
    streams would corrupt each other). The counters must be zero on entry, so the workspace is never allocated while a
    CUDA graph is being captured (the zero fill would be captured instead of executed, and the memory would belong to
    the graph's private pool): it is created on first eager use, which the engine's warm-up always precedes capture
    with. Returns None when it does not exist and cannot be created now.
    """
    ws = _RING_WS.get(device)
    if ws is None and allocate and not torch.cuda.is_current_stream_capturing():
        partial = max((sp * m * n for m, t in RING_BY_M.items() for (n, _k), (_c, sp) in t.items() if sp > 1),
                      default=1)
        # One counter per row block of the narrowest tile (16 rows); the kernel indexes them by blockIdx.x.
        blocks = max((-(-n // 16) for t in RING_BY_M.values() for (n, _k) in t), default=1)
        ws = (torch.empty(partial, dtype=torch.float32, device=device),
              torch.zeros(max(blocks, 1), dtype=torch.int32, device=device))
        _RING_WS[device] = ws
    return ws


def _pq2_mma_ring(x, weight, cfg, split):
    from .pq2_mmq import _quantize_prefill

    m, k = x.shape
    n = weight.shape[0]
    x = x.contiguous()
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
    t = torch.empty((m, k // 128), dtype=torch.int32, device=x.device)
    groups = m * (k // 128)
    _quantize_prefill[(triton.cdiv(groups, 8),)](x, q, s, t, groups, 8, num_warps=4)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    partial, counters = _ring_workspace(x.device)  # existence checked by the caller
    torch.ops._C_gguf.pq2_mma_ring(q, s, t, weight, y, partial, counters, cfg, split)
    return y


def cuda_for(m, n, k):
    """(warps, groups in flight, row tiles per warp) for the CUDA MMA kernel, or None."""
    if not _cuda_available():
        return None
    if _CUDA_RT:
        rt = CUDA_RT_BY_M.get(m, {}).get((n, k))
        if rt is not None:
            return rt
    if _SMALL_MMQ_OUTPUT and (n, k) == (5120, 6144) and 2 <= m <= 8:
        return (*OUTPUT_CUDA, 1)
    base = CUDA_BY_M.get(m, {}).get((n, k))
    return None if base is None else (*base, 1)


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
                      SPLIT: tl.constexpr = 1, BT: tl.constexpr = 16):
    """Codes stay in place: plane s is `byte & (3 << 2s)` = code * 4^s (plane 3 pre-shifted
    by 2 to stay below 128). The exact INT32 product is shifted back, and the ternary -1
    offset is applied once per group as -sum(q). Integer result equals the DP4A kernel's."""
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
    enabled = _SMALL_MMQ and 2 <= m <= _MAX_M
    return (
        enabled
        and x.dtype in (torch.bfloat16, torch.float16)
        and (geometry_for(m, weight.shape[0], x.shape[1]) is not None
             or gluon_for(m, weight.shape[0], x.shape[1]) is not None
             or cuda_for(m, weight.shape[0], x.shape[1]) is not None
             or ring_for(m, weight.shape[0], x.shape[1]) is not None)
        and torch.cuda.get_device_capability(x.device) == (8, 6)
    )


def pq2_small_mmq(x, weight, geometry=None):
    """x: (M <= 64, K) floating, already rotated; weight: prepared PQ2 planes."""
    global _CALLS
    _CALLS += 1
    m, k = x.shape
    n = weight.shape[0]
    gl = None if geometry is not None else gluon_for(m, n, k)
    if gl is not None:
        if gl[0] == "ring":
            from .pq2_gluon_ring import pq2_gluon_ring
            return pq2_gluon_ring(x, weight, *gl[1])
        from .pq2_gluon_small import pq2_gluon_small
        return pq2_gluon_small(x, weight, *gl[1])
    ring = None if geometry is not None else ring_for(m, n, k)
    if ring is not None and ring[1] > 1 and _ring_workspace(x.device) is None:
        ring = None  # first split-K call happens inside graph capture: take the previous path instead
    if ring is not None:
        return _pq2_mma_ring(x, weight, *ring)
    cuda = None if geometry is not None else cuda_for(m, n, k)
    if cuda is not None:
        return _pq2_mma_small(x, weight, *cuda)
    bn, g, warps, stages, split = geometry or geometry_for(m, n, k)
    if not 1 <= m <= _MAX_M or k % 128 or weight.shape[1] != k // 128 * 34:
        raise ValueError("PQ2 small MMQ requires 1 to 64 tokens and a PQ2 matrix")
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
        q, s, t, codes, scales, target, m, n, k, bn, g, stages, split, _token_tile(m),
        num_warps=warps,
    )
    if split > 1:
        _reduce_splits[(triton.cdiv(m * n, 1024),)](target, y, m * n, split, 1024)
    return y


def _pq2_mma_small(x, weight, warps, unroll, rt=1):
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
    torch.ops._C_gguf.pq2_mma_small(q, s, t, weight, y, warps, unroll, rt)
    return y
