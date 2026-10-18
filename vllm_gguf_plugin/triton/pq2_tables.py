# SPDX-License-Identifier: Apache-2.0
"""Per-architecture dispatch tables for the PQ2 CUDA kernels.

Everything that was tuned for a specific GPU lives here, keyed by compute capability, so retuning for another card
(for example the 48-SM RTX A4000) means adding a ``Tables`` entry instead of editing the dispatch code. Tables were
selected on an RTX 3090 Ti (84 SMs, 300 W) against real captured decode rows; see
pq2-dev/benchmarks/results/pq2-decode-kernel-design-20261015 and pq2-batched-decode-longctx-20261015.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field, replace

# (N, K) shapes shared by several tables.
_RT_WIDE = {(34816, 5120): (2, 8, 2), (16384, 5120): (4, 8, 2), (14336, 5120): (2, 8, 2)}
_RING_DOWN_OUT = {(5120, 17408): (1, 2), (5120, 6144): (1, 2)}
_RING_WIDE = {(5120, 17408): (1, 2), (5120, 6144): (6, 1), (14336, 5120): (9, 1)}


@dataclass(frozen=True)
class Tables:
    # M -> (N, K) -> (warps, groups in flight, row tiles per warp) for pq2_mma_small. Only entries >= 4% faster than
    # the one-tile-per-warp kernel are listed. Not listed: 5120 x 17408 and 5120 x 6144 (320 row tiles).
    rt_by_m: dict = field(default_factory=dict)
    # M -> (N, K) -> (cfg, split) for pq2_mma_ring. cfg selects the CTA tile; split > 1 is split-K with a fused
    # fixed-order reduction (deterministic).
    ring_by_m: dict = field(default_factory=dict)
    # (M, N, K) ring entries that sum in the same order as the path they replace (split 1): bit-identical outputs.
    ring_exact: frozenset = frozenset()
    # (N, K) -> kernel config of pq2_gemv_m1 for single-token decode.
    m1_cfg: dict = field(default_factory=dict)
    # (N, K) -> (rows per program, groups per iteration, warps, stages) of the Triton plane-DP4A single-token kernel.
    m1_triton: dict = field(default_factory=dict)
    # (N, K) -> ((largest M, (BM, BN, warps, stages), faster than pq2_mma_prefill), ...) for the Triton masked-plane
    # prefill kernel (pq2_mmq._mmq_planes), M buckets ascending; the last bucket covers every larger M.
    prefill_planes: dict = field(default_factory=dict)
    # M -> (N, K) -> (warps, row tiles per warp, groups per iteration, K splits) for the Gluon direct-fragment kernel
    # (pq2_gluon_small), preferred over every CUDA small-batch kernel where listed.
    gluon_small: dict = field(default_factory=dict)
    # M -> (N, K) -> (warps, row tiles per warp, groups per stage, stages, K splits) for the Gluon cp.async ring kernel
    # (pq2_gluon_ring); a cell is in at most one of gluon_small / gluon_ring.
    gluon_ring: dict = field(default_factory=dict)
    # (N, K) -> ((largest M, (WR, WT, RT, NT, SG, ST)), ...) for the Gluon prefill kernel (pq2_gluon_prefill), M buckets
    # ascending; the last bucket covers every larger M.
    gluon_prefill: dict = field(default_factory=dict)
    # (M, N, K) Gluon entries whose output is bit-identical to the path they replace.
    gluon_exact: frozenset = frozenset()


SM86 = Tables(
    rt_by_m={
        2: {(16384, 5120): (2, 8, 2), (14336, 5120): (2, 8, 2)},
        3: {**_RT_WIDE, (248320, 5120): (4, 8, 2)},
        4: {**_RT_WIDE, (248320, 5120): (4, 8, 2)},
        5: {**_RT_WIDE, (248320, 5120): (4, 8, 2)},
        6: {**_RT_WIDE, (34816, 5120): (2, 4, 4), (248320, 5120): (4, 8, 2)},
        7: {**_RT_WIDE, (34816, 5120): (2, 4, 4), (248320, 5120): (4, 8, 2)},
        8: {**_RT_WIDE, (248320, 5120): (4, 8, 2)},
        9: {(248320, 5120): (4, 4, 2)},
        **{m: {(34816, 5120): (4, 4, 2), (248320, 5120): (4, 2, 4) if m % 2 else (2, 4, 2)} for m in range(10, 17)},
    },
    ring_by_m={
        2: {(5120, 17408): (1, 2), (5120, 6144): (9, 4)},
        3: _RING_DOWN_OUT,
        4: _RING_DOWN_OUT,
        5: _RING_DOWN_OUT,
        6: {(5120, 17408): (1, 2), (5120, 6144): (6, 1)},
        7: _RING_WIDE,
        8: _RING_WIDE,
    },
    ring_exact=frozenset({(6, 5120, 6144), (7, 5120, 6144), (8, 5120, 6144), (7, 14336, 5120), (8, 14336, 5120)}),
    m1_cfg={(34816, 5120): 0, (16384, 5120): 1, (14336, 5120): 1, (5120, 17408): 2, (5120, 6144): 3, (248320, 5120): 4},
    # Tuned under the serving power proxy (pq2-m1-triton-planes-20261017/tune/m1.log): 0.94-0.98 of the CUDA kernel.
    # lm_head is at parity; the matmul-only sweep (power-capped, ~1.4 GHz) had picked other tiles for four shapes.
    m1_triton={(34816, 5120): (16, 8, 4, 1), (16384, 5120): (8, 8, 1, 1), (14336, 5120): (8, 8, 1, 1),
               (5120, 17408): (16, 8, 4, 1), (5120, 6144): (8, 8, 1, 1), (248320, 5120): (8, 8, 1, 1)},
    # Both Gluon kernels against production under the serving power proxy (spin kernel after each projection, per-kernel
    # times from nsys; pq2-m1-triton-planes-20261017/tune/m*.log, tune/ring-m*.log); the faster Gluon kernel of a cell is
    # listed when it is < 0.98 of production (and qkv at M = 2, 3, below).
    # qkv at M = 2, 3: 0.99 of production (tune/m2.log, m3.log); listed so that no M = 1..8 decode cell uses CUDA.
    gluon_small={2: {(34816, 5120): (2, 2, 4, 1), (16384, 5120): (2, 2, 4, 1), (14336, 5120): (2, 2, 4, 1)},
                 3: {(14336, 5120): (2, 2, 4, 1)}},
    gluon_ring={
        2: {(5120, 6144): (8, 1, 4, 3, 2), (5120, 17408): (4, 2, 4, 2, 2), (248320, 5120): (4, 2, 4, 3, 1)},
        3: {(5120, 6144): (4, 2, 4, 3, 1), (5120, 17408): (4, 2, 4, 2, 2), (16384, 5120): (4, 2, 4, 3, 1), (34816, 5120): (4, 2, 4, 3, 1), (248320, 5120): (4, 2, 4, 3, 1)},
        4: {(5120, 6144): (4, 2, 4, 3, 1), (5120, 17408): (4, 2, 4, 2, 2), (14336, 5120): (4, 2, 4, 2, 2), (16384, 5120): (4, 2, 4, 3, 1), (34816, 5120): (4, 2, 4, 3, 1),
            (248320, 5120): (4, 2, 4, 3, 1)},
        **{m: {(5120, 6144): (4, 2, 4, 3, 1), (5120, 17408): (4, 2, 4, 3, 1), (14336, 5120): (4, 2, 4, 2, 2), (16384, 5120): (4, 2, 4, 3, 1),
                (34816, 5120): (4, 2, 4, 3, 1), (248320, 5120): (4, 2, 4, 3, 1)} for m in (6, 7, 8)},
        5: {(5120, 6144): (4, 2, 4, 3, 1), (5120, 17408): (4, 2, 4, 3, 1), (14336, 5120): (4, 2, 4, 3, 1), (16384, 5120): (4, 2, 4, 3, 1), (34816, 5120): (4, 2, 4, 3, 1),
            (248320, 5120): (4, 2, 4, 3, 1)},
        # M = 9..16 (16-token tile): ring kernel on every shape, 0.33-0.84 of production (tune/t0-ring-m12.log, m16.log);
        # 9..12 from the M = 12 run, 13..16 from M = 16.
        **{m: {(5120, 6144): (4, 2, 4, 2, 2), (5120, 17408): (4, 2, 4, 2, 2), (14336, 5120): (4, 2, 4, 2, 2), (16384, 5120): (4, 2, 4, 3, 1), (34816, 5120): (4, 2, 4, 3, 1),
                (248320, 5120): (4, 2, 4, 3, 1)} for m in range(9, 13)},
        **{m: {(5120, 6144): (4, 2, 2, 4, 2), (5120, 17408): (4, 2, 4, 2, 2), (14336, 5120): (4, 2, 4, 2, 2), (16384, 5120): (4, 2, 4, 3, 1),
                (34816, 5120): (4, 2, 4, 3, 1), (248320, 5120): (4, 2, 4, 3, 1)} for m in range(13, 17)},
    },
    # Bit-identical to production on the real rows (gluon/exact_check_v3.log + exact_check_qkv.log: 30 of 42). The other twelve use another K
    # split than the path they replace (FP32 order, relative RMS <= 1.3e-4): out M = 2..5, down M = 5..8, qkv M = 4, 6..8.
    gluon_exact=frozenset({
        (2, 5120, 17408), (2, 14336, 5120), (2, 16384, 5120), (2, 34816, 5120), (2, 248320, 5120),
        (3, 14336, 5120), (3, 5120, 17408), (3, 16384, 5120), (3, 34816, 5120), (3, 248320, 5120),
        (4, 5120, 17408), (4, 16384, 5120), (4, 34816, 5120), (4, 248320, 5120),
        (5, 14336, 5120), (5, 16384, 5120), (5, 34816, 5120), (5, 248320, 5120),
        (6, 5120, 6144), (6, 16384, 5120), (6, 34816, 5120), (6, 248320, 5120),
        (7, 5120, 6144), (7, 16384, 5120), (7, 34816, 5120), (7, 248320, 5120),
        (8, 5120, 6144), (8, 16384, 5120), (8, 34816, 5120), (8, 248320, 5120),
        # M = 9..16 (m16/numerics16.log): down, in_proj, gate/up, lm_head bit-identical to production; qkv FP32-order
        # (split 2), out ~0.6% (production computes it from floating activations above M = 8)
        *((m, n, k) for m in range(9, 17) for (n, k) in ((5120, 17408), (16384, 5120), (34816, 5120), (248320, 5120))),
    }),
    # Sustained loops at M = 128 / 512 / 1024 / 2048, real weights: 0.85-0.98 of the CUDA prefill kernel in every cell
    # (pq2-m1-triton-planes-20261017/gluon_prefill/sweep_*_all.log); bit-exact with it. In serving: prefill +7-9% tok/s,
    # -7% energy per token (serve/prefill_energy_g*.log).
    gluon_prefill={
        (5120, 6144): ((255, (4, 2, 2, 4, 2, 3)), (1 << 30, (4, 2, 4, 4, 2, 2))),
        (5120, 17408): ((255, (4, 1, 2, 8, 2, 2)), (1 << 30, (4, 2, 4, 4, 2, 2))),
        (14336, 5120): ((255, (4, 1, 2, 8, 2, 2)), (1535, (4, 2, 4, 4, 2, 2)), (1 << 30, (2, 4, 4, 4, 2, 2))),
        (16384, 5120): ((255, (4, 2, 4, 4, 2, 2)), (1 << 30, (2, 4, 4, 4, 2, 2))),
        (34816, 5120): ((1 << 30, (2, 4, 4, 4, 2, 2)),),
    },
    # Interleaved sustained runs against the CUDA kernel at M = 128 / 512 / 1024 / 2048, real weights; "faster" means
    # < 0.98 of its time (pq2-m1-triton-planes-20261017/confirm_prefill.log).
    prefill_planes={
        (5120, 6144): ((255, (32, 256, 8, 3), True), (767, (64, 256, 8, 2), False), (1535, (64, 256, 8, 2), False),
                       (1 << 30, (64, 256, 8, 3), False)),
        (5120, 17408): ((255, (64, 128, 4, 3), True), (767, (64, 256, 8, 2), False), (1535, (64, 256, 8, 2), False),
                        (1 << 30, (64, 256, 8, 2), False)),
        (14336, 5120): ((255, (64, 128, 4, 3), True), (767, (64, 256, 8, 3), True), (1535, (64, 256, 8, 3), True),
                        (1 << 30, (64, 256, 8, 3), True)),
        (16384, 5120): ((255, (64, 256, 8, 2), False), (767, (64, 256, 8, 2), True), (1535, (64, 256, 8, 3), True),
                        (1 << 30, (64, 256, 8, 3), True)),
        (34816, 5120): ((255, (64, 256, 8, 2), False), (767, (64, 256, 8, 3), True), (1535, (64, 256, 8, 3), True),
                        (1 << 30, (64, 256, 8, 3), True)),
    },
)


# ---------------------------------------------------------------------------------------------------------------------
# Per-card overrides. Cards of one compute capability can still want different tiles (the 48-SM, 140 W RTX A4000 against
# the 84-SM RTX 3090 Ti these tables were tuned on), so GGUF_PQ2_TABLES may name a JSON file written by the A4000 tuner
# (pq2-dev/deploy/a4000/tune/tune_a4000.py) that replaces or removes cells of the SM86 tables:
#
#   {"m1_triton":    {"34816x5120": [16, 8, 4, 1], ...},                  # null removes a cell (CUDA kernel)
#    "gluon_small":  {"2": {"34816x5120": [2, 2, 4, 1]}, ...},
#    "gluon_ring":   {"9": {"5120x6144": [4, 2, 4, 2, 2]}, ...},
#    "gluon_exact":  {"add": [[2, 34816, 5120], ...], "remove": [...]},
#    "gluon_prefill": {"5120x6144": [[255, [4, 2, 2, 4, 2, 3]], [1073741824, [4, 2, 4, 4, 2, 2]]], ...}}
#
# OVERRIDE_HASH (sha256 of the file, or "") enters the compile identity.
# ---------------------------------------------------------------------------------------------------------------------
def _shape(key):
    n, k = key.split("x")
    return int(n), int(k)


def _apply_override(base, data):
    """Tables with the cells of a tuner override replaced or removed (null)."""
    changes = {}
    if "m1_triton" in data:
        m1 = dict(base.m1_triton)
        for key, cfg in data["m1_triton"].items():
            if cfg is None:
                m1.pop(_shape(key), None)
            else:
                m1[_shape(key)] = tuple(cfg)
        changes["m1_triton"] = m1
    for name in ("gluon_small", "gluon_ring"):
        if name in data:
            table = {m: dict(cells) for m, cells in getattr(base, name).items()}
            for m, cells in data[name].items():
                row = table.setdefault(int(m), {})
                for key, cfg in cells.items():
                    if cfg is None:
                        row.pop(_shape(key), None)
                    else:
                        row[_shape(key)] = tuple(cfg)
            changes[name] = {m: row for m, row in table.items() if row}
    if "gluon_exact" in data:
        exact = set(base.gluon_exact)
        exact |= {tuple(c) for c in data["gluon_exact"].get("add", ())}
        exact -= {tuple(c) for c in data["gluon_exact"].get("remove", ())}
        changes["gluon_exact"] = frozenset(exact)
    if "gluon_prefill" in data:
        pf = dict(base.gluon_prefill)
        for key, buckets in data["gluon_prefill"].items():
            if buckets is None:
                pf.pop(_shape(key), None)
            else:
                pf[_shape(key)] = tuple((int(m), tuple(cfg)) for m, cfg in buckets)
        changes["gluon_prefill"] = pf
    # a cell may sit in at most one of the two Gluon small-batch tables; the override's table wins
    if "gluon_small" in data or "gluon_ring" in data:
        small = {m: dict(c) for m, c in changes.get("gluon_small", base.gluon_small).items()}
        ring = {m: dict(c) for m, c in changes.get("gluon_ring", base.gluon_ring).items()}
        for m, cells in data.get("gluon_small", {}).items():
            for key, cfg in cells.items():
                if cfg is not None and _shape(key) in ring.get(int(m), {}):
                    ring[int(m)].pop(_shape(key))
        for m, cells in data.get("gluon_ring", {}).items():
            for key, cfg in cells.items():
                if cfg is not None and _shape(key) in small.get(int(m), {}):
                    small[int(m)].pop(_shape(key))
        changes["gluon_small"], changes["gluon_ring"] = small, ring
    return replace(base, **changes)


OVERRIDE_PATH = os.environ.get("GGUF_PQ2_TABLES", "")
OVERRIDE_HASH = ""
if OVERRIDE_PATH:
    with open(OVERRIDE_PATH, "rb") as f:
        _raw = f.read()
    OVERRIDE_HASH = hashlib.sha256(_raw).hexdigest()[:16]
    SM86 = _apply_override(SM86, json.loads(_raw))

BY_CAPABILITY = {(8, 6): SM86}
EMPTY = Tables()


def for_capability(capability):
    """Tables for a (major, minor) compute capability; empty tables (every kernel falls back) for unknown cards."""
    return BY_CAPABILITY.get(tuple(capability), EMPTY)
