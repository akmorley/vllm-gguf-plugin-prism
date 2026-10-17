# SPDX-License-Identifier: Apache-2.0
"""Per-architecture dispatch tables for the PQ2 CUDA kernels.

Everything that was tuned for a specific GPU lives here, keyed by compute capability, so retuning for another card
(for example the 48-SM RTX A4000) means adding a ``Tables`` entry instead of editing the dispatch code. Tables were
selected on an RTX 3090 Ti (84 SMs, 300 W) against real captured decode rows; see
pq2-dev/benchmarks/results/pq2-decode-kernel-design-20261015 and pq2-batched-decode-longctx-20261015.
"""

from dataclasses import dataclass, field

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
    # Register-prefetch Gluon kernel vs production under the serving power proxy (spin kernel after each projection,
    # per-kernel times from nsys; pq2-m1-triton-planes-20261017/tune/): only cells < 0.98 of production are listed.
    # Matmul-only replays ran power-capped at ~1.4 GHz and picked different (worse in serving) configs.
    gluon_small={
        2: {(16384, 5120): (2, 2, 4, 1), (5120, 6144): (4, 1, 4, 2), (34816, 5120): (2, 2, 4, 1), (5120, 17408): (8, 1, 4, 2),
            (248320, 5120): (2, 2, 4, 1)},
        3: {(16384, 5120): (2, 2, 4, 1), (5120, 6144): (8, 1, 4, 2), (34816, 5120): (2, 2, 4, 1), (5120, 17408): (8, 1, 4, 2)},
        4: {(16384, 5120): (2, 2, 4, 1), (5120, 6144): (8, 1, 4, 2), (34816, 5120): (2, 2, 4, 1), (5120, 17408): (8, 1, 4, 2)},
        5: {(16384, 5120): (2, 2, 4, 1), (5120, 6144): (8, 1, 4, 2), (34816, 5120): (2, 2, 4, 1)},
        6: {(16384, 5120): (2, 2, 4, 1), (5120, 6144): (8, 2, 2, 4), (34816, 5120): (2, 2, 4, 1)},
        7: {(14336, 5120): (2, 2, 4, 1), (5120, 6144): (8, 2, 2, 4), (34816, 5120): (2, 2, 4, 1), (248320, 5120): (2, 2, 4, 1)},
        8: {(14336, 5120): (2, 2, 4, 1), (5120, 6144): (8, 2, 2, 4), (34816, 5120): (2, 2, 4, 1)},
    },
    # Bit-identical to production on the real rows (gluon/exact_check_v2.log: 22 of 26); the other four are `out` at
    # M = 2, 6, 7, 8, whose K split differs from production's (FP32 order, relative RMS <= 8.3e-5).
    gluon_exact=frozenset({
        (2, 5120, 17408), (2, 16384, 5120), (2, 34816, 5120), (2, 248320, 5120),
        (3, 5120, 6144), (3, 5120, 17408), (3, 16384, 5120), (3, 34816, 5120),
        (4, 5120, 6144), (4, 5120, 17408), (4, 16384, 5120), (4, 34816, 5120),
        (5, 5120, 6144), (5, 16384, 5120), (5, 34816, 5120),
        (6, 16384, 5120), (6, 34816, 5120),
        (7, 14336, 5120), (7, 34816, 5120), (7, 248320, 5120),
        (8, 14336, 5120), (8, 34816, 5120),
    }),
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

BY_CAPABILITY = {(8, 6): SM86}
EMPTY = Tables()


def for_capability(capability):
    """Tables for a (major, minor) compute capability; empty tables (every kernel falls back) for unknown cards."""
    return BY_CAPABILITY.get(tuple(capability), EMPTY)
