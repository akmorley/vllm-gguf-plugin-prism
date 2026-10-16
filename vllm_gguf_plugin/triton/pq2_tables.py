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
)

BY_CAPABILITY = {(8, 6): SM86}
EMPTY = Tables()


def for_capability(capability):
    """Tables for a (major, minor) compute capability; empty tables (every kernel falls back) for unknown cards."""
    return BY_CAPABILITY.get(tuple(capability), EMPTY)
