# SPDX-License-Identifier: Apache-2.0
"""Kernel-set selection for the PQ2 fast paths.

``GGUF_PQ2_KERNELS`` picks a whole set at once:

``tuned`` (default)
    everything below.
``exact``
    only the paths whose outputs are bit-identical to the ``legacy`` kernels: shape-adaptive MMQ tiles, batched
    Hadamard/quantize rows, the INT8 tensor-core prefill kernel, register-tiled small-batch rows, and the ring-kernel
    entries that sum in the same order as the path they replace.
``legacy``
    none of them (the kernels as of plugin revision e52b0af).

The ``tuned`` set additionally contains the paths that change the numerics at the rounding level or below: the ring
kernel's split-K entries, the single-token CUDA kernel (different FP32 group order, relative RMS <= 1.4e-4), and the
(5120, 6144) output projections on INT8 activations (~0.7% per call).

Each path keeps its own switch (``GGUF_PQ2_MMQ_TUNED_TILES`` and so on) as an override of the selected set, for
debugging and A/B runs: an explicitly set variable always wins.
"""

import os

KERNEL_SETS = ("tuned", "exact", "legacy")
KERNELS = os.environ.get("GGUF_PQ2_KERNELS", "tuned")
if KERNELS not in KERNEL_SETS:
    raise ValueError(f"GGUF_PQ2_KERNELS must be one of {KERNEL_SETS}, got {KERNELS!r}")

# Bumped when a kernel or table in the set changes; part of the compile identity (see plugin.py).
KERNEL_VERSION = "pq2-kernels-2026-10-15"

_ON = {
    "exact_class": {"tuned": True, "exact": True, "legacy": False},
    "numerics_class": {"tuned": True, "exact": False, "legacy": False},
    "exact_only_ring": {"tuned": False, "exact": True, "legacy": False},
}


def flag(name, kind):
    """Boolean for the env switch ``name``: an explicit "0"/"1" wins, otherwise the selected kernel set decides.

    kind: "exact_class" (bit-identical to legacy), "numerics_class" (changes rounding or numerics) or
    "exact_only_ring" (restricts the ring kernel to its bit-identical entries).
    """
    value = os.environ.get(name)
    if value is not None:
        return value == "1"
    return _ON[kind][KERNELS]
