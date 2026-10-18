# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest

from vllm_gguf_plugin.triton import pq2_config, pq2_tables

SWITCHES = {
    "GGUF_PQ2_MMQ_TUNED_TILES": "exact_class",
    "GGUF_PQ2_PREFILL_ROWS": "exact_class",
    "GGUF_PQ2_MMA_PREFILL": "exact_class",
    "GGUF_PQ2_SMALL_MMQ_RT": "exact_class",
    "GGUF_PQ2_SMALL_MMQ_RING": "exact_class",
    "GGUF_PQ2_MMQ_OUTPUT": "numerics_class",
    "GGUF_PQ2_M1_CUDA": "numerics_class",
    "GGUF_PQ2_SMALL_MMQ_RING_EXACT": "exact_only_ring",
    "GGUF_PQ2_MMQ_GLUON": "exact_class",
    "GGUF_PQ2_SMALL_GLUON": "exact_class",
    "GGUF_PQ2_M1_TRITON": "numerics_class",
    "GGUF_SMALL_GEMM_TRITON": "numerics_class",
}
EXPECTED = {
    "tuned": {"exact_class": True, "numerics_class": True, "exact_only_ring": False},
    "exact": {"exact_class": True, "numerics_class": False, "exact_only_ring": True},
    "legacy": {"exact_class": False, "numerics_class": False, "exact_only_ring": False},
}


@pytest.fixture
def config(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    yield lambda kernels=None: _reload(monkeypatch, kernels)
    monkeypatch.delenv("GGUF_PQ2_KERNELS", raising=False)
    importlib.reload(pq2_config)


def _reload(monkeypatch, kernels):
    if kernels is None:
        monkeypatch.delenv("GGUF_PQ2_KERNELS", raising=False)
    else:
        monkeypatch.setenv("GGUF_PQ2_KERNELS", kernels)
    return importlib.reload(pq2_config)


def test_default_is_tuned(config):
    assert config().KERNELS == "tuned"


@pytest.mark.parametrize("kernels", sorted(EXPECTED))
def test_each_set_resolves_every_switch(config, kernels):
    cfg = config(kernels)
    for name, kind in SWITCHES.items():
        assert cfg.flag(name, kind) is EXPECTED[kernels][kind], (kernels, name)


def test_explicit_switch_overrides_the_set(config, monkeypatch):
    cfg = config("legacy")
    monkeypatch.setenv("GGUF_PQ2_M1_CUDA", "1")
    assert cfg.flag("GGUF_PQ2_M1_CUDA", "numerics_class") is True
    cfg = config("tuned")
    monkeypatch.setenv("GGUF_PQ2_SMALL_MMQ_RING", "0")
    assert cfg.flag("GGUF_PQ2_SMALL_MMQ_RING", "exact_class") is False


def test_unknown_set_is_rejected(config):
    with pytest.raises(ValueError):
        config("fast")


def test_ring_exact_entries_are_split_one_and_in_the_table():
    t = pq2_tables.SM86
    for m, n, k in t.ring_exact:
        cfg, split = t.ring_by_m[m][(n, k)]
        assert split == 1, (m, n, k)


def test_tables_are_keyed_by_architecture():
    assert pq2_tables.for_capability((8, 6)) is pq2_tables.SM86
    other = pq2_tables.for_capability((9, 0))
    assert not other.rt_by_m and not other.ring_by_m and not other.m1_cfg


def test_modules_alias_the_sm86_tables():
    from vllm_gguf_plugin.triton import pq2_int_gemv, pq2_small_mmq

    assert pq2_small_mmq.CUDA_RT_BY_M is pq2_tables.SM86.rt_by_m
    assert pq2_small_mmq.RING_BY_M is pq2_tables.SM86.ring_by_m
    assert pq2_small_mmq.RING_EXACT is pq2_tables.SM86.ring_exact
    assert pq2_int_gemv._M1_CFG is pq2_tables.SM86.m1_cfg


def test_table_override(tmp_path, monkeypatch):
    """GGUF_PQ2_TABLES replaces / removes cells of the SM86 tables and sets OVERRIDE_HASH; without it nothing changes."""
    import json
    base = pq2_tables.SM86
    data = {
        "m1_triton": {"34816x5120": [8, 8, 1, 1], "5120x6144": None},
        "gluon_small": {"4": {"34816x5120": [2, 2, 4, 1]}},           # moves a ring cell to the direct table
        "gluon_ring": {"16": {"5120x6144": None}, "17": {"5120x6144": [4, 2, 4, 3, 1]}},
        "gluon_exact": {"add": [[4, 34816, 5120]], "remove": [[2, 5120, 17408]]},
        "gluon_prefill": {"5120x6144": [[1073741824, [4, 2, 4, 4, 2, 2]]], "5120x17408": None},
    }
    path = tmp_path / "override.json"
    path.write_text(json.dumps(data))
    t = pq2_tables._apply_override(base, json.loads(path.read_text()))
    assert t.m1_triton[(34816, 5120)] == (8, 8, 1, 1) and (5120, 6144) not in t.m1_triton
    assert t.gluon_small[4][(34816, 5120)] == (2, 2, 4, 1) and (34816, 5120) not in t.gluon_ring[4]
    assert (5120, 6144) not in t.gluon_ring[16] and t.gluon_ring[17][(5120, 6144)] == (4, 2, 4, 3, 1)
    assert (4, 34816, 5120) in t.gluon_exact and (2, 5120, 17408) not in t.gluon_exact
    assert t.gluon_prefill[(5120, 6144)] == ((1 << 30, (4, 2, 4, 4, 2, 2)),) and (5120, 17408) not in t.gluon_prefill
    # the base tables are not modified
    assert (34816, 5120) in base.gluon_ring[4] and (5120, 6144) in base.m1_triton and (5120, 6144) in base.gluon_ring[16]
    monkeypatch.setenv("GGUF_PQ2_TABLES", str(path))
    reloaded = importlib.reload(pq2_tables)
    assert reloaded.OVERRIDE_HASH and reloaded.SM86.m1_triton[(34816, 5120)] == (8, 8, 1, 1)
    monkeypatch.delenv("GGUF_PQ2_TABLES")
    reloaded = importlib.reload(pq2_tables)
    assert reloaded.OVERRIDE_HASH == "" and reloaded.SM86.m1_triton[(34816, 5120)] == (16, 8, 4, 1)
