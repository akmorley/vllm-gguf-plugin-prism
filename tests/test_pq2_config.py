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
