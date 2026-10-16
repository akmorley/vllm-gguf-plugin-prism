# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gguf_plugin.triton import pq2_mmq, prism


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prepared", [False, True])
def test_mmq_dispatch_measured_prefill(monkeypatch, prepared):
    monkeypatch.setattr(prism, "_EXPERIMENTAL_MMQ", True)
    monkeypatch.setattr(prism, "_EXPERIMENTAL_INT_GEMV", False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    x = torch.zeros((128, 5120), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((14336, 1360), device="cuda", dtype=torch.uint8)
    result = object()
    calls = []

    def mmq(a, b, *, prepared=False, activation_group=128):
        calls.append((a.shape, prepared, activation_group))
        return result

    monkeypatch.setattr(pq2_mmq, "pq2_mmq", mmq)
    assert prism.pq2_matmul(x, w, prepared=prepared) is result
    assert calls == [(x.shape, prepared, 128)]
    for size in (1, 4, 127):
        assert prism.pq2_matmul(x[:size], w).shape == (size, 14336)
    monkeypatch.setattr(prism, "_EXPERIMENTAL_MMQ", False)
    assert prism.pq2_matmul(x, w).shape == (128, 14336)
    monkeypatch.setattr(prism, "_EXPERIMENTAL_MMQ", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert prism.pq2_matmul(x, w).shape == (128, 14336)
    assert len(calls) == 1


def test_mmq_cache_factor(monkeypatch):
    import vllm.envs as envs
    from vllm.config.utils import hash_factors

    from vllm_gguf_plugin import plugin

    monkeypatch.setattr(envs, "environment_variables", {})
    plugin._register_pq2_compile_factors()
    monkeypatch.setattr(prism, "_EXPERIMENTAL_MMQ", False)
    raw = hash_factors(envs.compile_factors())
    monkeypatch.setattr(prism, "_EXPERIMENTAL_MMQ", True)
    assert hash_factors(envs.compile_factors()) != raw
    assert (
        envs.environment_variables["GGUF_PQ2_MMQ_VERSION"]()
        == "q8-groups-m128-byte32-v3"
    )

