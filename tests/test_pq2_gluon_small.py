# SPDX-License-Identifier: Apache-2.0
"""Gluon direct-fragment small-batch kernel versus the CUDA pq2_mma_small kernel, and its dispatch."""
import pytest
import torch

from vllm_gguf_plugin.triton import pq2_config, pq2_gluon_small as gl, pq2_small_mmq as sm
from vllm_gguf_plugin.triton.pq2_layout import prepare_pq2_layout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6) or not gl.AVAILABLE,
    reason="SM86 CUDA and Triton Gluon required",
)
_WEIGHTS = {}


def _weight(n, k):
    if (n, k) not in _WEIGHTS:
        torch.manual_seed(11)
        raw = torch.randint(0, 256, (n, k // 128, 34), device="cuda", dtype=torch.uint8)
        raw[:, :, :2] = (torch.rand(n, k // 128, device="cuda") * 0.03 + 0.001).half().view(torch.uint8).reshape(
            n, k // 128, 2)
        _WEIGHTS[(n, k)] = prepare_pq2_layout(raw.reshape(n, -1))
    return _WEIGHTS[(n, k)]


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("cfg", [(2, 1, 8, 1), (4, 1, 4, 1), (8, 1, 2, 1), (2, 2, 4, 1), (4, 2, 8, 1)])
@pytest.mark.parametrize("m", [1, 2, 3, 5, 8])
def test_split1_bit_identical_to_cuda(m, cfg):
    if not sm._cuda_available():
        pytest.skip("pq2_mma_small not built")
    n, k = 4096 + 48, 5120  # N not a tile multiple
    w = _weight(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(gl.pq2_gluon_small(x, w, *cfg), sm._pq2_mma_small(x, w, 2, 4, 1))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [2, 4, 6])
def test_split_k_deterministic(m, dtype):
    if not sm._cuda_available():
        pytest.skip("pq2_mma_small not built")
    n, k = 5120, 6144
    w = _weight(n, k)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    first = gl.pq2_gluon_small(x, w, 2, 1, 2, 4)
    assert torch.equal(first, gl.pq2_gluon_small(x, w, 2, 1, 2, 4))
    assert torch.isfinite(first).all()
    assert _rel(first, sm._pq2_mma_small(x, w, 2, 4, 1)) < 1e-4


def test_rejects_bad_shapes():
    w = _weight(5120, 6144)
    with pytest.raises(ValueError):
        gl.pq2_gluon_small(torch.randn(9, 6144, device="cuda", dtype=torch.bfloat16), w, 2, 1, 8, 1)   # M > 8
    with pytest.raises(ValueError):
        gl.pq2_gluon_small(torch.randn(2, 6144, device="cuda", dtype=torch.bfloat16), w, 2, 1, 8, 5)   # 48 % 5


def test_dispatch(monkeypatch):
    n, k = 5120, 6144
    w = _weight(n, k)
    calls = []
    real = gl.pq2_gluon_small
    monkeypatch.setattr(gl, "pq2_gluon_small", lambda *a: calls.append(a[2:]) or real(*a))
    table = {2: {(n, k): (2, 1, 2, 4)}, 3: {(n, k): (2, 1, 2, 4)}}
    monkeypatch.setattr(sm, "GLUON_BY_M", table)
    monkeypatch.setattr(sm, "GLUON_EXACT", frozenset({(2, n, k)}))
    monkeypatch.setattr(sm, "_GLUON", True)
    x2 = torch.randn(2, k, device="cuda", dtype=torch.bfloat16)
    x3 = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    sm.pq2_small_mmq(x2, w)
    sm.pq2_small_mmq(x3, w)
    assert calls == [(2, 1, 2, 4), (2, 1, 2, 4)]
    calls.clear()
    monkeypatch.setattr(pq2_config, "KERNELS", "exact")          # exact set keeps only bit-identical entries
    sm.pq2_small_mmq(x2, w)
    sm.pq2_small_mmq(x3, w)
    assert calls == [(2, 1, 2, 4)]
    calls.clear()
    monkeypatch.setattr(pq2_config, "KERNELS", "tuned")
    sm.pq2_small_mmq(torch.randn(4, k, device="cuda", dtype=torch.bfloat16), w)   # not listed
    monkeypatch.setattr(sm, "_GLUON", False)
    sm.pq2_small_mmq(x2, w)
    monkeypatch.setattr(sm, "_GLUON", True)
    monkeypatch.setattr(gl, "AVAILABLE", False)                  # Triton without Gluon: previous path
    sm.pq2_small_mmq(x2, w)
    assert calls == []


def test_table_entries_match_their_replacement():
    """Every SM86 table entry: bit-identical where listed exact, otherwise within FP32 reordering."""
    if not sm._cuda_available():
        pytest.skip("pq2_mma_small not built")
    for m, shapes in sm.GLUON_BY_M.items():
        for (n, k), cfg in shapes.items():
            w = _weight(n, k)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            y = gl.pq2_gluon_small(x, w, *cfg)
            ref = sm._pq2_mma_small(x, w, 2, 4, 1)
            if (m, n, k) in sm.GLUON_EXACT and cfg[3] == 1:
                assert torch.equal(y, ref), (m, n, k)
            assert _rel(y, ref) < 1e-4, (m, n, k)
