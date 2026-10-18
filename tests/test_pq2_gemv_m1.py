# SPDX-License-Identifier: Apache-2.0
"""Single-token CUDA DP4A-on-planes kernel versus the Triton `_single` path."""
import pytest
import torch

from vllm_gguf_plugin.triton import pq2_int_gemv as ig
from vllm_gguf_plugin.triton.pq2_layout import prepare_pq2_layout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6),
    reason="SM86 CUDA required",
)


@pytest.fixture(autouse=True)
def _new_default_path_off(monkeypatch):
    """These tests exercise the CUDA M = 1 kernel and Triton `_single` (fallbacks of the Triton plane kernel, the default)."""
    monkeypatch.setattr(ig, "_M1_TRITON", False)


_WEIGHTS = {}


def _weight(n, k):
    if (n, k) not in _WEIGHTS:
        torch.manual_seed(41)
        raw = torch.randint(0, 3, (n, k // 128, 34), device="cuda", dtype=torch.uint8) * 0x55  # codes 0..2 only
        raw[:, :, 2:] = torch.randint(0, 256, (n, k // 128, 32), device="cuda", dtype=torch.uint8) & 0x55 | (
            torch.randint(0, 2, (n, k // 128, 32), device="cuda", dtype=torch.uint8) << 1) & 0xAA
        raw[:, :, 2:] = torch.where((raw[:, :, 2:] & 0xC0) == 0xC0, raw[:, :, 2:] & 0x7F, raw[:, :, 2:])
        scales = (torch.rand(n, k // 128, device="cuda") * 0.03 + 0.001).half()
        raw[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
        _WEIGHTS[(n, k)] = prepare_pq2_layout(raw.reshape(n, -1))
    return _WEIGHTS[(n, k)]


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", sorted(ig._M1_CFG))
def test_cuda_kernel_matches_triton_single(monkeypatch, dtype, shape):
    if not ig._m1_available():
        pytest.skip("pq2_gemv_m1 not built")
    n, k = shape
    w = _weight(n, k)
    torch.manual_seed(n)
    x = torch.randn(1, k, device="cuda", dtype=dtype)
    monkeypatch.setattr(ig, "_M1_CUDA", False)
    expected = ig.pq2_int_gemv(x, w, prepared=True)
    monkeypatch.setattr(ig, "_M1_CUDA", True)
    calls = []
    real = ig._gemv_m1
    monkeypatch.setattr(ig, "_gemv_m1", lambda *a: calls.append(1) or real(*a))
    first = ig.pq2_int_gemv(x, w, prepared=True)
    second = ig.pq2_int_gemv(x, w, prepared=True)
    assert calls == [1, 1]
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert _rel(first, expected) < 3e-4


def test_dispatch_scope(monkeypatch):
    if not ig._m1_available():
        pytest.skip("pq2_gemv_m1 not built")
    n, k = 5120, 6144
    w = _weight(n, k)
    calls = []
    real = ig._gemv_m1
    monkeypatch.setattr(ig, "_gemv_m1", lambda *a: calls.append(1) or real(*a))
    monkeypatch.setattr(ig, "_M1_CUDA", True)
    x1 = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    ig.pq2_int_gemv(x1, w, prepared=True)
    assert calls == [1]
    calls.clear()
    ig.pq2_int_gemv(torch.randn(2, k, device="cuda", dtype=torch.bfloat16), w, prepared=True)   # M = 2
    ig.pq2_int_gemv(x1.float(), w, prepared=True)                                                # FP32 activations
    ig.pq2_int_gemv(x1, w, prepared=True, chained=True)                                          # chained DP4A
    assert calls == []
    monkeypatch.setattr(ig, "_M1_CUDA", False)
    ig.pq2_int_gemv(x1, w, prepared=True)
    assert calls == []


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", sorted(ig._M1_TRITON_CFG))
def test_triton_planes_matches_triton_single(monkeypatch, dtype, shape):
    n, k = shape
    w = _weight(n, k)
    torch.manual_seed(n)
    x = torch.randn(1, k, device="cuda", dtype=dtype)
    monkeypatch.setattr(ig, "_M1_TRITON", False)
    monkeypatch.setattr(ig, "_M1_CUDA", False)
    expected = ig.pq2_int_gemv(x, w, prepared=True)
    monkeypatch.setattr(ig, "_M1_TRITON", True)
    calls = []
    real = ig._gemv_m1_triton
    monkeypatch.setattr(ig, "_gemv_m1_triton", lambda *a: calls.append(1) or real(*a))
    first = ig.pq2_int_gemv(x, w, prepared=True)
    second = ig.pq2_int_gemv(x, w, prepared=True)
    assert calls == [1, 1]
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert _rel(first, expected) < 3e-4


def test_triton_planes_dispatch_scope(monkeypatch):
    assert (248320, 5120) in ig._M1_TRITON_CFG  # every M = 1 shape, lm_head included
    n, k = 5120, 6144
    w = _weight(n, k)
    calls = []
    real = ig._gemv_m1_triton
    monkeypatch.setattr(ig, "_gemv_m1_triton", lambda *a: calls.append(1) or real(*a))
    monkeypatch.setattr(ig, "_M1_TRITON", True)
    x1 = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    ig.pq2_int_gemv(x1, w, prepared=True)
    assert calls == [1]
    calls.clear()
    ig.pq2_int_gemv(torch.randn(2, k, device="cuda", dtype=torch.bfloat16), w, prepared=True)   # M = 2
    ig.pq2_int_gemv(x1.float(), w, prepared=True)                                                # FP32 activations
    ig.pq2_int_gemv(x1, w, prepared=True, chained=True)                                          # chained DP4A
    assert calls == []
    monkeypatch.setattr(ig, "_M1_TRITON", False)
    ig.pq2_int_gemv(x1, w, prepared=True)
    assert calls == []
