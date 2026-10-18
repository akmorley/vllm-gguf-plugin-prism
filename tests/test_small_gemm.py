# SPDX-License-Identifier: Apache-2.0
"""Split-K Triton GEMM for small unquantized projections versus torch / cuBLAS, and its routing."""
import pytest
import torch

from vllm_gguf_plugin.triton import small_gemm as sg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 13, 16])
@pytest.mark.parametrize("n", [96, 100, 1024])
def test_matches_fp32_reference_like_cublas(m, n, dtype):
    torch.manual_seed(m * n)
    k = 5120
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    w = torch.randn(n, k, device="cuda", dtype=dtype) * 0.02
    ref = x.float() @ w.float().t()
    y = sg.small_gemm(x, w)
    assert y.dtype == dtype and y.shape == (m, n)
    err = float((y.float() - ref).norm() / ref.norm())
    cublas = float(((x @ w.t()).float() - ref).norm() / ref.norm())
    assert err <= max(2 * cublas, 1e-3), (err, cublas)
    assert torch.equal(y, sg.small_gemm(x, w))                  # deterministic (fixed-order split reduction)


def test_eligibility(monkeypatch):
    monkeypatch.setattr(sg, "ENABLED", True)
    w = torch.randn(96, 5120, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16)
    assert sg.eligible(x, w)
    assert not sg.eligible(torch.randn(17, 5120, device="cuda", dtype=torch.bfloat16), w)        # M > 16
    assert not sg.eligible(x, torch.randn(2048, 5120, device="cuda", dtype=torch.bfloat16))      # N > 1024
    assert not sg.eligible(x.float(), w.float())                                                 # FP32
    assert not sg.eligible(torch.randn(4, 5000, device="cuda", dtype=torch.bfloat16),
                           torch.randn(96, 5000, device="cuda", dtype=torch.bfloat16))           # K % 256
    monkeypatch.setattr(sg, "ENABLED", False)
    assert not sg.eligible(x, w)


def test_routing_through_unquantized_linear_method(monkeypatch):
    """GGUF layers with unquantized tensors get GGUFUnquantizedLinearMethod (not the GGUF op); with the switch on it
    routes through the custom op, which takes the Triton kernel at M <= 16 and cuBLAS above."""
    from vllm_gguf_plugin.quantization import linear
    w = torch.randn(96, 5120, device="cuda", dtype=torch.bfloat16) * 0.02
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    method = linear.GGUFUnquantizedLinearMethod.__new__(linear.GGUFUnquantizedLinearMethod)
    method._gemm_impl = lambda layer, x, weight, bias: torch.nn.functional.linear(x, weight, bias)
    calls = []
    real = sg.small_gemm
    monkeypatch.setattr(sg, "small_gemm", lambda *a: calls.append(a[0].shape[0]) or real(*a))
    monkeypatch.setattr(sg, "ENABLED", True)
    for m in (4, 2048):
        x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
        y = method.apply(layer, x)
        ref = x @ w.t()
        assert y.shape == ref.shape and float((y.float() - ref.float()).norm() / ref.float().norm()) < 1e-2
    assert calls == [4]                                          # M = 2048 stays on cuBLAS
    x3 = torch.randn(2, 3, 5120, device="cuda", dtype=torch.bfloat16)  # leading dims kept
    assert method.apply(layer, x3).shape == (2, 3, 96)
    monkeypatch.setattr(sg, "ENABLED", False)
    calls.clear()
    method.apply(layer, torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16))
    assert calls == []


def test_config_gives_unquantized_layers_the_gguf_method():
    import inspect
    from vllm_gguf_plugin.quantization import config
    src = inspect.getsource(config)
    assert "return GGUFUnquantizedLinearMethod()" in src
