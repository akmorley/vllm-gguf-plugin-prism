# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m,k", [(1, 384), (4, 640), (16, 17408)])
@pytest.mark.parametrize("strided", [False, True])
def test_integer_gemv_quantized_reference_and_graph(dtype, m, k, strided):
    from vllm_gguf_plugin.triton.pq2_int_gemv import pq2_int_gemv

    torch.manual_seed(41)
    n = 9
    storage = torch.randint(
        0, 256, (n * 2, k // 128 * 34), dtype=torch.uint8, device="cuda"
    )
    w = storage[::2] if strided else storage[:n]
    blocks = w.view(n, k // 128, 34)
    scales = torch.randn(n, k // 128, dtype=torch.float16, device="cuda")
    scales[:, 1] = 0
    blocks[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
    codes = (
        ((blocks[:, :, 2:, None].int() >> (2 * torch.arange(4, device="cuda"))) & 3) - 1
    ).reshape(n, k // 128, 128)
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    x[0, :128] = 0
    a = x.float().reshape(m, k // 128, 128)
    xs = a.abs().amax(-1, keepdim=True) / 127
    q = torch.where(xs > 0, (a / xs).round().clamp(-127, 127), 0)
    expected = torch.einsum(
        "mbk,nbk->mn", q * xs, codes.float() * scales[:, :, None]
    ).to(dtype)
    for _ in range(3):
        actual = pq2_int_gemv(x, w)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = pq2_int_gemv(x, w)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, expected * 0.5, atol=0.02, rtol=0.002)


def test_integer_gemv_masks_tokens_and_reuses_runtime_m():
    from vllm_gguf_plugin.triton.pq2_int_gemv import _int_gemv, pq2_int_gemv

    n, k = 9, 640
    w = torch.zeros(n, k // 128, 34, dtype=torch.uint8, device="cuda")
    scales = torch.ones(n, k // 128, dtype=torch.float16, device="cuda")
    w[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
    w[:, :, 2:] = 0xE4
    sizes = []
    for m in (5, 7, 8):
        x = torch.ones(m, k, dtype=torch.bfloat16, device="cuda")
        # Hold the kernel algorithm fixed: the M8 default intentionally uses
        # chaining, which has its own specialization and separate tests.
        result = pq2_int_gemv(x, w.reshape(n, -1), chained=False)
        torch.testing.assert_close(
            result, torch.full_like(result, k / 2), rtol=0, atol=0
        )
        sizes.append(sum(len(cache[0]) for cache in _int_gemv.device_caches.values()))
    assert len(set(sizes)) == 1


@pytest.mark.parametrize("m", [0, 17])
def test_integer_gemv_rejects_unsupported_batches(m):
    from vllm_gguf_plugin.triton.pq2_int_gemv import pq2_int_gemv

    x = torch.empty(m, 128, device="cuda")
    w = torch.empty(9, 34, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="1 to 16"):
        pq2_int_gemv(x, w)


def test_integer_serving_dispatch_preserves_fallbacks(monkeypatch):
    from vllm_gguf_plugin.triton import prism

    monkeypatch.setattr(prism, "_EXPERIMENTAL_INT_GEMV", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    x = torch.empty(1, 5120, dtype=torch.bfloat16, device="cuda")
    for m in (1, 2, 4, 5, 8):
        for n, k in prism._BATCHED_GEMV_SHAPES:
            assert prism._use_int_gemv(x.expand(m, -1), n, k)
    for m in (0, 3, 6, 7, 9, 16, 32):
        assert not prism._use_int_gemv(x.expand(m, -1), 34816, 5120)
    assert not prism._use_int_gemv(x, 37, 5120)
    assert not prism._use_int_gemv(x.float(), 34816, 5120)
    assert not prism._use_int_gemv(x.cpu(), 34816, 5120)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert not prism._use_int_gemv(x, 34816, 5120)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    monkeypatch.setattr(prism, "_EXPERIMENTAL_INT_GEMV", False)
    assert not prism._use_int_gemv(x, 34816, 5120)


@pytest.mark.parametrize("m", [1, 2, 4, 5, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_integer_serving_dispatch_graph_replay(monkeypatch, m, dtype):
    from vllm_gguf_plugin.triton import prism
    from vllm_gguf_plugin.triton.pq2_int_gemv import pq2_int_gemv

    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("Serving experiment is limited to SM86")
    monkeypatch.setattr(prism, "_EXPERIMENTAL_INT_GEMV", True)
    n, k = 5120, 17408
    blocks = torch.zeros(n, k // 128, 34, dtype=torch.uint8, device="cuda")
    scales = torch.full((n, k // 128), 0.01, dtype=torch.float16, device="cuda")
    blocks[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
    blocks[:, :, 2:] = 0xE4
    weight = blocks.reshape(n, -1)
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    calls = []

    def observed(x, weight):
        calls.append(x.shape[0])
        return pq2_int_gemv(x, weight)

    from vllm_gguf_plugin.triton import pq2_int_gemv as module

    monkeypatch.setattr(module, "pq2_int_gemv", observed)
    for _ in range(3):
        prism.pq2_matmul(x, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prism.pq2_matmul(x, weight)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, pq2_int_gemv(x, weight), rtol=0, atol=0)
    assert calls == [m] * 4  # Integer precedence even when both flags are enabled.
