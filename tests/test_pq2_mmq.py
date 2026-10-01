# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gguf_plugin.triton.pq2_mmq import pq2_mmq

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m,k", [(1, 128), (5, 384), (33, 640)])
@pytest.mark.parametrize("strided", [False, True])
def test_mmq_quantized_reference_and_graph(dtype, m, k, strided):
    torch.manual_seed(19)
    n = 37
    storage = torch.randint(
        0, 256, (n * 2, k // 128 * 34), dtype=torch.uint8, device="cuda"
    )
    w = storage[::2] if strided else storage[:n]
    blocks = w.view(n, k // 128, 34)
    scales = torch.randn(n, k // 128, dtype=torch.float16, device="cuda")
    scales[:, 0] = 0
    blocks[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
    codes = (
        ((blocks[:, :, 2:, None].int() >> (2 * torch.arange(4, device="cuda"))) & 3) - 1
    ).reshape(n, k // 128, 128)
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    x[0] = 0
    a = x.float().reshape(m, k // 128, 128)
    xs = a.abs().amax(-1, keepdim=True) / 127
    q = torch.where(xs > 0, (a / xs).round().clamp(-127, 127), 0)
    expected = torch.einsum(
        "mbk,nbk->mn", q * xs, codes.float() * scales[:, :, None]
    ).to(dtype)
    for _ in range(3):
        actual = pq2_mmq(x, w)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = pq2_mmq(x, w)
    graph.replay()
    torch.testing.assert_close(captured, expected, atol=0.02, rtol=0.002)
    # A replay must consume new activation values rather than captured Q8 data.
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, expected * 0.5, atol=0.02, rtol=0.002)


def test_mmq_runtime_m_reuses_compilation():
    from vllm_gguf_plugin.triton.pq2_mmq import _mmq

    n, k = 129, 384
    w = torch.zeros(n, k // 128, 34, device="cuda", dtype=torch.uint8)
    scales = torch.ones(n, k // 128, device="cuda", dtype=torch.float16)
    w[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
    w[:, :, 2:] = 0xAA
    sizes = []
    for m in (5, 8, 65, 128, 258):
        # Exactly representable Q8 values also test both grid boundary masks.
        x = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        actual = pq2_mmq(x, w.reshape(n, -1))
        torch.testing.assert_close(actual, torch.full_like(actual, k), rtol=0, atol=0)
        sizes.append(sum(len(cache[0]) for cache in _mmq.device_caches.values()))
    assert len(set(sizes)) == 1, "M must not cause new MMQ compiled variants"
