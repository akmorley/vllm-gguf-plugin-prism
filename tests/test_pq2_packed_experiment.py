# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import triton
import triton.language as tl

from vllm_gguf_plugin.triton.pq2_packed_experiment import (
    _expand_byte,
    packed_gemv,
    prepare_pq2,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@triton.jit
def _expand_test(X, Y, FAST: tl.constexpr):
    i = tl.arange(0, 256)
    tl.store(Y + i, _expand_byte(tl.load(X + i), FAST))


@pytest.mark.parametrize("fast", [False, True])
def test_all_packed_bytes(fast):
    p = torch.arange(256, dtype=torch.int32, device="cuda")
    expected = sum(
        ((((p >> (2 * i)) & 3) - 1) & 255).long() << (8 * i) for i in range(4)
    ).int()
    actual = torch.empty_like(p)
    _expand_test[(1,)](p, actual, fast)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "load,separate", [(1, False), (2, False), (1, True), (2, True), (4, True)]
)
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m,k", [(1, 384), (4, 640), (8, 17408)])
def test_packed_formats_reference_and_graph(load, separate, fast, dtype, m, k):
    torch.manual_seed(83)
    n = 9
    storage = torch.randint(
        0, 256, (n * 2, k // 128 * 34), dtype=torch.uint8, device="cuda"
    )
    weight = storage[::2]
    blocks = weight.view(n, k // 128, 34)
    scales = torch.randn(n, k // 128, dtype=torch.float16, device="cuda") * 0.02
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
    prepared = prepare_pq2(weight, load_bytes=load, separate=separate)
    assert prepared.storage_bytes == weight.numel()
    config = dict(tile=(16, 8, 4), fast_pack=fast, stages=2, unroll=2)
    for _ in range(3):
        actual = packed_gemv(x, prepared, **config)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = packed_gemv(x, prepared, **config)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, expected * 0.5, atol=0.02, rtol=0.002)


def test_raw_half_load_alignment_and_empty_rows():
    storage = torch.zeros(9 * 35 + 1, dtype=torch.uint8, device="cuda")
    odd_stride = storage[: 9 * 35].view(9, 35)[:, :34]
    odd_base = storage[1:35].reshape(1, 34)
    for weight in (odd_stride, odd_base):
        with pytest.raises(ValueError, match="even base and row stride"):
            prepare_pq2(weight, load_bytes=2)
        assert prepare_pq2(weight, load_bytes=1).k == 128
        assert prepare_pq2(weight, load_bytes=4, separate=True).k == 128
    empty = prepare_pq2(
        torch.empty(0, 34, dtype=torch.uint8, device="cuda"),
        load_bytes=4,
        separate=True,
    )
    assert packed_gemv(torch.ones(1, 128, device="cuda"), empty).shape == (1, 0)


@pytest.mark.parametrize("load,separate", [(2, False), (2, True), (4, True)])
@pytest.mark.parametrize("m", [1, 4, 8])
def test_wide_load_keeps_dot_lanes(load, separate, m):
    from vllm_gguf_plugin.triton.pq2_int_gemv import pq2_int_gemv

    torch.manual_seed(109)
    n, k = 9, 640
    weight = torch.randint(0, 256, (n, k // 128, 34), dtype=torch.uint8, device="cuda")
    scales = torch.randn(n, k // 128, dtype=torch.float16, device="cuda") * 0.02
    scales[:, 1] = 0
    weight[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
    weight = weight.reshape(n, -1)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    expected = pq2_int_gemv(x, weight)
    prepared = prepare_pq2(weight, load_bytes=load, separate=separate)
    config = dict(fast_pack=True, keep_lanes=True)
    for _ in range(3):
        actual = packed_gemv(x, prepared, **config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = packed_gemv(x, prepared, **config)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, expected * 0.5, rtol=0, atol=0)
