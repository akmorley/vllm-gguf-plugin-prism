# SPDX-License-Identifier: Apache-2.0
import weakref
from types import SimpleNamespace

import pytest
import torch
from gguf import GGMLQuantizationType

from vllm_gguf_plugin.triton.pq2_gemv import pq2_batched_gemv
from vllm_gguf_plugin.triton.pq2_int_gemv import pq2_int_gemv
from vllm_gguf_plugin.triton.pq2_layout import prepare_pq2_layout
from vllm_gguf_plugin.triton.pq2_mmq import pq2_mmq
from vllm_gguf_plugin.triton.prism import pq2_matmul

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def weight(n, k):
    torch.manual_seed(127)
    w = torch.randint(0, 256, (n, k // 128, 34), dtype=torch.uint8, device="cuda")
    scales = torch.randn(n, k // 128, device="cuda", dtype=torch.float16) * 0.02
    scales[:, 0] = 0
    w[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
    return w.reshape(n, -1)


@pytest.mark.parametrize("k", [128, 384, 5120, 17408])
@pytest.mark.parametrize("strides", [False, True])
def test_prepared_bytes_and_single_allocation(k, strides):
    w = weight(9, k)
    if strides:
        storage = torch.empty(
            (18, w.shape[1] * 2 + 1), dtype=torch.uint8, device="cuda"
        )
        view = storage[::2, 1::2]
        view.copy_(w)
        w = view
    prepared = prepare_pq2_layout(w)
    n, width = w.shape
    codes = prepared.flatten()[: n * k // 4].reshape(n, k // 128, 32)
    scales = prepared.flatten()[n * k // 4 :].reshape(n, k // 128, 2)
    torch.testing.assert_close(codes, w.reshape(n, -1, 34)[:, :, 2:], atol=0, rtol=0)
    torch.testing.assert_close(scales, w.reshape(n, -1, 34)[:, :, :2], atol=0, rtol=0)
    assert prepared.shape == (n, width)
    assert prepared.untyped_storage().nbytes() == w.numel()
    assert codes.untyped_storage().data_ptr() == scales.untyped_storage().data_ptr()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "m,k", [(1, 128), (3, 384), (4, 640), (5, 1024), (8, 5120), (16, 17408), (65, 384)]
)
@pytest.mark.parametrize(
    "method", [pq2_matmul, pq2_int_gemv, pq2_batched_gemv, pq2_mmq]
)
def test_prepared_paths_bitwise_and_graph(dtype, m, k, method):
    if m > 16 and method in (pq2_int_gemv, pq2_batched_gemv):
        pytest.skip("Decode APIs accept at most 16 tokens")
    w = weight(9, k)
    prepared = prepare_pq2_layout(w)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    expected = method(x, w)
    for _ in range(3):
        actual = method(x, prepared, prepared=True)
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method(x, prepared, prepared=True)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, expected * 0.5, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_prepared_embedding_custom_op_and_compile(dtype):
    from vllm_gguf_plugin.quantization.vocal_embeds import apply_gguf_embedding

    w = weight(9, 640)
    indices = torch.tensor([[8, 0, 2], [2, 8, 1]], device="cuda")
    prepared = prepare_pq2_layout(w)
    pq2 = int(GGMLQuantizationType.PQ2_0)
    expected = apply_gguf_embedding(indices, w, pq2, 640, dtype)

    def run(indices, prepared):
        return apply_gguf_embedding(indices, prepared, pq2, 640, dtype, True)

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(indices, prepared).view(torch.uint8),
        expected.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    for _ in range(3):
        actual = run(indices, prepared)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(indices, prepared)
    indices.fill_(1)
    graph.replay()
    torch.testing.assert_close(
        captured, apply_gguf_embedding(indices, w, pq2, 640, dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
    )


def test_loader_releases_source_and_prepares_tied_parameter_once(monkeypatch):
    from vllm_gguf_plugin.quantization import linear
    from vllm_gguf_plugin.quantization.vocal_embeds import GGUFEmbeddingMethod

    monkeypatch.setattr(linear, "_EXPERIMENTAL_PREPARED_PQ2", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    source = weight(9, 1024)
    old = source.clone()
    reference = weakref.ref(source)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(source, requires_grad=False)
    layer.weight.data_container = [source]
    layer.weight.shard_id = []
    layer.weight.tensor_shape = (9, 1024)
    layer.weight_type = SimpleNamespace(
        weight_type=int(GGMLQuantizationType.PQ2_0), shard_weight_type={}
    )
    method = GGUFEmbeddingMethod(None)
    method.params_dtype = torch.bfloat16
    head = torch.nn.Module()
    method.tie_weights(head, layer)
    del source
    method._prepare_pq2_weight(layer)
    assert reference() is None
    assert layer.weight.data_container == []
    pointer = layer.weight.data_ptr()
    method._prepare_pq2_weight(head)
    assert head.weight is layer.weight
    assert head.weight.data_ptr() == pointer
    x = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        method.apply(head, x), pq2_matmul(x, old), atol=0, rtol=0
    )
    indices = torch.tensor([0, 8, 1], device="cuda")
    from vllm_gguf_plugin.quantization.vocal_embeds import apply_gguf_embedding

    torch.testing.assert_close(
        method.embedding(layer, indices),
        apply_gguf_embedding(
            indices, old, int(GGMLQuantizationType.PQ2_0), 1024, torch.bfloat16
        ),
        atol=0,
        rtol=0,
    )


def test_preparation_rejects_capture_and_partial_planes(monkeypatch):
    w = weight(9, 384)
    prepared = prepare_pq2_layout(w)
    with pytest.raises(ValueError, match="complete contiguous"):
        pq2_matmul(torch.ones(1, 384, device="cuda"), prepared[:3], prepared=True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        prepare_pq2_layout(w)


@pytest.mark.parametrize("reason", ["disabled", "mixed", "unquantized", "architecture"])
def test_loader_preserves_unsupported_weights(monkeypatch, reason):
    from vllm_gguf_plugin.quantization import linear

    monkeypatch.setattr(linear, "_EXPERIMENTAL_PREPARED_PQ2", reason != "disabled")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: (9, 0) if reason == "architecture" else (8, 6),
    )
    w = torch.nn.Parameter(weight(9, 384), requires_grad=False)
    w.shard_id = ["a", "b"] if reason == "mixed" else []
    pq2 = int(GGMLQuantizationType.PQ2_0)
    layer = SimpleNamespace(
        weight=w,
        weight_type=SimpleNamespace(
            weight_type=0 if reason == "unquantized" else pq2,
            shard_weight_type={"a": pq2, "b": 0} if reason == "mixed" else {},
        ),
    )
    pointer = w.data_ptr()
    linear.GGUFLinearMethod(None)._prepare_pq2_weight(layer)
    assert w.data_ptr() == pointer
    assert not getattr(w, "gguf_pq2_prepared", False)


def test_empty_prepared_rows_and_embedding_indices():
    from vllm_gguf_plugin.triton.pq2_layout import pq2_prepared_embedding

    w = prepare_pq2_layout(torch.empty(0, 102, dtype=torch.uint8, device="cuda"))
    assert pq2_matmul(torch.ones(1, 384, device="cuda"), w, prepared=True).shape == (
        1,
        0,
    )
    assert pq2_int_gemv(torch.ones(1, 384, device="cuda"), w, prepared=True).shape == (
        1,
        0,
    )
    indices = torch.empty(0, 2, dtype=torch.int64, device="cuda")
    assert pq2_prepared_embedding(indices, w, 384, torch.bfloat16).shape == (0, 2, 384)


def test_prepared_layout_changes_compilation_factors(monkeypatch):
    import vllm.envs as envs
    from vllm.config.utils import hash_factors

    from vllm_gguf_plugin import plugin
    from vllm_gguf_plugin.quantization import linear

    monkeypatch.setattr(envs, "environment_variables", {})
    plugin._register_pq2_compile_factors()
    hashes = []
    for enabled in (False, True):
        monkeypatch.setattr(linear, "_EXPERIMENTAL_PREPARED_PQ2", enabled)
        values = envs.compile_factors()
        assert values["GGUF_PQ2_PREPARED"] is enabled
        assert values["GGUF_PQ2_LAYOUT_VERSION"] == "planes-v1"
        hashes.append(hash_factors(values))
    assert hashes[0] != hashes[1]
