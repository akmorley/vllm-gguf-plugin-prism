from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_gguf_plugin.hadamard import (
    HadamardPermutation,
    HadamardRuntimeConfig,
    PrismHadamardConfig,
    apply_forward_hadamard,
)
from vllm_gguf_plugin.quantization.config import GGUFConfig
from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod
from vllm_gguf_plugin.quantization.vocal_embeds import GGUFEmbeddingMethod


def make_hadamard_config():
    return PrismHadamardConfig(
        version=1,
        block_size=4,
        transform="normalized-sylvester-walsh-hadamard",
        axis="input-last-dimension",
        sign_mode="identity",
    )


def test_runtime_registration_and_resolution():
    config = GGUFConfig()
    hcfg = make_hadamard_config()
    runtime = HadamardRuntimeConfig(hcfg)
    config.register_hadamard(hcfg, {"lm_head": runtime}, {"model.embed_tokens"})
    assert config._resolve_packed_hadamard("language_model.lm_head") is runtime
    assert config._resolve_packed_hadamard("missing") is None
    assert "language_model.model.embed_tokens" in config.hadamard_inverse_modules
    config.register_hadamard(None, {}, set())
    assert not config.hadamard_forward_modules
    assert not config.hadamard_inverse_modules


@pytest.mark.parametrize(
    "suffix,sources",
    [
        (
            "self_attn.qkv_proj",
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ),
        (
            "linear_attn.in_proj_qkvz",
            ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
        ),
        ("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
    ],
)
def test_packed_runtime_agreement(suffix, sources):
    config = GGUFConfig()
    hcfg = make_hadamard_config()
    base = "language_model.model.layers.0."
    runtimes = {base + name: HadamardRuntimeConfig(hcfg) for name in sources}
    config.register_hadamard(hcfg, runtimes, set())
    assert config._resolve_packed_hadamard(base + suffix) is runtimes[base + sources[0]]
    runtimes[base + sources[-1]].skip_input_layout = True
    with pytest.raises(ValueError, match="runtime configuration mismatch"):
        config._resolve_packed_hadamard(base + suffix)
    del config.hadamard_forward_modules[base + sources[-1]]
    assert config._resolve_packed_hadamard(base + suffix) is None


def test_embedding_installs_forward_runtime_and_inverse_metadata():
    hcfg = make_hadamard_config()
    runtime = HadamardRuntimeConfig(hcfg)
    method = GGUFEmbeddingMethod(
        None, inverse_hadamard_config=hcfg, hadamard_runtime_config=runtime
    )
    assert method.hadamard_runtime_config is runtime
    assert method.inverse_hadamard_config is hcfg


@pytest.mark.parametrize("skip_input_layout", [False, True])
def test_linear_applies_transform_and_respects_layout(skip_input_layout):
    hcfg = make_hadamard_config()
    runtime = HadamardRuntimeConfig(
        hcfg, HadamardPermutation(2, 2, 2), skip_input_layout
    )
    layout = Mock()
    layout.input_to_gguf.side_effect = lambda x: x + 10
    layout.output_to_vllm.side_effect = lambda x: x
    method = GGUFLinearMethod(None, layout=layout, hadamard_runtime_config=runtime)
    layer = SimpleNamespace(
        weight=SimpleNamespace(shard_id=[]), weight_type=SimpleNamespace(weight_type=0)
    )
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    captured = []

    def matmul(x, *args, **kwargs):
        captured.append(x)
        return x

    with patch("vllm_gguf_plugin.quantization.fused_mul_mat_gguf", side_effect=matmul):
        method.apply(layer, x)
    expected = apply_forward_hadamard(x, hcfg)
    if not skip_input_layout:
        expected = expected + 10
    torch.testing.assert_close(captured[0], expected)
    assert layout.input_to_gguf.call_count == int(not skip_input_layout)


@pytest.mark.parametrize("kind", ["linear", "embedding"])
def test_quant_method_receives_runtime(kind):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    config = GGUFConfig()
    hcfg = make_hadamard_config()
    runtime = HadamardRuntimeConfig(hcfg, skip_input_layout=True)
    prefix = "language_model.lm_head"
    config.register_hadamard(hcfg, {prefix: runtime}, {prefix})
    layer = Mock(spec=LinearBase if kind == "linear" else VocabParallelEmbedding)
    method = config.get_quant_method(layer, prefix)
    assert method.hadamard_runtime_config is runtime
    if kind == "embedding":
        assert method.inverse_hadamard_config is hcfg
