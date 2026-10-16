# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from ..hadamard import (
    HadamardRuntimeConfig,
    PrismHadamardConfig,
    apply_inverse_hadamard,
)
from .linear import GGUFLinearMethod
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_embedding_weight_loader,
    _gguf_embedding_weight_type_loader,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
)
from .utils import DEQUANT_TYPES, UNQUANTIZED_TYPES


def recursive_replace_vocab_modules(
    model: torch.nn.Module,
    quant_config: QuantizationConfig,
    prefix: str = "",
) -> None:
    """Recursively rebuild vocab modules missing their quantization method."""
    replacements: dict[int, torch.nn.Module] = {}

    def replace(module: torch.nn.Module, prefix: str) -> None:
        # named_children() de-duplicates tied modules.
        for child_name, child_module in tuple(module._modules.items()):
            if child_module is None:
                continue
            qual_name = maybe_prefix(prefix, child_name)
            replacement = replacements.get(id(child_module))
            if replacement is not None:
                setattr(module, child_name, replacement)
                continue

            if type(child_module) not in (VocabParallelEmbedding, ParallelLMHead):
                replace(child_module, qual_name)
                continue

            expected_method = quant_config.get_quant_method(child_module, qual_name)
            if type(child_module.quant_method) is type(expected_method):
                continue

            kwargs = (
                {"bias": child_module.bias is not None}
                if type(child_module) is ParallelLMHead
                else {}
            )
            replacement = type(child_module)(
                child_module.num_embeddings,
                child_module.embedding_dim,
                params_dtype=child_module.params_dtype,
                org_num_embeddings=child_module.org_vocab_size,
                padding_size=child_module.padding_size,
                quant_config=quant_config,
                prefix=qual_name,
                **kwargs,
            )
            replacements[id(child_module)] = replacement
            setattr(module, child_name, replacement)

    replace(model, prefix)

    for module in model.modules():
        pending = module.__dict__.pop("_gguf_pending_tie", None)
        if pending is None:
            continue
        embed_tokens = replacements.get(id(pending), pending)
        if not hasattr(embed_tokens, "weight_type"):
            raise RuntimeError(
                "GGUF lm_head is tied to an embedding that is not GGUF-quantized"
            )
        module.quant_method.tie_weights(module, embed_tokens)


def _apply_gguf_embedding(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
    pq2_prepared: bool = False,
) -> torch.Tensor:
    if pq2_prepared:
        if weight_type != WeightType.PQ2_0:
            raise ValueError("Prepared embedding weights must be PQ2")
        from ..triton.pq2_layout import pq2_prepared_embedding

        return pq2_prepared_embedding(x, weight, hidden_size, dtype)
    if weight_type in UNQUANTIZED_TYPES:
        return torch.embedding(weight, x)
    if weight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[weight_type]
        x_flat = x.flatten()
        assert hidden_size == weight.shape[1] // type_size * block_size
        quant = torch.index_select(weight, dim=0, index=x_flat)
        dequant = ops.ggml_dequantize(
            quant, weight_type, hidden_size, x_flat.shape[0], dtype
        )
        return dequant.view(*x.shape, hidden_size)
    weight_type = WeightType(weight_type)
    raise NotImplementedError(f"Unsupported GGUF quantization type: {weight_type}")


def _apply_gguf_embedding_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
    pq2_prepared: bool = False,
) -> torch.Tensor:
    del weight, weight_type
    return torch.empty(*x.shape, hidden_size, dtype=dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_apply_gguf_embedding",
        op_func=_apply_gguf_embedding,
        fake_impl=_apply_gguf_embedding_fake,
    )
    apply_gguf_embedding = torch.ops.vllm._apply_gguf_embedding
except AttributeError as error:
    raise error


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF."""

    def __init__(
        self,
        quant_config,
        inverse_hadamard_config: PrismHadamardConfig | None = None,
        hadamard_runtime_config: HadamardRuntimeConfig | None = None,
    ):

        # ParallelLMHead uses the inherited GGUFLinearMethod.apply().
        #
        # Therefore the forward-Hadamard state must be installed
        # in the base linear method.
        super().__init__(
            quant_config,
            layout=None,
            hadamard_runtime_config=hadamard_runtime_config,
        )


        self.inverse_hadamard_config = (
            inverse_hadamard_config
        )




    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        extra_weight_attrs.pop("weight_loader", None)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            weight,
            {
                "weight_loader": partial(_gguf_embedding_weight_loader, layer),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

        weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            weight_type,
            {
                "weight_loader": _gguf_embedding_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(weight_type, extra_weight_attrs)
        layer.register_parameter("weight_type", weight_type)

    def _materialize_weight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(
            layer,
            "weight",
            fallback_weight_loader=partial(_gguf_embedding_weight_loader, layer),
        )

    def _materialize_weight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(
            layer,
            "weight_type",
            fallback_weight_loader=_gguf_embedding_weight_type_loader,
        )

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        from . import apply_gguf_embedding as apply_gguf_embedding_op

        weight = layer.weight
        weight_type = layer.weight_type.weight_type
        hidden_size = weight.tensor_shape[1]
        out = apply_gguf_embedding_op(
            x, weight, weight_type, hidden_size, dtype=self.params_dtype,
            pq2_prepared=getattr(weight, "gguf_pq2_prepared", False),
        )


        if (
            self.inverse_hadamard_config
            is not None
        ):
            out = apply_inverse_hadamard(
                out,
                self.inverse_hadamard_config,
            )

        return out

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: VocabParallelEmbedding):
        if not hasattr(embed_tokens, "weight_type"):
            # Some models (vLLM's Qwen3.5) build embed_tokens without a quant config
            # and tie to it in __init__. recursive_replace_vocab_modules swaps in the
            # GGUF embedding afterwards and completes the tie then.
            # Plain __dict__ entry: nn.Module.__setattr__ would register it as a child.
            layer.__dict__["_gguf_pending_tie"] = embed_tokens
            return layer
        layer.weight = embed_tokens.weight
        layer.weight_type = embed_tokens.weight_type
        return layer
