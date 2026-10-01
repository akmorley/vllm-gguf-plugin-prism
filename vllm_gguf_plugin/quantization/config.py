# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix

from ..hadamard import HadamardRuntimeConfig, PrismHadamardConfig
from .utils import is_layer_skipped_gguf

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

    from .layout import GGUFLinearLayout


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, unquantized_modules: list[str] | None = None) -> None:
        super().__init__()
        self.unquantized_modules = unquantized_modules or []
        self.linear_layouts: dict[str, GGUFLinearLayout] = {}
        self.hadamard_config: PrismHadamardConfig | None = None
        self.hadamard_forward_modules: dict[
            str,
            HadamardRuntimeConfig,
        ] = {}
        self.hadamard_inverse_modules: set[str] = set()

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def _to_runtime_module_prefix(
        self,
        name: str,
    ) -> str:
        # lm_head lives directly under language_model
        if name == "lm_head":
            return "language_model.lm_head"

        if name == "model.language_model.lm_head":
            return "language_model.lm_head"

        # embedding/body live under language_model.model
        if name.startswith("model.language_model."):
            rest = name.removeprefix("model.language_model.")

            return "language_model.model." + rest

        # If the map produced bare model.* names
        if name.startswith("model."):
            rest = name.removeprefix("model.")

            return "language_model.model." + rest

        return name

    def _resolve_packed_hadamard(
        self,
        prefix: str,
    ) -> HadamardRuntimeConfig | None:
        """
        Resolve Prism Hadamard metadata for vLLM modules that pack
        multiple GGUF/HF projections into one runtime module.

        Returns:
            the runtime configuration, or None when no transform is registered
        """

        # Direct, non-packed module.
        if prefix in self.hadamard_forward_modules:
            return self.hadamard_forward_modules[prefix]

        sources: tuple[str, ...] | None = None

        # Qwen GDN:
        #
        #   in_proj_qkv + in_proj_z
        #       -> in_proj_qkvz
        if prefix.endswith(".linear_attn.in_proj_qkvz"):
            base = prefix.removesuffix(".linear_attn.in_proj_qkvz")

            sources = (
                base + ".linear_attn.in_proj_qkv",
                base + ".linear_attn.in_proj_z",
            )

        # Standard attention:
        #
        #   q_proj + k_proj + v_proj
        #       -> qkv_proj
        elif prefix.endswith(".self_attn.qkv_proj"):
            base = prefix.removesuffix(".self_attn.qkv_proj")

            sources = (
                base + ".self_attn.q_proj",
                base + ".self_attn.k_proj",
                base + ".self_attn.v_proj",
            )

        # MLP:
        #
        #   gate_proj + up_proj
        #       -> gate_up_proj
        elif prefix.endswith(".mlp.gate_up_proj"):
            base = prefix.removesuffix(".mlp.gate_up_proj")

            sources = (
                base + ".mlp.gate_proj",
                base + ".mlp.up_proj",
            )

        if sources is None:
            return None

        # All constituent projections must be Prism-folded.
        if not all(source in self.hadamard_forward_modules for source in sources):
            return None

        runtime_configs = [self.hadamard_forward_modules[source] for source in sources]

        # All packed components consume the same input tensor,
        # therefore their input transform must agree.
        first = runtime_configs[0]

        if any(
            runtime.config is not first.config
            or runtime.permutation != first.permutation
            or runtime.skip_input_layout != first.skip_input_layout
            for runtime in runtime_configs[1:]
        ):
            raise ValueError(
                "Packed Prism Hadamard runtime configuration mismatch "
                f"for {prefix}: "
                f"{dict(zip(sources, runtime_configs))}"
            )

        return first

    def get_name(self) -> QuantizationMethods:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        del config
        return cls()

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None
    ) -> "QuantizationMethods | None":
        del hf_quant_cfg
        if user_quant == "gguf":
            return "gguf"
        return None

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> "QuantizeMethodBase | None":
        from .fused_moe import GGUFMoEMethod
        from .linear import GGUFLinearMethod
        from .vocal_embeds import GGUFEmbeddingMethod

        # ----------------------------------------------------------
        # Linear layers
        # ----------------------------------------------------------
        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(
                prefix,
                self.unquantized_modules,
                self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()

            return GGUFLinearMethod(
                self,
                layout=self.linear_layouts.get(prefix),
                hadamard_runtime_config=self._resolve_packed_hadamard(prefix),
            )

        # ----------------------------------------------------------
        # Vocabulary embedding / ParallelLMHead
        #
        # In your vLLM version ParallelLMHead goes through this
        # embedding quantization method as well.
        #
        # embed_tokens:
        #     row lookup -> inverse Hadamard
        #
        # lm_head:
        #     forward Hadamard -> inherited linear apply
        # ----------------------------------------------------------
        if isinstance(
            layer,
            VocabParallelEmbedding,
        ):
            has_inverse_hadamard = prefix in self.hadamard_inverse_modules

            if is_layer_skipped_gguf(
                prefix,
                self.unquantized_modules,
                self.packed_modules_mapping,
            ):
                return UnquantizedEmbeddingMethod()

            return GGUFEmbeddingMethod(
                self,
                inverse_hadamard_config=(
                    self.hadamard_config if has_inverse_hadamard else None
                ),
                hadamard_runtime_config=self._resolve_packed_hadamard(prefix),
            )

        # ----------------------------------------------------------
        # MoE
        # ----------------------------------------------------------
        if isinstance(
            layer,
            RoutedExperts,
        ):
            return GGUFMoEMethod(
                self,
                layer.moe_config,
            )

        return None

    def register_linear_layouts(
        self,
        layouts: Mapping[str, "GGUFLinearLayout"],
        prefix: str = "",
    ) -> None:
        """Register GGUF linear layouts before model initialization."""
        self.linear_layouts.update(
            (maybe_prefix(prefix, module_name), layout)
            for module_name, layout in layouts.items()
        )

    def register_hadamard(
        self,
        config: PrismHadamardConfig | None,
        forward_modules: Mapping[str, HadamardRuntimeConfig],
        inverse_modules: set[str],
        prefix: str = "",
    ) -> None:
        self.hadamard_config = config

        self.hadamard_forward_modules.clear()
        self.hadamard_inverse_modules.clear()

        if config is None:
            return

        for name, runtime in forward_modules.items():
            runtime_name = self._to_runtime_module_prefix(name)

            self.hadamard_forward_modules[runtime_name] = runtime

        for name in inverse_modules:
            runtime_name = self._to_runtime_module_prefix(name)

            self.hadamard_inverse_modules.add(runtime_name)

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        """
        Interface for models to update module names referenced in
        quantization configs in order to reflect the vllm model structure

        :param hf_to_vllm_mapper: maps from hf model structure (the assumed
            structure of the qconfig) to vllm model structure
        """
        if self.unquantized_modules is not None:
            self.unquantized_modules = hf_to_vllm_mapper.apply_list(
                self.unquantized_modules
            )
        if self.linear_layouts:
            layouts = self.linear_layouts
            mapped_names = hf_to_vllm_mapper.apply_list(list(layouts))
            self.linear_layouts = dict(zip(mapped_names, layouts.values(), strict=True))
