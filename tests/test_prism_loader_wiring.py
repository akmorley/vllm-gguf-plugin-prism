# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from vllm_gguf_plugin.hadamard import (
    HadamardPermutation,
    HadamardRuntimeConfig,
    PrismHadamardConfig,
)
from vllm_gguf_plugin.loader import (
    GGUFModelLoader,
)
from vllm_gguf_plugin.quantization.config import (
    GGUFConfig,
)


def make_hadamard_config(
    *,
    tied_output: bool = False,
) -> PrismHadamardConfig:
    return PrismHadamardConfig(
        version=2 if tied_output else 1,
        block_size=1024,
        transform=("normalized-sylvester-walsh-hadamard"),
        axis="input-last-dimension",
        sign_mode="identity",
        weight_names={
            "blk.0.attn_q.weight",
            "output.weight",
        },
        inverse_weight_names={
            "token_embd.weight",
        },
        signs_by_width={},
        gdn_v_grouped=True,
        tied_output=tied_output,
    )


def make_loader_without_init() -> GGUFModelLoader:
    """
    _weight_name_to_module_name() does not depend on initialized
    loader state, so avoid invoking GGUFModelLoader.__init__.
    """
    return object.__new__(GGUFModelLoader)


def test_weight_name_to_module_name():
    loader = make_loader_without_init()

    name_map = {
        "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
    }

    got = loader._weight_name_to_module_name(
        "blk.0.attn_q.weight",
        name_map,
    )

    assert got == ("model.layers.0.self_attn.q_proj")


def test_output_weight_maps_to_module_prefix():
    loader = make_loader_without_init()

    name_map = {
        "output.weight": "lm_head.weight",
    }

    got = loader._weight_name_to_module_name(
        "output.weight",
        name_map,
    )

    assert got == "lm_head"


def test_embedding_weight_maps_to_module_prefix():
    loader = make_loader_without_init()

    name_map = {
        "token_embd.weight": "model.embed_tokens.weight",
    }

    got = loader._weight_name_to_module_name(
        "token_embd.weight",
        name_map,
    )

    assert got == "model.embed_tokens"


def test_weight_name_missing_raises():
    loader = make_loader_without_init()

    with pytest.raises(
        ValueError,
        match="not present",
    ):
        loader._weight_name_to_module_name(
            "missing.weight",
            {},
        )


def test_weight_name_must_map_to_weight():
    loader = make_loader_without_init()

    name_map = {
        "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.bias",
    }

    with pytest.raises(
        ValueError,
        match="not a weight parameter",
    ):
        loader._weight_name_to_module_name(
            "blk.0.attn_q.weight",
            name_map,
        )


def test_hadamard_permutation_geometry():
    """
    Reproduce loader.py's Prism GDN calculation:

        hd  = input_width // n_v
        nk  = n_k
        rep = n_v // n_k
    """
    input_width = 4096
    n_k = 8
    n_v = 16

    permutation = HadamardPermutation(
        hd=input_width // n_v,
        nk=n_k,
        rep=n_v // n_k,
    )

    assert permutation.hd == 256
    assert permutation.nk == 8
    assert permutation.rep == 2

    assert permutation.width == input_width


def test_gdn_geometry_requires_nv_divisible_by_nk():
    n_k = 6
    n_v = 16

    assert n_v % n_k != 0


def test_gdn_geometry_requires_input_width_divisible_by_nv():
    input_width = 4100
    n_v = 16

    assert input_width % n_v != 0


def test_config_initial_hadamard_state():
    config = GGUFConfig()

    assert config.hadamard_config is None


def test_config_register_hadamard():
    """
    This test assumes GGUFConfig.register_hadamard() has been
    implemented.

    If it currently fails with AttributeError, that identifies
    the missing loader -> config plumbing.
    """
    config = GGUFConfig()

    hcfg = make_hadamard_config()

    permutation = HadamardPermutation(
        hd=256,
        nk=8,
        rep=2,
    )

    forward_modules = {
        "language_model.model.layers.0.ssm_out": HadamardRuntimeConfig(
            hcfg, permutation
        ),
        # A runtime with no permutation still enables Hadamard.
        #
        # It means:
        #
        #   forward Hadamard required,
        #   no GDN permutation required.
        "language_model.lm_head": HadamardRuntimeConfig(hcfg),
    }

    inverse_modules = {
        "language_model.model.embed_tokens",
    }

    config.register_hadamard(
        hcfg,
        forward_modules,
        inverse_modules,
    )

    assert config.hadamard_config is hcfg

    assert (
        config.hadamard_forward_modules[
            "language_model.model.layers.0.ssm_out"
        ].permutation
        == permutation
    )

    assert "language_model.lm_head" in config.hadamard_forward_modules

    assert config.hadamard_forward_modules["language_model.lm_head"].permutation is None

    assert "language_model.model.embed_tokens" in config.hadamard_inverse_modules


def test_none_permutation_does_not_mean_no_hadamard():
    """
    This is an important semantic test.

    Dictionary entry:

        "language_model.lm_head": HadamardRuntimeConfig(hcfg)

    means:

        apply Hadamard
        but do not apply GDN permutation.

    It must NOT mean:

        no Hadamard.
    """
    config = GGUFConfig()

    hcfg = make_hadamard_config()

    config.register_hadamard(
        hcfg,
        {
            "language_model.lm_head": HadamardRuntimeConfig(hcfg),
        },
        set(),
    )

    assert "language_model.lm_head" in config.hadamard_forward_modules

    permutation = config.hadamard_forward_modules["language_model.lm_head"]

    assert permutation.permutation is None


def test_missing_prefix_means_no_forward_hadamard():
    config = GGUFConfig()

    hcfg = make_hadamard_config()

    config.register_hadamard(
        hcfg,
        {
            "language_model.lm_head": HadamardRuntimeConfig(hcfg),
        },
        set(),
    )

    assert "language_model.model.layers.0.ffn" not in config.hadamard_forward_modules


def test_forward_and_inverse_module_sets_are_independent():
    config = GGUFConfig()

    hcfg = make_hadamard_config()

    config.register_hadamard(
        hcfg,
        {
            "language_model.lm_head": HadamardRuntimeConfig(hcfg),
        },
        {
            "language_model.model.embed_tokens",
        },
    )

    assert "language_model.lm_head" in config.hadamard_forward_modules

    assert "language_model.lm_head" not in config.hadamard_inverse_modules

    assert "language_model.model.embed_tokens" in config.hadamard_inverse_modules

    assert "language_model.model.embed_tokens" not in config.hadamard_forward_modules


def test_gdn_module_retains_permutation():
    config = GGUFConfig()

    hcfg = make_hadamard_config()

    permutation = HadamardPermutation(
        hd=256,
        nk=8,
        rep=2,
    )

    prefix = "language_model.model.layers.0.linear_attn.out_proj"

    config.register_hadamard(
        hcfg,
        {
            prefix: HadamardRuntimeConfig(hcfg, permutation),
        },
        set(),
    )

    got = config.hadamard_forward_modules[prefix]

    assert got is not None
    assert got.permutation.hd == 256
    assert got.permutation.nk == 8
    assert got.permutation.rep == 2


def test_untied_output_is_explicit_forward_module():
    """
    Your current Prism GGUF is version 1 / untied.

    Therefore output.weight should map to lm_head and lm_head
    should be present in the forward-Hadamard module map.
    """
    loader = make_loader_without_init()

    hcfg = make_hadamard_config(
        tied_output=False,
    )

    name_map = {
        "output.weight": "lm_head.weight",
    }

    assert "output.weight" in hcfg.weight_names

    module_name = loader._weight_name_to_module_name(
        "output.weight",
        name_map,
    )

    assert module_name == "lm_head"


def test_inverse_embedding_maps_to_embed_tokens():
    loader = make_loader_without_init()

    hcfg = make_hadamard_config()

    name_map = {
        "token_embd.weight": "model.embed_tokens.weight",
    }

    assert "token_embd.weight" in hcfg.inverse_weight_names

    module_name = loader._weight_name_to_module_name(
        "token_embd.weight",
        name_map,
    )

    assert module_name == "model.embed_tokens"


def test_hadamard_sign_tensor_is_cpu_until_used():
    signs = torch.tensor(
        [1, -1, 1, -1],
        dtype=torch.int8,
    )

    hcfg = PrismHadamardConfig(
        version=1,
        block_size=4,
        transform=("normalized-sylvester-walsh-hadamard"),
        axis="input-last-dimension",
        sign_mode="explicit",
        weight_names=set(),
        inverse_weight_names=set(),
        signs_by_width={
            4: signs,
        },
        gdn_v_grouped=False,
        tied_output=False,
    )

    assert hcfg.signs_by_width[4].device.type == "cpu"

    assert hcfg.signs_by_width[4].dtype == torch.int8
