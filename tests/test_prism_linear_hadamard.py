from __future__ import annotations

from unittest.mock import patch

import torch


from vllm_gguf_plugin.hadamard import (
    HadamardPermutation,
    PrismHadamardConfig,
    apply_forward_hadamard,
    apply_inverse_hadamard,
    fwht_blockwise,
    permute_gdn_v,
)


from vllm_gguf_plugin.quantization.linear import (
    GGUFLinearMethod,
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
        block_size=4,
        transform="normalized-sylvester-walsh-hadamard",
        axis="input-last-dimension",
        sign_mode="identity",
        weight_names=set(),
        inverse_weight_names=set(),
        signs_by_width={},
        gdn_v_grouped=False,
        tied_output=tied_output,
    )


def test_linear_method_stores_hadamard_config():
    hcfg = make_hadamard_config()

    method = GGUFLinearMethod(
        quant_config=None,
        layout=None,
        hadamard_config=hcfg,
        hadamard_permutation=None,
    )

    assert method.hadamard_config is hcfg
    assert method.hadamard_permutation is None


def test_linear_method_stores_gdn_permutation():
    hcfg = make_hadamard_config()

    permutation = HadamardPermutation(
        hd=2,
        nk=2,
        rep=2,
    )

    method = GGUFLinearMethod(
        quant_config=None,
        layout=None,
        hadamard_config=hcfg,
        hadamard_permutation=permutation,
    )

    assert method.hadamard_config is hcfg
    assert method.hadamard_permutation == permutation


def test_none_permutation_does_not_mean_no_hadamard():
    config = GGUFConfig()

    hcfg = make_hadamard_config()

    config.register_hadamard(
        hcfg,
        {
            "lm_head": None,
        },
        set(),
    )

    assert "lm_head" in config.hadamard_forward_modules

    assert (
        config.hadamard_forward_modules["lm_head"]
        is None
    )

    assert config.hadamard_config is hcfg


def test_missing_prefix_means_no_hadamard():
    config = GGUFConfig()

    hcfg = make_hadamard_config()

    config.register_hadamard(
        hcfg,
        {
            "lm_head": None,
        },
        set(),
    )

    assert (
        "model.layers.0.foo"
        not in config.hadamard_forward_modules
    )


def test_gdn_permutation_survives_registration():
    config = GGUFConfig()

    hcfg = make_hadamard_config()

    permutation = HadamardPermutation(
        hd=256,
        nk=8,
        rep=2,
    )

    prefix = "model.layers.0.ssm_out"

    config.register_hadamard(
        hcfg,
        {
            prefix: permutation,
        },
        set(),
    )

    got = config.hadamard_forward_modules[
        prefix
    ]

    assert got == permutation
    assert got.hd == 256
    assert got.nk == 8
    assert got.rep == 2


def test_inverse_module_registration():
    config = GGUFConfig()

    hcfg = make_hadamard_config()

    config.register_hadamard(
        hcfg,
        {},
        {
            "model.embed_tokens",
        },
    )

    assert (
        "model.embed_tokens"
        in config.hadamard_inverse_modules
    )


def test_forward_and_inverse_roles_are_independent():
    config = GGUFConfig()

    hcfg = make_hadamard_config()

    config.register_hadamard(
        hcfg,
        {
            "lm_head": None,
        },
        {
            "model.embed_tokens",
        },
    )

    assert (
        "lm_head"
        in config.hadamard_forward_modules
    )

    assert (
        "lm_head"
        not in config.hadamard_inverse_modules
    )

    assert (
        "model.embed_tokens"
        in config.hadamard_inverse_modules
    )

    assert (
        "model.embed_tokens"
        not in config.hadamard_forward_modules
    )


def test_untied_output_role():
    hcfg = make_hadamard_config(
        tied_output=False,
    )

    assert hcfg.tied_output is False
    assert hcfg.version == 1
