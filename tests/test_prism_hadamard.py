# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from vllm_gguf_plugin.hadamard import (
    HadamardPermutation,
    PrismHadamardConfig,
    apply_forward_hadamard,
    apply_inverse_hadamard,
    fwht_blockwise,
    permute_gdn_v,
)

def prism_hadamard_matrix(
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Reproduce Prism's explicit Hadamard matrix construction:

        H[row, col] =
            (-1) ** popcount(row & col) / sqrt(n)

    This matches the construction in Prism's llama-model.cpp.
    """
    if n <= 0 or n & (n - 1):
        raise ValueError(
            f"n must be a positive power of two, got {n}"
        )

    scale = 1.0 / math.sqrt(n)

    H = torch.empty(
        (n, n),
        dtype=dtype,
        device=device,
    )

    for row in range(n):
        for col in range(n):
            parity = (row & col).bit_count()

            H[row, col] = (
                -scale
                if parity & 1
                else scale
            )

    return H


class DummyHadamardConfig:
    """
    Minimal config object for exercising the plugin helpers.

    Adjust attribute names here if your real PrismHadamardConfig
    uses a different access pattern.
    """

    def __init__(
        self,
        block_size: int,
        signs_by_width: dict[int, torch.Tensor],
        sign_mode: str = "explicit",
    ):
        self.block_size = block_size
        self.signs_by_width = signs_by_width
        self.sign_mode = sign_mode
        self._device_signs = {}

    signs_for = PrismHadamardConfig.signs_for


def test_prism_matrix_n4_exact():
    H = prism_hadamard_matrix(4)

    expected = 0.5 * torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        H,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_fwht_known_n4_vector():
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    actual = fwht_blockwise(
        x,
        block_size=4,
    )

    expected = torch.tensor(
        [[5.0, -1.0, -2.0, 0.0]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "n",
    [
        2,
        4,
        8,
        16,
        32,
        64,
    ],
)
def test_fwht_matches_prism_matrix_small(n: int):
    torch.manual_seed(1234)

    x = torch.randn(
        5,
        n,
        dtype=torch.float32,
    )

    H = prism_hadamard_matrix(n)

    # Prism does:
    #
    #   ggml_mul_mat(rot, x)
    #
    # For row-vector notation here that corresponds to x @ H.T.
    # The Hadamard matrix is symmetric anyway.
    expected = x @ H.T

    actual = fwht_blockwise(
        x,
        block_size=n,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_fwht_matches_prism_matrix_1024():
    """
    This is the most important direct equivalence test for the
    Prism model if its block size is 1024.
    """
    torch.manual_seed(1234)

    n = 1024

    x = torch.randn(
        3,
        n,
        dtype=torch.float32,
    )

    H = prism_hadamard_matrix(n)

    expected = x @ H.T

    actual = fwht_blockwise(
        x,
        block_size=n,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-5,
        atol=2e-5,
    )


def test_fwht_is_self_inverse():
    torch.manual_seed(1234)

    x = torch.randn(
        4,
        1024,
        dtype=torch.float32,
    )

    y = fwht_blockwise(
        x,
        block_size=1024,
    )

    z = fwht_blockwise(
        y,
        block_size=1024,
    )

    torch.testing.assert_close(
        z,
        x,
        rtol=2e-5,
        atol=2e-5,
    )


def test_fwht_preserves_l2_norm():
    torch.manual_seed(1234)

    x = torch.randn(
        8,
        1024,
        dtype=torch.float32,
    )

    y = fwht_blockwise(
        x,
        block_size=1024,
    )

    before = torch.linalg.vector_norm(
        x,
        dim=-1,
    )

    after = torch.linalg.vector_norm(
        y,
        dim=-1,
    )

    torch.testing.assert_close(
        after,
        before,
        rtol=2e-5,
        atol=2e-5,
    )


def test_forward_sign_order_is_sign_then_hadamard():
    """
    Prism forward path is:

        signs -> Hadamard

    not:

        Hadamard -> signs
    """
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    signs = torch.tensor(
        [1.0, -1.0, 1.0, -1.0],
        dtype=torch.float32,
    )

    expected = fwht_blockwise(
        x * signs,
        block_size=4,
    )

    wrong_order = (
        fwht_blockwise(
            x,
            block_size=4,
        )
        * signs
    )

    assert not torch.allclose(
        expected,
        wrong_order,
    )


def test_inverse_sign_order_is_hadamard_then_sign():
    """
    Prism inverse embedding path is:

        Hadamard -> signs

    not:

        signs -> Hadamard
    """
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    signs = torch.tensor(
        [1.0, -1.0, 1.0, -1.0],
        dtype=torch.float32,
    )

    expected = (
        fwht_blockwise(
            x,
            block_size=4,
        )
        * signs
    )

    wrong_order = fwht_blockwise(
        x * signs,
        block_size=4,
    )

    assert not torch.allclose(
        expected,
        wrong_order,
    )


def test_forward_helper_matches_prism_semantics_without_permutation():
    """
    Directly test apply_forward_hadamard() against the Prism rule:

        x -> signs -> Hadamard
    """
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    signs = torch.tensor(
        [1.0, -1.0, 1.0, -1.0],
        dtype=torch.float32,
    )

    cfg = DummyHadamardConfig(
        block_size=4,
        signs_by_width={
            4: signs,
        },
    )

    expected = fwht_blockwise(
        x * signs,
        block_size=4,
    )

    actual = apply_forward_hadamard(
        x,
        cfg,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_inverse_helper_matches_prism_semantics():
    """
    Directly test apply_inverse_hadamard() against the Prism rule:

        x -> Hadamard -> signs
    """
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    signs = torch.tensor(
        [1.0, -1.0, 1.0, -1.0],
        dtype=torch.float32,
    )

    cfg = DummyHadamardConfig(
        block_size=4,
        signs_by_width={
            4: signs,
        },
    )

    expected = (
        fwht_blockwise(
            x,
            block_size=4,
        )
        * signs
    )

    actual = apply_inverse_hadamard(
        x,
        cfg,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_forward_then_inverse_roundtrip_without_permutation():
    """
    Since:

        forward = H S
        inverse = S H

    and both H and S are self-inverse,

        inverse(forward(x)) == x
    """
    torch.manual_seed(1234)

    width = 1024

    x = torch.randn(
        3,
        width,
        dtype=torch.float32,
    )

    signs = torch.randint(
        low=0,
        high=2,
        size=(width,),
    ).to(torch.float32)

    signs = signs * 2.0 - 1.0

    cfg = DummyHadamardConfig(
        block_size=1024,
        signs_by_width={
            width: signs,
        },
    )

    y = apply_forward_hadamard(
        x,
        cfg,
    )

    z = apply_inverse_hadamard(
        y,
        cfg,
    )

    torch.testing.assert_close(
        z,
        x,
        rtol=3e-5,
        atol=3e-5,
    )


def test_sign_values_are_only_plus_or_minus_one():
    """
    Generic sanity test for a Prism-style sign vector.
    """
    signs = torch.tensor(
        [
            1,
            -1,
            1,
            1,
            -1,
            -1,
        ],
        dtype=torch.float32,
    )

    unique = set(
        signs.tolist()
    )

    assert unique <= {
        -1.0,
        1.0,
    }


def test_prism_sign_slice_logic():
    """
    Reproduce Prism's sequential sign slicing logic:

        width_0 consumes values[0:width_0]
        width_1 consumes the next width_1 values
        etc.
    """

    sign_widths = [
        4,
        8,
    ]

    sign_values = [
        1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
    ]

    signs_by_width = {}

    offset = 0

    for width in sign_widths:
        signs_by_width[width] = torch.tensor(
            sign_values[
                offset:offset + width
            ],
            dtype=torch.float32,
        )

        offset += width

    assert offset == len(sign_values)

    torch.testing.assert_close(
        signs_by_width[4],
        torch.tensor(
            [1, -1, 1, -1],
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        signs_by_width[8],
        torch.tensor(
            [
                1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
            ],
            dtype=torch.float32,
        ),
    )


def test_gdn_permutation_geometry_example():
    """
    Sanity-check only the expected Prism geometry:

        [hd, nk, rep]
            ->
        [hd, rep, nk]

    This does not call the plugin helper directly because implementations
    may differ in how leading dimensions are handled.
    """

    hd = 2
    nk = 2
    rep = 3

    width = hd * nk * rep

    x = torch.arange(
        width,
        dtype=torch.float32,
    ).reshape(
        hd,
        nk,
        rep,
    )

    expected = (
        x
        .permute(
            0,
            2,
            1,
        )
        .contiguous()
        .reshape(-1)
    )

    manual = torch.tensor(
        [
            0, 3,
            1, 4,
            2, 5,
            6, 9,
            7, 10,
            8, 11,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        expected,
        manual,
    )


@pytest.mark.parametrize(
    "width",
    [
        5120,
        6144,
        17408,
    ],
)
def test_prism_model_widths_divisible_by_1024(width: int):
    """
    Expected widths for the current Prism model should each be compatible
    with block_size=1024.
    """
    assert width % 1024 == 0


def test_multi_block_fwht_matches_explicit_matrix():
    """
    Verify blockwise behavior across multiple consecutive Hadamard blocks.
    """
    torch.manual_seed(1234)

    block_size = 8
    width = 24

    x = torch.randn(
        2,
        width,
        dtype=torch.float32,
    )

    H = prism_hadamard_matrix(
        block_size
    )

    expected_blocks = []

    for start in range(
        0,
        width,
        block_size,
    ):
        block = x[
            ...,
            start:start + block_size,
        ]

        expected_blocks.append(
            block @ H.T
        )

    expected = torch.cat(
        expected_blocks,
        dim=-1,
    )

    actual = fwht_blockwise(
        x,
        block_size=block_size,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )



def make_config(
    *,
    block_size: int = 4,
    sign_mode: str = "identity",
    signs_by_width: dict[int, torch.Tensor] | None = None,
    gdn_v_grouped: bool = False,
    tied_output: bool = False,
) -> PrismHadamardConfig:
    return PrismHadamardConfig(
        version=2 if tied_output else 1,
        block_size=block_size,
        transform="normalized-sylvester-walsh-hadamard",
        axis="input-last-dimension",
        sign_mode=sign_mode,
        weight_names=set(),
        inverse_weight_names=set(),
        signs_by_width=signs_by_width or {},
        gdn_v_grouped=gdn_v_grouped,
        tied_output=tied_output,
    )


def explicit_hadamard_4(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Explicit normalized Sylvester H4.

        H4 =
            1  1  1  1
            1 -1  1 -1
            1  1 -1 -1
            1 -1 -1  1

    Normalization is 1 / sqrt(4).
    """
    h = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=x.dtype,
        device=x.device,
    )

    return x @ h.T / math.sqrt(4.0)


def test_fwht_known_vector():
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    got = fwht_blockwise(
        x,
        block_size=4,
    )

    expected = explicit_hadamard_4(x)

    torch.testing.assert_close(
        got,
        expected,
        rtol=0,
        atol=1e-6,
    )


def test_fwht_known_result():
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
    )

    got = fwht_blockwise(
        x,
        4,
    )

    # Raw H4 transform:
    #
    # [10, -2, -4, 0]
    #
    # normalized by sqrt(4)=2.
    expected = torch.tensor(
        [[5.0, -1.0, -2.0, 0.0]],
    )

    torch.testing.assert_close(
        got,
        expected,
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4),
        (3, 4),
        (8, 4),
        (2, 3, 4),
        (1, 8),
        (3, 16),
    ],
)
def test_fwht_is_self_inverse_parametrized(shape):
    torch.manual_seed(1234)

    x = torch.randn(
        shape,
        dtype=torch.float32,
    )

    y = fwht_blockwise(
        x,
        4,
    )

    z = fwht_blockwise(
        y,
        4,
    )

    torch.testing.assert_close(
        z,
        x,
        rtol=1e-5,
        atol=1e-5,
    )


def test_fwht_is_blockwise_not_global():
    x = torch.tensor(
        [[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
        ]]
    )

    got = fwht_blockwise(
        x,
        block_size=4,
    )

    expected = torch.cat(
        [
            explicit_hadamard_4(
                x[:, 0:4]
            ),
            explicit_hadamard_4(
                x[:, 4:8]
            ),
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        got,
        expected,
        rtol=0,
        atol=1e-6,
    )


def test_fwht_preserves_shape():
    x = torch.randn(
        2,
        3,
        16,
    )

    got = fwht_blockwise(
        x,
        4,
    )

    assert got.shape == x.shape


def test_fwht_requires_power_of_two_block_size():
    x = torch.randn(
        1,
        12,
    )

    with pytest.raises(ValueError):
        fwht_blockwise(
            x,
            6,
        )


def test_fwht_requires_divisible_width():
    x = torch.randn(
        1,
        10,
    )

    with pytest.raises(ValueError):
        fwht_blockwise(
            x,
            4,
        )


def test_gdn_permutation_known_values():
    """
    Validate:

        [hd, nk, rep]
            ->
        [hd, rep, nk]

    with:

        hd  = 2
        nk  = 2
        rep = 3
    """
    permutation = HadamardPermutation(
        hd=2,
        nk=2,
        rep=3,
    )

    x = torch.arange(
        12,
        dtype=torch.float32,
    ).reshape(1, 12)

    got = permute_gdn_v(
        x,
        permutation,
    )

    expected = torch.tensor(
        [[
            0.0,
            3.0,
            1.0,
            4.0,
            2.0,
            5.0,
            6.0,
            9.0,
            7.0,
            10.0,
            8.0,
            11.0,
        ]]
    )

    torch.testing.assert_close(
        got,
        expected,
    )


def test_gdn_permutation_matches_reference_reshape():
    permutation = HadamardPermutation(
        hd=3,
        nk=2,
        rep=4,
    )

    x = torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(1, 24)

    got = permute_gdn_v(
        x,
        permutation,
    )

    expected = (
        x.reshape(
            1,
            3,
            2,
            4,
        )
        .transpose(-2, -1)
        .reshape(1, 24)
    )

    torch.testing.assert_close(
        got,
        expected,
    )


def test_gdn_permutation_preserves_leading_dimensions():
    permutation = HadamardPermutation(
        hd=2,
        nk=2,
        rep=2,
    )

    x = torch.arange(
        3 * 5 * 8,
        dtype=torch.float32,
    ).reshape(
        3,
        5,
        8,
    )

    got = permute_gdn_v(
        x,
        permutation,
    )

    assert got.shape == x.shape


def test_gdn_permutation_width_property():
    permutation = HadamardPermutation(
        hd=64,
        nk=8,
        rep=4,
    )

    assert permutation.width == (
        64 * 8 * 4
    )


def test_gdn_permutation_rejects_wrong_width():
    permutation = HadamardPermutation(
        hd=2,
        nk=2,
        rep=3,
    )

    x = torch.randn(
        1,
        13,
    )

    with pytest.raises(
        ValueError,
        match="width",
    ):
        permute_gdn_v(
            x,
            permutation,
        )


def test_forward_identity_sign_mode_equals_fwht():
    cfg = make_config(
        block_size=4,
        sign_mode="identity",
    )

    torch.manual_seed(1)

    x = torch.randn(
        2,
        4,
    )

    got = apply_forward_hadamard(
        x,
        cfg,
    )

    expected = fwht_blockwise(
        x,
        4,
    )

    torch.testing.assert_close(
        got,
        expected,
    )


def test_forward_signs_are_applied_before_fwht():
    signs = torch.tensor(
        [
            1,
            -1,
            1,
            -1,
        ],
        dtype=torch.int8,
    )

    cfg = make_config(
        block_size=4,
        sign_mode="explicit",
        signs_by_width={
            4: signs,
        },
    )

    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]]
    )

    got = apply_forward_hadamard(
        x,
        cfg,
    )

    expected = fwht_blockwise(
        x * signs.float(),
        4,
    )

    torch.testing.assert_close(
        got,
        expected,
        rtol=0,
        atol=1e-6,
    )


def test_inverse_is_fwht_then_signs():
    signs = torch.tensor(
        [
            1,
            -1,
            -1,
            1,
        ],
        dtype=torch.int8,
    )

    cfg = make_config(
        block_size=4,
        sign_mode="explicit",
        signs_by_width={
            4: signs,
        },
    )

    torch.manual_seed(2)

    x = torch.randn(
        3,
        4,
    )

    got = apply_inverse_hadamard(
        x,
        cfg,
    )

    expected = (
        fwht_blockwise(
            x,
            4,
        )
        * signs.float()
    )

    torch.testing.assert_close(
        got,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_forward_then_inverse_roundtrip():
    signs = torch.tensor(
        [
            1,
            -1,
            -1,
            1,
        ],
        dtype=torch.int8,
    )

    cfg = make_config(
        block_size=4,
        sign_mode="explicit",
        signs_by_width={
            4: signs,
        },
    )

    torch.manual_seed(3)

    x = torch.randn(
        5,
        4,
    )

    forward = apply_forward_hadamard(
        x,
        cfg,
    )

    recovered = apply_inverse_hadamard(
        forward,
        cfg,
    )

    torch.testing.assert_close(
        recovered,
        x,
        rtol=1e-5,
        atol=1e-5,
    )


def test_forward_gdn_order_is_permute_sign_fwht():
    """
    This test pins the Prism forward transform order with the GDN layout
    supplied before the runtime helper:

        GDN permutation
        -> signs
        -> FWHT
    """
    permutation = HadamardPermutation(
        hd=1,
        nk=2,
        rep=2,
    )

    signs = torch.tensor(
        [
            1,
            -1,
            1,
            -1,
        ],
        dtype=torch.int8,
    )

    cfg = make_config(
        block_size=4,
        sign_mode="explicit",
        signs_by_width={
            4: signs,
        },
        gdn_v_grouped=True,
    )

    x = torch.tensor(
        [[
            1.0,
            2.0,
            3.0,
            4.0,
        ]]
    )

    got = apply_forward_hadamard(
        permute_gdn_v(x, permutation),
        cfg,
    )

    permuted = (
        x.reshape(
            1,
            1,
            2,
            2,
        )
        .transpose(-2, -1)
        .reshape(1, 4)
    )

    expected = fwht_blockwise(
        permuted
        * signs.float(),
        4,
    )

    torch.testing.assert_close(
        got,
        expected,
        rtol=0,
        atol=1e-6,
    )


def test_forward_gdn_result_changes_if_order_is_wrong():
    """
    This catches accidental:

        signs -> permutation -> FWHT

    instead of:

        permutation -> signs -> FWHT
    """
    permutation = HadamardPermutation(
        hd=1,
        nk=2,
        rep=2,
    )

    signs = torch.tensor(
        [
            1,
            -1,
            1,
            -1,
        ],
        dtype=torch.int8,
    )

    cfg = make_config(
        block_size=4,
        sign_mode="explicit",
        signs_by_width={
            4: signs,
        },
        gdn_v_grouped=True,
    )

    x = torch.tensor(
        [[
            1.0,
            2.0,
            4.0,
            8.0,
        ]]
    )

    correct = apply_forward_hadamard(
        permute_gdn_v(x, permutation),
        cfg,
    )

    wrong = permute_gdn_v(
        x * signs.float(),
        permutation,
    )

    wrong = fwht_blockwise(
        wrong,
        4,
    )

    assert not torch.allclose(
        correct,
        wrong,
    )



@pytest.mark.parametrize(
    "sign_mode",
    ["none", "identity"],
)
def test_no_sign_modes_do_not_require_sign_vector(
    sign_mode,
):
    cfg = make_config(
        block_size=4,
        sign_mode=sign_mode,
        signs_by_width={},
    )

    torch.manual_seed(123)

    x = torch.randn(
        2,
        8,
    )

    expected = fwht_blockwise(
        x,
        block_size=4,
    )

    forward = apply_forward_hadamard(
        x,
        cfg,
    )

    inverse = apply_inverse_hadamard(
        x,
        cfg,
    )

    torch.testing.assert_close(
        forward,
        expected,
    )

    torch.testing.assert_close(
        inverse,
        expected,
    )



def test_explicit_signs_missing_width_raises():
    signs = torch.tensor(
        [1, -1, 1, -1],
        dtype=torch.int8,
    )

    cfg = make_config(
        block_size=4,
        sign_mode="explicit",
        signs_by_width={
            4: signs,
        },
    )

    x = torch.randn(
        1,
        8,
    )

    with pytest.raises(ValueError):
        apply_forward_hadamard(
            x,
            cfg,
        )


def test_inverse_explicit_signs_missing_width_raises():
    signs = torch.tensor(
        [1, -1, 1, -1],
        dtype=torch.int8,
    )

    cfg = make_config(
        block_size=4,
        sign_mode="explicit",
        signs_by_width={
            4: signs,
        },
    )

    x = torch.randn(
        1,
        8,
    )

    with pytest.raises(
        ValueError,
        match="missing.*width 8",
    ):
        apply_inverse_hadamard(
            x,
            cfg,
        )



def test_sign_mode_none_allows_missing_width():
    cfg = make_config(
        block_size=4,
        sign_mode="none",
        signs_by_width={},
    )

    x = torch.randn(
        1,
        8,
    )

    actual = apply_forward_hadamard(
        x,
        cfg,
    )

    expected = fwht_blockwise(
        x,
        block_size=4,
    )

    torch.testing.assert_close(
        actual,
        expected,
    )



def test_signs_preserve_activation_dtype():
    signs = torch.tensor(
        [1, -1, 1, -1],
        dtype=torch.int8,
    )

    cfg = make_config(
        block_size=4,
        sign_mode="explicit",
        signs_by_width={
            4: signs,
        },
    )

    x = torch.randn(
        2,
        4,
        dtype=torch.float16,
    )

    got = apply_forward_hadamard(
        x,
        cfg,
    )

    assert got.dtype == torch.float16
