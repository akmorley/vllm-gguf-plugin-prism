# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from gguf import GGUFReader


PRISM_HADAMARD_PREFIX = "prism.hadamard."

SUPPORTED_VERSIONS = {1, 2}
SUPPORTED_TRANSFORM = "normalized-sylvester-walsh-hadamard"
SUPPORTED_AXIS = "input-last-dimension"
SUPPORTED_SIGN_MODES = {"identity", "explicit"}


@dataclass(frozen=True)
class HadamardPermutation:
    """
    Optional Prism GDN feature permutation.

    Prism describes the incoming feature order as:

        [hd, nk, rep]

    while the folded weight was produced in:

        [hd, rep, nk]

    The permutation therefore swaps the final nk/rep dimensions before
    flattening the feature dimension again.
    """

    hd: int
    nk: int
    rep: int

    @property
    def width(self) -> int:
        return self.hd * self.nk * self.rep


@dataclass
class PrismHadamardConfig:
    """
    Parsed `prism.hadamard.*` GGUF metadata.

    Keep sign vectors as CPU integer tensors here. Move/cast them only when
    applying a transform, because this config is normally created before the
    final execution device/dtype is known.
    """

    version: int
    block_size: int
    transform: str
    axis: str
    sign_mode: str

    weight_names: set[str] = field(default_factory=set)
    inverse_weight_names: set[str] = field(default_factory=set)

    # Keyed by activation width.
    signs_by_width: dict[int, torch.Tensor] = field(default_factory=dict)

    gdn_v_grouped: bool = False
    tied_output: bool = False

    def has_forward_transform(self, weight_name: str) -> bool:
        return weight_name in self.weight_names

    def has_inverse_transform(self, weight_name: str) -> bool:
        return weight_name in self.inverse_weight_names

    def signs_for(
        self,
        width: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """
        Return the explicit sign vector for an activation width.

        Identity sign mode returns None.
        """
        if self.sign_mode != "explicit":
            return None

        try:
            signs = self.signs_by_width[width]
        except KeyError as exc:
            raise ValueError(
                "Prism Hadamard metadata has no sign vector "
                f"for activation width {width}"
            ) from exc

        return signs.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

@dataclass
class HadamardRuntimeConfig:
    config: PrismHadamardConfig
    permutation: HadamardPermutation | None = None
    skip_input_layout: bool = False



def _field_contents(
    reader: GGUFReader,
    key: str,
    *,
    required: bool,
    default: Any = None,
) -> Any:
    """
    Read one GGUF metadata value using ReaderField.contents().
    """
    field = reader.get_field(key)

    if field is None:
        if required:
            raise ValueError(
                f"required GGUF metadata key is missing: {key}"
            )
        return default

    return field.contents()


def _scalar(
    reader: GGUFReader,
    key: str,
    *,
    required: bool = True,
    default: Any = None,
) -> Any:
    """
    Read metadata expected to contain one scalar.

    ReaderField.contents() returns scalars for scalar GGUF fields in current
    gguf-py, but this also tolerates a one-element list/tuple for robustness.
    """
    value = _field_contents(
        reader,
        key,
        required=required,
        default=default,
    )

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"{key} must be scalar, got {len(value)} values"
            )
        return value[0]

    return value


def _array(
    reader: GGUFReader,
    key: str,
    *,
    required: bool = True,
) -> list[Any]:
    value = _field_contents(
        reader,
        key,
        required=required,
        default=[],
    )

    if value is None:
        return []

    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{key} must be a GGUF array, got {type(value).__name__}"
        )

    return list(value)


def _validate_power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(
            f"{name} must be a positive power of two, got {value}"
        )


def _parse_signs(
    reader: GGUFReader,
    *,
    block_size: int,
    sign_mode: str,
) -> dict[int, torch.Tensor]:
    if sign_mode == "identity":
        return {}

    if sign_mode != "explicit":
        raise ValueError(
            f"unsupported prism.hadamard.sign_mode: {sign_mode!r}"
        )

    widths_raw = _array(
        reader,
        PRISM_HADAMARD_PREFIX + "sign_widths",
    )
    values_raw = _array(
        reader,
        PRISM_HADAMARD_PREFIX + "sign_values",
    )

    widths = [int(v) for v in widths_raw]
    values = [int(v) for v in values_raw]

    if not widths:
        raise ValueError(
            "prism.hadamard.sign_mode is explicit but "
            "prism.hadamard.sign_widths is empty"
        )

    result: dict[int, torch.Tensor] = {}

    offset = 0

    for width in widths:
        if width <= 0:
            raise ValueError(
                f"invalid Prism Hadamard sign width: {width}"
            )

        if width % block_size != 0:
            raise ValueError(
                f"Prism Hadamard sign width {width} is not divisible "
                f"by block size {block_size}"
            )

        end = offset + width

        if end > len(values):
            raise ValueError(
                "prism.hadamard.sign_values is shorter than described "
                "by prism.hadamard.sign_widths"
            )

        chunk = values[offset:end]

        if any(v not in (-1, 1) for v in chunk):
            raise ValueError(
                "prism.hadamard.sign_values must contain only -1 or +1"
            )

        if width in result:
            raise ValueError(
                f"duplicate Prism Hadamard sign width: {width}"
            )

        result[width] = torch.tensor(
            chunk,
            dtype=torch.int8,
            device="cpu",
        )

        offset = end

    if offset != len(values):
        raise ValueError(
            "prism.hadamard.sign_values length does not match "
            "prism.hadamard.sign_widths"
        )

    return result

def _get_signs_for_width(
    cfg: PrismHadamardConfig,
    width: int,
) -> torch.Tensor | None:
    if cfg.sign_mode in ("none", "identity"):
        return None

    if cfg.sign_mode == "explicit":
        signs = cfg.signs_by_width.get(width)

        if signs is None:
            raise ValueError(
                "Prism Hadamard explicit signs are missing "
                f"for width {width}. "
                f"Available widths: "
                f"{sorted(cfg.signs_by_width)}"
            )

        if signs.numel() != width:
            raise ValueError(
                "Prism Hadamard sign vector has wrong length "
                f"for width {width}: "
                f"got {signs.numel()}"
            )

        return signs

    raise ValueError(
        f"Unsupported Prism Hadamard sign_mode: "
        f"{cfg.sign_mode!r}"
    )




def read_prism_hadamard_config(
    source: str | Path | GGUFReader,
) -> PrismHadamardConfig | None:
    """
    Parse Prism Hadamard metadata from a GGUF.

    Returns None when the GGUF has no `prism.hadamard.version`, allowing
    ordinary GGUF models to continue through the plugin unchanged.
    """
    owns_reader = not isinstance(source, GGUFReader)

    reader = (
        GGUFReader(str(source))
        if owns_reader
        else source
    )

    version_raw = _scalar(
        reader,
        PRISM_HADAMARD_PREFIX + "version",
        required=False,
        default=None,
    )

    if version_raw is None:
        return None

    version = int(version_raw)

    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported prism.hadamard.version: {version}"
        )

    tied_output = bool(
        _scalar(
            reader,
            PRISM_HADAMARD_PREFIX + "tied_output",
            required=False,
            default=False,
        )
    )

    # This reproduces Prism's version/tied-output contract.
    if (version == 2) != tied_output:
        raise ValueError(
            "prism.hadamard version 2 requires tied_output=true; "
            "version 1 forbids it"
        )

    block_size = int(
        _scalar(
            reader,
            PRISM_HADAMARD_PREFIX + "block_size",
        )
    )

    _validate_power_of_two(
        block_size,
        "prism.hadamard.block_size",
    )

    transform = str(
        _scalar(
            reader,
            PRISM_HADAMARD_PREFIX + "transform",
        )
    )

    if transform != SUPPORTED_TRANSFORM:
        raise ValueError(
            "unsupported prism.hadamard.transform: "
            f"{transform!r}"
        )

    axis = str(
        _scalar(
            reader,
            PRISM_HADAMARD_PREFIX + "axis",
        )
    )

    if axis != SUPPORTED_AXIS:
        raise ValueError(
            f"unsupported prism.hadamard.axis: {axis!r}"
        )

    sign_mode = str(
        _scalar(
            reader,
            PRISM_HADAMARD_PREFIX + "sign_mode",
        )
    )

    if sign_mode not in SUPPORTED_SIGN_MODES:
        raise ValueError(
            "unsupported prism.hadamard.sign_mode: "
            f"{sign_mode!r}"
        )

    weight_names = {
        str(v)
        for v in _array(
            reader,
            PRISM_HADAMARD_PREFIX + "weight_names",
        )
    }

    if not weight_names:
        raise ValueError(
            "prism.hadamard.weight_names is empty"
        )

    inverse_weight_names = {
        str(v)
        for v in _array(
            reader,
            PRISM_HADAMARD_PREFIX + "inverse_weight_names",
            required=False,
        )
    }

    overlap = weight_names & inverse_weight_names

    # Normally Prism keeps the forward/inverse manifest entries distinct.
    # Tied-output handling later aliases token_embd.weight on the runtime side,
    # so an accidental manifest overlap should still be rejected here.
    if overlap:
        raise ValueError(
            "Prism Hadamard weights appear in both forward and inverse "
            f"metadata sets: {sorted(overlap)!r}"
        )

    signs_by_width = _parse_signs(
        reader,
        block_size=block_size,
        sign_mode=sign_mode,
    )

    gdn_v_grouped = bool(
        _scalar(
            reader,
            PRISM_HADAMARD_PREFIX + "gdn_v_grouped",
            required=False,
            default=False,
        )
    )

    if tied_output:
        if version != 2:
            raise ValueError(
                "prism.hadamard.tied_output requires version 2"
            )

        if "token_embd.weight" not in inverse_weight_names:
            raise ValueError(
                "prism.hadamard.tied_output requires "
                "token_embd.weight in inverse_weight_names"
            )

    return PrismHadamardConfig(
        version=version,
        block_size=block_size,
        transform=transform,
        axis=axis,
        sign_mode=sign_mode,
        weight_names=weight_names,
        inverse_weight_names=inverse_weight_names,
        signs_by_width=signs_by_width,
        gdn_v_grouped=gdn_v_grouped,
        tied_output=tied_output,
    )


def permute_gdn_v(
    x: torch.Tensor,
    permutation: HadamardPermutation,
) -> torch.Tensor:
    """
    Prism's tiled -> grouped GDN V feature permutation:

        [hd, nk, rep] -> [hd, rep, nk]

    The operation is performed independently for every leading activation
    position.
    """
    if x.ndim == 0:
        raise ValueError(
            "GDN permutation requires an activation feature dimension"
        )

    width = x.shape[-1]

    if width != permutation.width:
        raise ValueError(
            "GDN permutation width mismatch: "
            f"x.shape[-1]={width}, "
            f"hd*nk*rep={permutation.width}"
        )

    original_shape = x.shape

    x = x.reshape(
        *original_shape[:-1],
        permutation.hd,
        permutation.nk,
        permutation.rep,
    )

    x = x.transpose(-2, -1)

    return x.reshape(original_shape)


def fwht_blockwise(
    x: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Apply a normalized Sylvester/Walsh-Hadamard transform independently
    to each block of `block_size` values along the last dimension.

    Transform is normalized by 1 / sqrt(block_size).
    """
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            f"width {x.shape[-1]} is not divisible "
            f"by Hadamard block size {block_size}"
        )

    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(
            f"Hadamard block size must be a power of two, "
            f"got {block_size}"
        )

    original_shape = x.shape
    original_dtype = x.dtype

    # Each row here is ONE contiguous Prism Hadamard block.
    #
    # [T, 6144] -> [T * 6, 1024]
    y = x.reshape(-1, block_size).float()

    h = 1

    while h < block_size:
        # Split every row into independent butterfly groups:
        #
        # [..., 2*h] = [a | b]
        y = y.reshape(
            -1,
            block_size // (2 * h),
            2,
            h,
        )

        a = y[:, :, 0, :]
        b = y[:, :, 1, :]

        # Don't update through aliased views.
        y = torch.stack(
            (
                a + b,
                a - b,
            ),
            dim=2,
        )

        y = y.reshape(
            -1,
            block_size,
        )

        h *= 2

    # Prism uses normalized Sylvester Hadamard.
    y = y * (1.0 / math.sqrt(block_size))

    return y.reshape(original_shape).to(original_dtype)




def apply_forward_hadamard(
    x: torch.Tensor,
    config: PrismHadamardConfig,
) -> torch.Tensor:
    # `permutation` is intentionally ignored on the runtime
    # activation path.
    #
    # The Qwen3.5 GGUF adapter already converts the GDN/value-head
    # layout into vLLM order while loading the associated weights.
    # vLLM's flattened GDN output therefore corresponds to Prism's
    # post-permutation activation order.

        
    signs = _get_signs_for_width(config,x.shape[-1])

    if signs is not None:
        signs = signs.to(
            device=x.device,
            dtype=x.dtype,
        )

        x = x * signs

    x =  fwht_blockwise(x,config.block_size)

    return x


def apply_inverse_hadamard(
    x: torch.Tensor,
    config: PrismHadamardConfig,
) -> torch.Tensor:
    # Prism inverse semantics:
    #     H -> signs
    #
    # H is self-inverse because it is the normalized Sylvester
    # Hadamard transform.

    x = fwht_blockwise(
        x,
        config.block_size,
    )

    signs = _get_signs_for_width(
        config,
        x.shape[-1],
    )

    if signs is not None:
        signs = signs.to(
            device=x.device,
            dtype=x.dtype,
        )
        x = x * signs

    return x
