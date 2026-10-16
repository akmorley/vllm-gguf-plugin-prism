import struct

import pytest
import torch

import vllm_gguf_plugin._C_gguf  # noqa: F401

from gguf import GGMLQuantizationType

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.quantization.utils import (
    DEQUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
)


PQ2 = int(
    GGMLQuantizationType.PQ2_0
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_pq2_known_block_cuda():
    # code order:
    #
    # 00 -> -1
    # 01 ->  0
    # 10 -> +1
    # 11 -> +2
    #
    # packed low-to-high:
    #
    # 00, 01, 10, 11
    #
    # = 0b11100100 = 0xE4
    packed = bytearray(
        struct.pack("<e", 1.0)
    )

    packed.extend(
        [0xE4] * 32
    )

    W = torch.tensor(
        list(packed),
        dtype=torch.uint8,
        device="cuda",
    )

    got = torch.ops._C_gguf.ggml_dequantize(
        W,
        PQ2,
        1,
        128,
        torch.float32,
    )

    expected = torch.tensor(
        [-1.0, 0.0, 1.0, 2.0] * 32,
        dtype=torch.float32,
    ).reshape(1, 128)

    torch.testing.assert_close(
        got.cpu(),
        expected,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_pq2_multiple_blocks_have_correct_stride():
    packed = bytearray()

    packed.extend(
        struct.pack("<e", 1.0)
    )
    packed.extend(
        [0xE4] * 32
    )

    packed.extend(
        struct.pack("<e", 2.0)
    )
    packed.extend(
        [0xE4] * 32
    )

    W = torch.tensor(
        list(packed),
        dtype=torch.uint8,
        device="cuda",
    )

    got = torch.ops._C_gguf.ggml_dequantize(
        W,
        PQ2,
        1,
        256,
        torch.float32,
    ).cpu()

    expected = torch.tensor(
        (
            [-1.0, 0.0, 1.0, 2.0] * 32
            + [-2.0, 0.0, 2.0, 4.0] * 32
        ),
        dtype=torch.float32,
    ).reshape(1, 256)

    torch.testing.assert_close(
        got,
        expected,
        rtol=0,
        atol=0,
    )


def test_pq2_is_dequant_only():
    qt = GGMLQuantizationType.PQ2_0

    assert qt in DEQUANT_TYPES

    assert qt not in MMVQ_QUANT_TYPES
    assert qt not in MMQ_QUANT_TYPES


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_pq2_ggml_dequantize_wrapper_uses_cuda(
    monkeypatch,
):
    triton_called = False

    def fail_triton(*args, **kwargs):
        nonlocal triton_called
        triton_called = True

        raise AssertionError(
            "PQ2 must not enter Triton dequant"
        )

    monkeypatch.setattr(
        ops,
        "ggml_dequantize_triton",
        fail_triton,
    )

    packed = bytearray(
        struct.pack("<e", 1.0)
    )
    packed.extend(
        [0xE4] * 32
    )

    W = torch.tensor(
        list(packed),
        dtype=torch.uint8,
        device="cuda",
    )

    got = ops.ggml_dequantize(
        W,
        PQ2,
        1,
        128,
        torch.float32,
    )

    assert not triton_called

    assert got.shape == (
        1,
        128,
    )
