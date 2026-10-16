# SPDX-License-Identifier: Apache-2.0

import math
import struct

import pytest
import torch
import vllm_gguf_plugin._C_gguf
from gguf import GGMLQuantizationType


QK_PQ2_0 = 128
PQ2_BLOCK_BYTES = 34  # fp16 scale (2 B) + 128 * 2 bits (32 B)
PQ2_TYPE = int(GGMLQuantizationType.PQ2_0)


def make_values(n: int, seed: int) -> torch.Tensor:
    """Equivalent role to Prism's make_weights()."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    return torch.normal(
        mean=0.0,
        std=0.5,
        size=(n,),
        generator=generator,
        dtype=torch.float32,
    )


def c_round(x: float) -> int:
    """
    Match C roundf(): halfway values round away from zero.

    Python round() uses bankers rounding, so don't use it here.
    """
    if x >= 0.0:
        return math.floor(x + 0.5)

    return math.ceil(x - 0.5)


def quantize_pq2_reference(x: torch.Tensor) -> bytes:
    """
    Reference encoder matching Prism's quantize_row_pq2_0_ref().

    Each 128-weight block is:

        fp16 d
        32 bytes of 2-bit codes

    code:
        0 -> -1
        1 ->  0
        2 -> +1
        3 -> +2

    reconstructed weight:
        (code - 1) * d
    """
    x = x.detach().cpu().float().contiguous()

    assert x.numel() % QK_PQ2_0 == 0

    output = bytearray()

    for block_start in range(0, x.numel(), QK_PQ2_0):
        block = x[block_start : block_start + QK_PQ2_0]

        d = float(block.abs().max().item())
        inv_d = 1.0 / d if d > 0.0 else 0.0

        # Important: Prism stores the scale as FP16.
        output.extend(struct.pack("<e", d))

        qs = bytearray(QK_PQ2_0 // 4)

        for j in range(QK_PQ2_0):
            w = float(block[j].item())

            q = c_round(w * inv_d) + 1
            q = max(0, min(3, q))

            byte_index = j // 4
            bit_offset = (j % 4) * 2

            qs[byte_index] |= q << bit_offset

        output.extend(qs)

    return bytes(output)


def dequantize_pq2_reference(
    packed: bytes,
    num_values: int,
) -> torch.Tensor:
    """
    Reference decoder matching Prism's dequantize_row_pq2_0().
    """
    assert num_values % QK_PQ2_0 == 0

    num_blocks = num_values // QK_PQ2_0

    assert len(packed) == num_blocks * PQ2_BLOCK_BYTES

    output = torch.empty(num_values, dtype=torch.float32)

    for ib in range(num_blocks):
        offset = ib * PQ2_BLOCK_BYTES

        d = struct.unpack_from("<e", packed, offset)[0]

        qs_offset = offset + 2

        for j in range(QK_PQ2_0):
            byte_index = j // 4
            bit_offset = (j % 4) * 2

            byte = packed[qs_offset + byte_index]
            q = (byte >> bit_offset) & 0x03

            output[ib * QK_PQ2_0 + j] = (q - 1) * d

    return output


def packed_bytes_to_cuda_tensor(packed: bytes) -> torch.Tensor:
    """
    Make a raw uint8 CUDA tensor containing the exact GGUF block bytes.
    """
    # bytearray is writable, avoiding the read-only-buffer warning that
    # torch.frombuffer(bytes(...)) can produce.
    buf = bytearray(packed)

    cpu = torch.frombuffer(
        buf,
        dtype=torch.uint8,
    ).clone()

    return cpu.cuda()


def relative_rms(
    got: torch.Tensor,
    ref: torch.Tensor,
) -> float:
    got = got.double().cpu()
    ref = ref.double().cpu()

    diff = got - ref

    se = torch.sum(diff * diff).item()
    sr = torch.sum(ref * ref).item()

    if sr == 0.0:
        return 0.0 if se == 0.0 else math.inf

    return math.sqrt(se / sr)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PQ2 CUDA dequantization test requires CUDA",
)


def test_pq2_known_block():
    import struct

    # Scale = exactly 1.0 in FP16.
    packed = bytearray(struct.pack("<e", 1.0))

    # Repeating codes:
    #   00 -> -1
    #   01 ->  0
    #   10 -> +1
    #   11 -> +2
    #
    # Four codes packed in little 2-bit order:
    #
    # bits = 11 10 01 00
    #      = 0b11100100
    #      = 0xE4
    packed.extend([0xE4] * 32)

    packed_cuda = torch.tensor(
        list(packed),
        dtype=torch.uint8,
        device="cuda",
    )

    got = torch.ops._C_gguf.ggml_dequantize(
        packed_cuda,
        142,
        1,
        128,
        torch.float32,
    )

    expected = torch.tensor(
        [-1.0, 0.0, 1.0, 2.0] * 32,
        dtype=torch.float32,
    ).reshape(1, 128)

    print("got first 16:     ", got[0, :16].cpu())
    print("expected first 16:", expected[0, :16])

    torch.testing.assert_close(
        got.cpu(),
        expected,
        rtol=0,
        atol=0,
    )

def test_pq2_known_multiple_blocks():
    import struct

    packed = bytearray()

    # Block 0: scale 1
    packed.extend(struct.pack("<e", 1.0))
    packed.extend([0xE4] * 32)

    # Block 1: scale 2
    packed.extend(struct.pack("<e", 2.0))
    packed.extend([0xE4] * 32)

    packed_cuda = torch.tensor(
        list(packed),
        dtype=torch.uint8,
        device="cuda",
    )

    got = torch.ops._C_gguf.ggml_dequantize(
        packed_cuda,
        142,
        1,
        256,
        torch.float32,
    ).cpu()[0]

    expected0 = torch.tensor(
        [-1.0, 0.0, 1.0, 2.0] * 32
    )

    expected1 = torch.tensor(
        [-2.0, 0.0, 2.0, 4.0] * 32
    )

    expected = torch.cat(
        [expected0, expected1]
    ).float()

    print("block 0:", got[:16])
    print("block 1:", got[128:144])

    torch.testing.assert_close(
        got,
        expected,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "k",
    [
        128,
        256,
        384,
        512,
        640,
    ],
)
def test_pq2_dequantize_row_shapes(k: int):
    """
    Test every legal PQ2 row width that Prism explicitly cares about.

    In particular, 128 and 384 are valid even though they are not
    multiples of 256.
    """
    m = 4

    weights = make_values(
        k * m,
        seed=1234 + k,
    ).reshape(m, k)

    packed = quantize_pq2_reference(
        weights.reshape(-1)
    )

    reference = dequantize_pq2_reference(
        packed,
        k * m,
    ).reshape(m, k)

    packed_cuda = packed_bytes_to_cuda_tensor(packed)

    # Call the CUDA extension directly.
    #
    # This deliberately bypasses ops.ggml_dequantize() during initial
    # development because that wrapper may route unknown quant types to
    # the Triton fallback.
    got = torch.ops._C_gguf.ggml_dequantize(
        packed_cuda,
        PQ2_TYPE,
        m,
        k,
        torch.float32,
    )

    assert got.shape == (m, k)

    torch.testing.assert_close(
        got.cpu(),
        reference,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PQ2 CUDA test requires CUDA",
)
@pytest.mark.parametrize(
    "k",
    [
        128,
        256,
        384,
        512,
        640,
    ],
)
@pytest.mark.parametrize(
    "n",
    [
        1,
        4,
        16,
    ],
)
def test_pq2_dequant_matmul_row_shapes(
    k: int,
    n: int,
):
    """
    Python equivalent of Prism's test-pq2-row-shapes.cpp.

    Weight matrix:
        W: [M, K]

    Activations:
        X: [N, K]

    Result:
        Y: [N, M]
    """
    m = 4

    # ----- Build the packed PQ2 model weight -----

    weights = make_values(
        k * m,
        seed=1234 + k,
    ).reshape(m, k)

    packed = quantize_pq2_reference(
        weights.reshape(-1)
    )

    weight_ref = dequantize_pq2_reference(
        packed,
        k * m,
    ).reshape(m, k)

    # ----- Activations -----

    x = make_values(
        k * n,
        seed=99,
    ).reshape(n, k)

    # Reference equivalent to Prism:
    #
    #   dequantized_weight x exact_float_activation
    #
    reference = x @ weight_ref.T

    # ----- Plugin CUDA dequantization -----

    packed_cuda = packed_bytes_to_cuda_tensor(packed)

    weight_cuda = torch.ops._C_gguf.ggml_dequantize(
        packed_cuda,
        PQ2_TYPE,
        m,
        k,
        torch.float32,
    )

    got = (
        x.cuda()
        @ weight_cuda.T
    ).cpu()

    err = relative_rms(
        got,
        reference,
    )

    print(
        f"K={k:4d} "
        f"M={m} "
        f"N={n:2d} "
        f"rel_rms={err:.8f}"
    )

    # This path does not quantize activations to Q8 yet, so its error
    # should be drastically smaller than Prism's 0.05 MMVQ/MMQ threshold.
    assert err < 1e-5


