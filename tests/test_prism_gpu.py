# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gguf_plugin.hadamard import (
    PrismHadamardConfig,
    apply_forward_hadamard,
    apply_inverse_hadamard,
    fwht_blockwise,
)
from vllm_gguf_plugin.triton.prism import pq2_matmul

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m", [1, 7, 33])
def test_pq2_dense_parity(dtype, m):
    torch.manual_seed(7)
    n, k = 37, 384
    packed = torch.randint(0, 256, (n, k // 128, 34), dtype=torch.uint8, device="cuda")
    scales = torch.randn(n, k // 128, dtype=torch.float16, device="cuda")
    packed[:, :, :2] = scales.view(torch.uint8).reshape(n, k // 128, 2)
    codes = (
        (packed[:, :, 2:, None].to(torch.int32) >> (2 * torch.arange(4, device="cuda")))
        & 3
    ) - 1
    dense = (
        (codes.reshape(n, k // 128, 128).float() * scales[:, :, None])
        .reshape(n, k)
        .to(dtype)
    )
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    torch.testing.assert_close(
        pq2_matmul(x, packed.reshape(n, -1)),
        (x.float() @ dense.float().T).to(dtype),
        atol=0.08 if dtype == torch.bfloat16 else 0.01,
        rtol=0.02 if dtype == torch.bfloat16 else 0.002,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_hadamard_and_graph(dtype):
    width = 5120
    signs = torch.randint(0, 2, (width,), dtype=torch.int8) * 2 - 1
    cfg = PrismHadamardConfig(
        1,
        1024,
        "normalized-sylvester-walsh-hadamard",
        "input-last-dimension",
        "explicit",
        signs_by_width={width: signs},
    )
    x = torch.randn(3, width, device="cuda", dtype=dtype)
    expected = fwht_blockwise(x * signs.to(device="cuda", dtype=dtype), 1024)
    y = apply_forward_hadamard(x, cfg)
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(
        apply_inverse_hadamard(y, cfg),
        x,
        atol=0.03 if dtype == torch.bfloat16 else 0.003,
        rtol=0.03 if dtype == torch.bfloat16 else 0.003,
    )
    assert cfg.signs_for(width, device=x.device, dtype=dtype) is cfg.signs_for(
        width, device=x.device, dtype=dtype
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply_forward_hadamard(x, cfg)
    graph.replay()
    torch.testing.assert_close(captured, expected)
