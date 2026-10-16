# SPDX-License-Identifier: Apache-2.0
import gguf
import numpy as np
import pytest
import torch

from vllm_gguf_plugin.quantization import linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("chunk_elems", [None, 3 * 256])
@pytest.mark.parametrize("batch", [32, 300])
def test_dequant_gemm_matches_reference(monkeypatch, chunk_elems, batch):
    # From _DEQUANT_GEMM_MIN_BATCH rows the weight is dequantised for a dense GEMM. Outputs
    # reach about +-32, so the tolerance allows two bf16 rounding steps; an indexing or
    # chunking error would be far larger.
    # chunk_elems=3*256 splits the 10 rows into chunks of 3 (last one partial).
    def no_mmq(*args):
        raise AssertionError("MMQ used above the dequant GEMM threshold")

    monkeypatch.setattr(linear.ops, "ggml_mul_mat_a8", no_mmq)
    if chunk_elems is not None:
        monkeypatch.setattr(linear, "_DEQUANT_GEMM_CHUNK_ELEMS", chunk_elems)
    qtype = gguf.GGMLQuantizationType.Q8_0
    rng = np.random.default_rng(0)
    w_float = rng.standard_normal((10, 256), dtype=np.float32)
    w_q = gguf.quants.quantize(w_float, qtype)
    w_ref = torch.from_numpy(gguf.quants.dequantize(w_q, qtype)).cuda()
    weight = torch.from_numpy(w_q).cuda()
    torch.manual_seed(0)
    x = torch.randn(batch, 256, device="cuda", dtype=torch.bfloat16)

    y = linear._fused_mul_mat_gguf(x, weight, int(qtype))

    assert y.shape == (batch, 10)
    torch.testing.assert_close(y.float(), x.float() @ w_ref.T, rtol=2e-2, atol=2.5e-1)
