# SPDX-License-Identifier: Apache-2.0
"""Triton masked-plane prefill (`_mmq_planes`) versus `_mmq` and the CUDA `pq2_mma_prefill` kernel."""
import pytest
import torch

from vllm_gguf_plugin.triton import pq2_mmq as mq
from vllm_gguf_plugin.triton.pq2_layout import prepare_pq2_layout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6),
    reason="SM86 CUDA required",
)


@pytest.fixture(autouse=True)
def _new_default_path_off(monkeypatch):
    """These tests exercise the Triton plane and CUDA prefill paths (fallbacks of the Gluon prefill kernel, the default); the Gluon tests switch it on."""
    monkeypatch.setattr(mq, "_MMQ_GLUON", False)


_WEIGHTS = {}
SHAPES = [(5120, 6144), (14336, 5120), (5120, 17408)]


def _weight(n, k):
    if (n, k) not in _WEIGHTS:
        torch.manual_seed(7)
        raw = torch.randint(0, 256, (n, k // 128, 34), device="cuda", dtype=torch.uint8)
        raw[:, :, :2] = (torch.rand(n, k // 128, device="cuda") * 0.03 + 0.001).half().view(torch.uint8).reshape(
            n, k // 128, 2)
        _WEIGHTS[(n, k)] = prepare_pq2_layout(raw.reshape(n, -1))
    return _WEIGHTS[(n, k)]


def _run(monkeypatch, x, w, planes, cuda, tiles=None):
    monkeypatch.setattr(mq, "_MMQ_PLANES", planes)
    monkeypatch.setattr(mq, "_MMA_PREFILL", cuda)
    if tiles is not None:
        monkeypatch.setattr(mq, "_PLANES_TILES", tiles)
    return mq.pq2_mmq(x, w, prepared=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [64, 200, 513])
@pytest.mark.parametrize("shape", SHAPES)
def test_planes_bit_exact(monkeypatch, shape, m, dtype):
    n, k = shape
    w = _weight(n, k)
    torch.manual_seed(m)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    expected = _run(monkeypatch, x, w, planes=False, cuda=False)                       # Triton _mmq
    assert torch.equal(_run(monkeypatch, x, w, planes=True, cuda=False), expected)     # fallback replacement
    tiles = {shape: ((1 << 30, (64, 128, 4, 3), True),)}
    assert torch.equal(_run(monkeypatch, x, w, planes=True, cuda=True, tiles=tiles), expected)  # table entry
    if mq._mma_prefill_available() and k % 256 == 0:
        assert torch.equal(_run(monkeypatch, x, w, planes=False, cuda=True), expected)  # CUDA kernel


def test_planes_dispatch(monkeypatch):
    n, k = 5120, 6144
    w = _weight(n, k)
    calls = []
    real = mq._mmq_planes_run
    monkeypatch.setattr(mq, "_mmq_planes_run", lambda *a: calls.append(a[-1]) or real(*a))
    x = torch.randn(128, k, device="cuda", dtype=torch.bfloat16)
    # listed and faster: the Triton kernel replaces the CUDA kernel, with the listed tile for the M bucket
    tiles = {(n, k): ((100, (32, 128, 4, 2), True), (1 << 30, (64, 128, 4, 3), True))}
    _run(monkeypatch, x, w, planes=True, cuda=True, tiles=tiles)
    _run(monkeypatch, x[:64], w, planes=True, cuda=True, tiles=tiles)
    assert calls == [(64, 128, 4, 3), (32, 128, 4, 2)]
    calls.clear()
    # listed but slower than CUDA: CUDA keeps it (when built); without CUDA the listed tile replaces _mmq
    slower = {(n, k): ((1 << 30, (64, 128, 4, 3), False),)}
    _run(monkeypatch, x, w, planes=True, cuda=True, tiles=slower)
    assert calls == ([] if mq._mma_prefill_available() else [(64, 128, 4, 3)])
    calls.clear()
    _run(monkeypatch, x, w, planes=True, cuda=False, tiles=slower)
    _run(monkeypatch, x, w, planes=True, cuda=False, tiles={})
    assert calls == [(64, 128, 4, 3), mq._PLANES_DEFAULT_TILE]
    calls.clear()
    # out of scope: switch off, M < 64, activation groups other than 128
    _run(monkeypatch, x, w, planes=False, cuda=False, tiles=tiles)
    _run(monkeypatch, x[:63], w, planes=True, cuda=False, tiles=tiles)
    monkeypatch.setattr(mq, "_MMQ_PLANES", True)
    mq.pq2_mmq(x, w, prepared=True, activation_group=64)
    assert calls == []


@pytest.mark.parametrize("shape", sorted(mq._PLANES_TILES))
def test_table_tiles_bit_exact(monkeypatch, shape):
    n, k = shape
    w = _weight(n, k)
    low = 64
    for m_max, tile, _faster in mq._PLANES_TILES[shape]:
        m = min(m_max, low + 37)  # one M inside each bucket, not a tile multiple
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        expected = _run(monkeypatch, x, w, planes=False, cuda=False)
        assert mq.planes_tile(m, n, k) == (*tile, _faster)
        assert torch.equal(_run(monkeypatch, x, w, planes=True, cuda=False), expected)
        low = m_max + 1


def _gluon_available():
    from vllm_gguf_plugin.triton import pq2_gluon_prefill
    return pq2_gluon_prefill.AVAILABLE


@pytest.mark.parametrize("m", [64, 200, 513, 2048])
@pytest.mark.parametrize("shape", SHAPES)
def test_gluon_prefill_bit_exact(monkeypatch, shape, m):
    if not _gluon_available():
        pytest.skip("Triton Gluon required")
    n, k = shape
    w = _weight(n, k)
    torch.manual_seed(m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    expected = _run(monkeypatch, x, w, planes=False, cuda=False)                       # Triton _mmq
    monkeypatch.setattr(mq, "_MMQ_GLUON", True)
    calls = []
    from vllm_gguf_plugin.triton import pq2_gluon_prefill as gp
    real = gp.pq2_gluon_prefill
    monkeypatch.setattr(gp, "pq2_gluon_prefill", lambda *a: calls.append(a[2]) or real(*a))
    for planes, cuda in ((False, True), (True, True), (True, False)):                  # takes precedence over both
        assert torch.equal(_run(monkeypatch, x, w, planes=planes, cuda=cuda), expected)
    assert calls == [mq.gluon_prefill_tile(m, n, k)] * 3


@pytest.mark.parametrize("shape", sorted(mq._GLUON_TILES))
def test_gluon_prefill_table_tiles_bit_exact(monkeypatch, shape):
    if not _gluon_available():
        pytest.skip("Triton Gluon required")
    n, k = shape
    w = _weight(n, k)
    low = 64
    for m_max, tile in mq._GLUON_TILES[shape]:
        for m in sorted({min(m_max, low + 37), min(m_max, 2048)} if m_max >= 1 << 29 else {min(m_max, low + 37)}):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            expected = _run(monkeypatch, x, w, planes=False, cuda=False)
            assert mq.gluon_prefill_tile(m, n, k) == tile
            monkeypatch.setattr(mq, "_MMQ_GLUON", True)
            assert torch.equal(mq.pq2_mmq(x, w, prepared=True), expected), (shape, m, tile)
            monkeypatch.setattr(mq, "_MMQ_GLUON", False)
        low = m_max + 1


def test_gluon_prefill_dispatch_scope(monkeypatch):
    if not _gluon_available():
        pytest.skip("Triton Gluon required")
    from vllm_gguf_plugin.triton import pq2_gluon_prefill as gp
    n, k = 5120, 6144
    w = _weight(n, k)
    calls = []
    real = gp.pq2_gluon_prefill
    monkeypatch.setattr(gp, "pq2_gluon_prefill", lambda *a: calls.append(1) or real(*a))
    monkeypatch.setattr(mq, "_MMQ_GLUON", True)
    x = torch.randn(128, k, device="cuda", dtype=torch.bfloat16)
    mq.pq2_mmq(x[:63], w, prepared=True)                                  # M < 64
    mq.pq2_mmq(x, w, prepared=True, activation_group=64)                  # other activation groups
    mq.pq2_mmq(x.float(), w, prepared=True)                               # FP32 activations
    assert calls == []
    mq.pq2_mmq(x, w, prepared=True)
    assert calls == [1]
    assert mq.gluon_prefill_tile(128, 4096, 5120) == mq._GLUON_DEFAULT_TILE   # unlisted shape
