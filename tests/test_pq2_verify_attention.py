"""Split-KV verification attention: exactness, empty prefix, graph replay and dispatch scope."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6),
    reason="SM86 CUDA required",
)
PAGE, HEADS, KV_HEADS, DIM = 784, 24, 4, 256


def _inputs(used, rows, seed=0):
    import vllm.vllm_flash_attn  # noqa: F401  (registers the FA2 op)

    torch.manual_seed(seed)
    pages = math.ceil((used + 64) / PAGE) + 2
    k = torch.randn(pages, PAGE, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return {
        "q": torch.randn(rows, HEADS, DIM, device="cuda", dtype=torch.bfloat16),
        "k": k, "v": v,
        "block_table": torch.randperm(pages, device="cuda", dtype=torch.int32)[None, :],
        "seqused_k": torch.tensor([used], device="cuda", dtype=torch.int32),
        "cu_seqlens_q": torch.tensor([0, rows], device="cuda", dtype=torch.int32),
        "max_seqlen_q": rows, "max_seqlen_k": 34816, "softmax_scale": DIM**-0.5, "out": None,
    }


def _reference(a):
    q, k, v, used = a["q"], a["k"], a["v"], int(a["seqused_k"][0])
    rows, group = q.shape[0], HEADS // KV_HEADS
    keys = k[a["block_table"][0].long()].reshape(-1, KV_HEADS, DIM)[:used].float()
    values = v[a["block_table"][0].long()].reshape(-1, KV_HEADS, DIM)[:used].float()
    scores = torch.einsum("qhgd,khd->hgqk", q.float().reshape(rows, KV_HEADS, group, DIM), keys)
    limit = used - rows + torch.arange(rows, device="cuda")
    scores = scores * a["softmax_scale"]
    scores = scores.masked_fill(torch.arange(used, device="cuda")[None, :] > limit[:, None], float("-inf"))
    return torch.einsum("hgqk,khd->qhgd", scores.softmax(-1), values).reshape(rows, HEADS, DIM)


def _fa2(a):
    return torch.ops._vllm_fa2_C.varlen_fwd(
        a["q"], a["k"], a["v"], None, a["cu_seqlens_q"], torch.zeros_like(a["cu_seqlens_q"]),
        a["seqused_k"], None, a["block_table"], None, a["max_seqlen_q"], a["max_seqlen_k"],
        0.0, a["softmax_scale"], False, True, -1, -1, 0.0, False, 0, None,
    )[0]


def _rel(x, y):
    return float((x.float() - y.float()).norm() / y.float().norm())


@pytest.mark.parametrize("rows", [2, 6, 8, 16])
@pytest.mark.parametrize("used", [None, 300, 5000, 32768])
def test_matches_reference_and_fa2(rows, used):
    from vllm_gguf_plugin import attention

    a = _inputs(rows if used is None else used, rows)  # None: empty prefix (prompt == query)
    assert attention._supports_verify(a | {"fa_version": 2, **_defaults()})
    actual = attention._verify_attention(a)
    assert torch.isfinite(actual).all()
    assert _rel(actual, _reference(a)) < 5e-3
    assert _rel(actual, _fa2(a)) < 5e-3


def _defaults():
    return {"dropout_p": 0.0, "softcap": 0.0, "causal": True, "cp_world_size": 1,
            "return_softmax_lse": False, "return_attn_probs": False, "window_size": [-1, -1],
            "cu_seqlens_k": None, **{key: None for key in __import__(
                "vllm_gguf_plugin.attention", fromlist=["_OPTIONAL"])._OPTIONAL}}


def test_graph_replay_follows_inputs():
    from vllm_gguf_plugin import attention

    a = _inputs(9000, 6)
    for _ in range(2):
        attention._verify_attention(a)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = attention._verify_attention(a)
    a["q"].mul_(0.5).add_(0.1)
    a["seqused_k"].fill_(7000)
    graph.replay()
    assert _rel(captured, _reference(a)) < 5e-3


def test_dispatch_scope(monkeypatch):
    from vllm_gguf_plugin import attention

    import inspect

    import vllm.v1.attention.backends.flash_attn as backend

    calls = []

    def original(*args, **kwargs):
        calls.append("original")

    original.__signature__ = inspect.signature(backend.flash_attn_varlen_func)
    monkeypatch.setattr(attention, "_verify_attention", lambda a: calls.append("verify") or a["q"])
    dispatch = attention._make_verify_dispatch(original)
    for rows, used in ((6, 5000), (1, 5000), (17, 5000)):
        a = _inputs(used, rows)
        dispatch(a["q"], a["k"], a["v"], rows, a["cu_seqlens_q"], 34816, seqused_k=a["seqused_k"],
                 softmax_scale=a["softmax_scale"], causal=True, window_size=[-1, -1],
                 block_table=a["block_table"], fa_version=2)
    assert calls == ["verify", "original", "original"]
    a = _inputs(5000, 6)
    dispatch(a["q"], a["k"], a["v"], 6, torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32),
             34816, seqused_k=torch.tensor([5000, 5000], device="cuda", dtype=torch.int32),
             causal=True, window_size=[-1, -1], block_table=a["block_table"].expand(2, -1), fa_version=2)
    assert calls[-1] == "original"


def test_compile_identity(monkeypatch):
    import vllm.envs as envs

    from vllm_gguf_plugin import attention
    from vllm_gguf_plugin.plugin import _register_pq2_compile_factors

    _register_pq2_compile_factors()
    for enabled in (False, True):
        monkeypatch.setattr(attention, "_VERIFY_ATTN", enabled)
        assert envs.environment_variables["GGUF_PQ2_VERIFY_ATTN"]() is enabled
    assert envs.environment_variables["GGUF_PQ2_VERIFY_ATTN_VERSION"]() == attention._VERIFY_ATTN_VERSION
