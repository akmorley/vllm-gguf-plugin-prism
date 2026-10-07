"""INT8 per-token-head KV attention: accuracy vs a dequantised reference, batching, verification."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="SM80+ CUDA required",
)
PAGE, HEADS, KV_HEADS, DIM = 1552, 24, 4, 256
SCALE_PAD = 4  # one fp32 scale after each head's K and V data, as in vLLM's layout


def _cache(pages, seed):
    """vLLM int8_per_token_head layout: logical [blocks, page, kv_heads, 2 * (D + 4)] bytes."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    storage = torch.randint(-127, 128, (pages, PAGE, KV_HEADS, 2 * (DIM + SCALE_PAD)),
                            device="cuda", dtype=torch.int8, generator=g)
    half = DIM + SCALE_PAD
    key, value = storage.split(half, dim=-1)
    f32 = torch.tensor([], dtype=torch.float32, device="cuda").set_(storage.untyped_storage())
    stride = (storage.stride(0) // 4, storage.stride(1) // 4, storage.stride(2) // 4)
    k_scale = torch.as_strided(f32, (pages, PAGE, KV_HEADS), stride, DIM // 4)
    v_scale = torch.as_strided(f32, (pages, PAGE, KV_HEADS), stride, (half + DIM) // 4)
    k_scale.copy_(torch.rand(pages, PAGE, KV_HEADS, device="cuda", generator=g) * 0.04 + 0.01)
    v_scale.copy_(torch.rand(pages, PAGE, KV_HEADS, device="cuda", generator=g) * 0.04 + 0.01)
    return key, value, k_scale, v_scale


def _inputs(used, rows, seed=0):
    """`used` and `rows` are per-request lists."""
    seqs = len(used)
    pages_per = [math.ceil(u / PAGE) for u in used]
    pages = sum(pages_per) + 2
    key, value, k_scale, v_scale = _cache(pages, seed)
    order = torch.randperm(pages, device="cuda", dtype=torch.int32)
    width = max(pages_per)
    table = torch.zeros(seqs, width, device="cuda", dtype=torch.int32)
    taken = 0
    for i, n in enumerate(pages_per):
        table[i, :n] = order[taken:taken + n]
        taken += n
    torch.manual_seed(seed)
    cu = torch.tensor([0] + list(torch.tensor(rows).cumsum(0)), device="cuda", dtype=torch.int32)
    q = torch.randn(sum(rows), HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    return dict(q=q, key_cache=key, value_cache=value, k_scale_cache=k_scale,
                v_scale_cache=v_scale, block_table=table, cu_seqlens_q=cu,
                seqused_k=torch.tensor(used, device="cuda", dtype=torch.int32),
                max_seqlen_q=max(rows), softmax_scale=DIM**-0.5)


def _reference(a):
    outs = []
    group = HEADS // KV_HEADS
    for s in range(a["block_table"].shape[0]):
        lo, hi = int(a["cu_seqlens_q"][s]), int(a["cu_seqlens_q"][s + 1])
        used, rows = int(a["seqused_k"][s]), hi - lo
        pos = torch.arange(used, device="cuda")
        blocks = a["block_table"][s, pos // PAGE].long()
        slots = pos % PAGE
        keys = a["key_cache"][blocks, slots, :, :DIM].float() * a["k_scale_cache"][blocks, slots][..., None]
        vals = a["value_cache"][blocks, slots, :, :DIM].float() * a["v_scale_cache"][blocks, slots][..., None]
        q = a["q"][lo:hi].float().reshape(rows, KV_HEADS, group, DIM)
        scores = torch.einsum("qhgd,khd->hgqk", q, keys) * a["softmax_scale"]
        limit = used - rows + torch.arange(rows, device="cuda")
        scores = scores.masked_fill(pos[None, :] > limit[:, None], float("-inf"))
        outs.append(torch.einsum("hgqk,khd->qhgd", scores.softmax(-1), vals).reshape(rows, HEADS, DIM))
    return torch.cat(outs)


def _run(a, **kw):
    from vllm_gguf_plugin.triton.int8_kv_attention import int8_kv_attention

    out = torch.empty_like(a["q"])
    return int8_kv_attention(**a, out=out, **kw)


def _rel(x, y):
    return ((x.float() - y).norm() / y.norm()).item()


@pytest.mark.parametrize("used", [1, 7, 63, 1552, 1553, 5000, 40000])
def test_decode_matches_reference(used):
    a = _inputs([used], [1])
    ref = _reference(a)
    assert _rel(_run(a, int_qk=False), ref) < 1e-2
    # Q quantised to int8 per row (llama.cpp q8_1 style) costs a little more.
    assert _rel(_run(a, int_qk=True), ref) < 2e-2


@pytest.mark.parametrize("rows", [2, 3, 6, 8, 16])
@pytest.mark.parametrize("used", [16, 100, 3000, 20000])
def test_verify_is_causal_and_matches_reference(rows, used):
    a = _inputs([used], [rows], seed=rows)
    ref = _reference(a)
    assert _rel(_run(a, int_qk=False), ref) < 1e-2
    assert _rel(_run(a, int_qk=True), ref) < 2e-2


def test_batched_requests_with_mixed_lengths():
    a = _inputs([5, 1600, 9000, 300], [1, 6, 1, 6], seed=3)
    ref = _reference(a)
    assert _rel(_run(a), ref) < 2e-2


@pytest.mark.parametrize("segments", [8, 32, 64])
def test_segment_counts_agree(segments):
    a = _inputs([12345], [6], seed=5)
    ref = _reference(a)
    assert _rel(_run(a, segments=segments), ref) < 2e-2


def test_cuda_graph_replay_tracks_lengths():
    a = _inputs([30000], [6], seed=7)
    out = torch.empty_like(a["q"])
    from vllm_gguf_plugin.triton.int8_kv_attention import int8_kv_attention

    int8_kv_attention(**a, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        int8_kv_attention(**a, out=out)
    for used in (29000, 30000, 6):
        a["seqused_k"].fill_(used)
        graph.replay()
        assert _rel(out, _reference(a)) < 2e-2


def _backend_call(a, **overrides):
    """Keyword arguments as TritonAttentionImpl.forward passes them to unified_attention."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    call = dict(q=a["q"], k=a["key_cache"], v=a["value_cache"], out=torch.empty_like(a["q"]),
                cu_seqlens_q=a["cu_seqlens_q"], max_seqlen_q=a["max_seqlen_q"],
                seqused_k=a["seqused_k"], max_seqlen_k=int(a["seqused_k"].max()),
                softmax_scale=a["softmax_scale"], causal=True, alibi_slopes=None,
                use_alibi_sqrt=False, window_size=(-1, -1), block_table=a["block_table"], softcap=0,
                q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=None,
                num_par_softmax_segments=None, softmax_segm_output=None, softmax_segm_max=None,
                softmax_segm_expsum=None, sinks=None, output_scale=None, mm_prefix_range=None,
                rswa_prefix_lens=None, rswa_window=None,
                kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, k_scale_cache=a["k_scale_cache"],
                v_scale_cache=a["v_scale_cache"], chunk_lookback=-1, use_td=False,
                mm_prefix_clamp_sliding_window=False)
    call.update(overrides)
    return call


def test_dispatch_routes_eligible_calls_and_falls_through(monkeypatch):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode
    from vllm_gguf_plugin import attention

    calls = []

    def original(*args, **kwargs):
        calls.append("original")

    original.__signature__ = __import__("inspect").signature(unified_attention)
    dispatch = attention._make_int8_kv_dispatch(original)
    before = attention._INT8_KV_CALLS
    a = _inputs([3000, 40], [6, 1], seed=11)
    call = _backend_call(a)
    dispatch(**call)
    assert calls == [] and attention._INT8_KV_CALLS == before + 1
    assert _rel(call["out"], _reference(a)) < 1e-2
    # Prefill-sized queries, windows, other KV modes and non-causal calls are not ours.
    for overrides in ({"max_seqlen_q": 17}, {"window_size": (2047, 0)}, {"causal": False},
                      {"kv_quant_mode": KVQuantMode.FP8_PER_TOKEN_HEAD}, {"softcap": 30.0}):
        dispatch(**_backend_call(a, **overrides))
    assert calls == ["original"] * 5


def test_install_is_idempotent_and_switchable(monkeypatch):
    from vllm_gguf_plugin import attention
    import vllm.v1.attention.backends.triton_attn as backend

    original = backend.unified_attention
    while hasattr(original, "__wrapped__"):
        original = original.__wrapped__
    monkeypatch.setattr(backend, "unified_attention", original)
    monkeypatch.setattr(attention, "_INT8_KV_ATTN", False)
    attention.install_int8_kv_attention()
    assert backend.unified_attention is original
    monkeypatch.setattr(attention, "_INT8_KV_ATTN", True)
    attention.install_int8_kv_attention()
    installed = backend.unified_attention
    attention.install_int8_kv_attention()
    assert backend.unified_attention is installed and installed._gguf_int8_kv_attention


def test_compile_factors_registered(monkeypatch):
    import vllm.envs as envs
    from vllm_gguf_plugin import attention
    from vllm_gguf_plugin.plugin import _register_pq2_compile_factors

    _register_pq2_compile_factors()
    monkeypatch.setattr(attention, "_INT8_KV_INT_QK", True)
    assert envs.environment_variables["GGUF_PQ2_INT8_KV_INT_QK"]() is True
    assert envs.environment_variables["GGUF_PQ2_INT8_KV_VERSION"]() == attention._INT8_KV_VERSION
