# SPDX-License-Identifier: Apache-2.0
"""Attention overrides for the tested SM86 shape (24 query heads, 4 KV heads, dim 256).

- GGUF_PQ2_SINGLE_ATTN_BACKEND=triton (opt-in): FA2 prefill, Triton segmented single-token decode.
- GGUF_PQ2_VERIFY_ATTN (default on; =0 disables): split-KV attention for short multi-token queries of one request
  (speculative verification). FA2 varlen refuses split-KV for paged KV when seqlen_q > 1, so a
  causal query of n <= 16 tokens is decomposed exactly: the shared prefix (all but the last n
  keys) runs as FA2's single-token split-KV path with the n tokens packed into the head
  dimension; the last n keys are attended directly with the causal mask; the two partial
  results are merged by their log-sum-exp.
- GGUF_PQ2_INT8_KV_ATTN (default on; =0 disables): with --kv-cache-dtype int8_per_token_head
  (Triton backend), decode and speculative verification (<= 16 query tokens per request) run the
  split-KV INT8 kernel in triton/int8_kv_attention.py instead of vLLM's unified_attention.
  GGUF_PQ2_INT8_KV_INT_QK=1 quantises Q to int8 for Q.K (llama.cpp style; faster, less exact).
"""

import inspect
import os
from functools import wraps

import torch

_SINGLE_ATTN_BACKEND = os.environ.get("GGUF_PQ2_SINGLE_ATTN_BACKEND", "fa2")
_SINGLE_ATTN_SEGMENTS = int(os.environ.get("GGUF_PQ2_SINGLE_ATTN_SEGMENTS", "32"))
_SINGLE_ATTN_VERSION = "fa2-prefill-triton-single-decode-v1"
_VERIFY_ATTN = os.environ.get("GGUF_PQ2_VERIFY_ATTN", "1") == "1"
_VERIFY_ATTN_VERSION = "fa2-split-prefix-direct-tail-v1"
_VERIFY_MAX_Q = 16
_VERIFY_CALLS = 0  # Python-side calls (warmup and graph capture); replay does not re-enter.
_INT8_KV_ATTN = os.environ.get("GGUF_PQ2_INT8_KV_ATTN", "1") == "1"
_INT8_KV_INT_QK = os.environ.get("GGUF_PQ2_INT8_KV_INT_QK", "0") == "1"
_INT8_KV_SEGMENTS = int(os.environ.get("GGUF_PQ2_INT8_KV_SEGMENTS", "0"))  # 0: per geometry
_INT8_KV_VERSION = "split-kv-int8-v2-geometry"
_INT8_KV_CALLS = 0
if _SINGLE_ATTN_BACKEND not in ("fa2", "triton"):
    raise ValueError("GGUF_PQ2_SINGLE_ATTN_BACKEND must be fa2 or triton")
if _SINGLE_ATTN_SEGMENTS not in (32, 64):
    raise ValueError("GGUF_PQ2_SINGLE_ATTN_SEGMENTS must be 32 or 64")


_OPTIONAL = (
        "q_v",
        "alibi_slopes",
        "scheduler_metadata",
        "output_scale",
        "s_aux",
        "cp_tot_seqused_k",
        "mask_mod",
        "block_sparse_tensors",
        "aux_tensors",
        "aux_tensor_leading_dims",
        "dynamic_causal",
)


def _supports_paged_fa2(a, rows):
    """Shared checks for the tested paged FA2 shape with `rows` query tokens of one request."""
    q, k, v = a["q"], a["k"], a["v"]
    if not (
        a["fa_version"] == 2
        and q.is_cuda
        and q.shape == (rows, 24, 256)
        and k.ndim == 4
        and k.shape[-2:] == (4, 256)
        and v.shape == k.shape
        and q.dtype == torch.bfloat16
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.device == k.device == v.device
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and k.shape[1] >= 16
        and k.shape[1] % 16 == 0
        and a["max_seqlen_q"] == rows
        and a["cu_seqlens_q"].numel() == 2
        and a["seqused_k"] is not None
        and a["seqused_k"].numel() == 1
        and a["block_table"] is not None
        and a["block_table"].shape[0] == 1
        and a["dropout_p"] == 0
        and a["softcap"] == 0
        and a["causal"] is True
        and a["cp_world_size"] == 1
        and not a["return_softmax_lse"]
        and not a["return_attn_probs"]
        and a["window_size"] in (None, [-1, -1], (-1, -1))
        and all(a[key] is None for key in _OPTIONAL)
    ):
        return False
    for name in ("cu_seqlens_q", "seqused_k", "block_table"):
        metadata = a[name]
        if metadata.device != q.device or metadata.dtype != torch.int32:
            return False
    if a["cu_seqlens_q"].ndim != 1 or a["seqused_k"].ndim != 1:
        return False
    if a["block_table"].ndim != 2 or a["block_table"].stride(-1) != 1:
        return False
    out = a["out"]
    if out is not None and (
        out.shape != q.shape
        or out.dtype != q.dtype
        or out.device != q.device
        or out.stride(-1) != 1
    ):
        return False
    return torch.cuda.get_device_capability(q.device) == (8, 6)


def _supports_single_decode(a):
    return _supports_paged_fa2(a, 1)


def _supports_verify(a):
    rows = a["q"].shape[0] if a["q"].ndim == 3 else 0
    return 2 <= rows <= _VERIFY_MAX_Q and _supports_paged_fa2(a, rows)


def _verify_attention(a):
    """Exact causal attention for the last n query tokens of one request (see module doc)."""
    q, k, v = a["q"], a["k"], a["v"]
    n, heads, dim = q.shape
    kv_heads = k.shape[2]
    group = heads // kv_heads
    scale = a["softmax_scale"] if a["softmax_scale"] is not None else dim**-0.5
    index = torch.arange(n, device=q.device, dtype=torch.int32)
    prefix_used = a["seqused_k"] - n
    # Prefix: one "token" whose heads are (head, position) pairs, head-major so that packed
    # head j maps to KV head j // (group * n) == head // group, as FA2 expects.
    packed = q.permute(1, 0, 2).reshape(1, heads * n, dim)
    cu = index[:2].clamp(max=1)
    out1, lse1 = torch.ops._vllm_fa2_C.varlen_fwd(
        packed, k, v, None, cu, torch.zeros_like(cu), prefix_used, None, a["block_table"],
        None, 1, a["max_seqlen_k"], 0.0, scale, False, True, -1, -1, 0.0, False, 0, None,
    )[:2]
    # An empty prefix (prompt of <= n tokens) yields an undefined LSE; it must contribute nothing.
    empty = prefix_used <= 0
    out1 = out1.reshape(heads, n, dim).transpose(0, 1).float().masked_fill(empty, 0.0)
    lse1 = lse1.reshape(heads, n).t().masked_fill(empty, float("-inf"))
    # Tail: the last n keys (positions prefix_used .. prefix_used + n - 1), causal.
    position = prefix_used + index
    page = k.shape[1]
    blocks = a["block_table"][0, (position // page).long()].long()
    offsets = (position % page).long()
    keys, values = k[blocks, offsets].float(), v[blocks, offsets].float()
    scores = torch.einsum("qhgd,khd->qhgk", q.float().reshape(n, kv_heads, group, dim), keys)
    scores = scores * scale
    scores = scores.masked_fill((index[None, :] > index[:, None])[:, None, None, :], float("-inf"))
    lse2 = torch.logsumexp(scores, -1).reshape(n, heads)
    out2 = torch.einsum("qhgk,khd->qhgd", scores.softmax(-1), values).reshape(n, heads, dim)
    # Merge by log-sum-exp. The tail always contains the query's own key, so lse2 is finite.
    top = torch.maximum(lse1, lse2)
    w1, w2 = torch.exp(lse1 - top), torch.exp(lse2 - top)
    merged = (out1 * w1[..., None] + out2 * w2[..., None]) / (w1 + w2)[..., None]
    out = a["out"] if a["out"] is not None else torch.empty_like(q)
    out.copy_(merged)
    return out


def _make_verify_dispatch(original):
    signature = inspect.signature(original)

    @wraps(original)
    def dispatch(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not _supports_verify(bound.arguments):
            return original(*args, **kwargs)
        global _VERIFY_CALLS
        _VERIFY_CALLS += 1
        return _verify_attention(bound.arguments)

    dispatch._gguf_verify_attention = True
    return dispatch


def _make_single_decode_dispatch(original, segments):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    signature = inspect.signature(original)

    @wraps(original)
    def dispatch(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if not _supports_single_decode(a):
            return original(*args, **kwargs)
        q = a["q"]
        # Capture retains these scratch allocations for replay.
        partial = torch.empty(
            (1, 24, segments, 256), device=q.device, dtype=torch.float32
        )
        maximum = torch.empty((1, 24, segments), device=q.device, dtype=torch.float32)
        expsum = torch.empty_like(maximum)
        out = a["out"] if a["out"] is not None else torch.empty_like(q)
        scale = a["softmax_scale"]
        unified_attention(
            q,
            a["k"],
            a["v"],
            out,
            a["cu_seqlens_q"],
            1,
            a["seqused_k"],
            a["max_seqlen_k"],
            scale if scale is not None else 256**-0.5,
            True,
            (-1, -1),
            a["block_table"],
            0.0,
            None,
            None,
            None,
            seq_threshold_3D=1,
            num_par_softmax_segments=segments,
            softmax_segm_output=partial,
            softmax_segm_max=maximum,
            softmax_segm_expsum=expsum,
        )
        return out

    dispatch._gguf_single_attention_segments = segments
    return dispatch


def _supports_int8_kv(a):
    from vllm.v1.kv_cache_interface import KVQuantMode

    q, k = a["q"], a["k"]
    if a["kv_quant_mode"] != KVQuantMode.INT8_PER_TOKEN_HEAD:
        return False
    if not (q.is_cuda and q.ndim == 3 and q.dtype in (torch.bfloat16, torch.float16)):
        return False
    heads, dim = q.shape[1], q.shape[2]
    kv_heads = k.shape[2] if k.ndim == 4 else 0
    if not (
        k.dtype == torch.int8
        and k.stride(-1) == 1
        and dim in (64, 128, 256)
        and kv_heads
        and heads % kv_heads == 0
        and k.shape[-1] >= dim
        and isinstance(a["max_seqlen_q"], int)
        and 1 <= a["max_seqlen_q"] <= 16
        and a["max_seqlen_q"] * heads // kv_heads <= 128
        and a["causal"] is True
        and a["window_size"] in (None, [-1, -1], (-1, -1))
        and not a["softcap"]
        and a["alibi_slopes"] is None
        and a["sinks"] is None
        and a["output_scale"] is None
        and a["q_descale"] is None
        and a["qq_bias"] is None
        and a["mm_prefix_range"] is None
        and a["rswa_prefix_lens"] is None
        and (a["chunk_lookback"] is None or a["chunk_lookback"] < 0)
        and not a["use_td"]
        and a["k_scale_cache"] is not None
        and a["v_scale_cache"] is not None
        and a["block_table"].shape[0] == a["cu_seqlens_q"].numel() - 1
    ):
        return False
    return torch.cuda.get_device_capability(q.device) >= (8, 0)


def _make_int8_kv_dispatch(original):
    signature = inspect.signature(original)

    @wraps(original)
    def dispatch(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if not _supports_int8_kv(a):
            return original(*args, **kwargs)
        from .triton.int8_kv_attention import int8_kv_attention

        global _INT8_KV_CALLS
        _INT8_KV_CALLS += 1
        return int8_kv_attention(
            a["q"], a["k"], a["v"], a["k_scale_cache"], a["v_scale_cache"], a["block_table"],
            a["cu_seqlens_q"], a["seqused_k"], a["max_seqlen_q"], a["softmax_scale"], a["out"],
            segments=_INT8_KV_SEGMENTS or None, int_qk=_INT8_KV_INT_QK,
        )

    dispatch._gguf_int8_kv_attention = True
    return dispatch


def install_int8_kv_attention():
    """Route eligible INT8-KV calls of the Triton backend to the split-KV kernel."""
    if not _INT8_KV_ATTN:
        return
    import vllm.v1.attention.backends.triton_attn as backend

    if not getattr(backend.unified_attention, "_gguf_int8_kv_attention", False):
        backend.unified_attention = _make_int8_kv_dispatch(backend.unified_attention)


def install_single_decode_attention():
    """Install the enabled overrides once; unmatched calls fall through unchanged."""
    install_int8_kv_attention()
    if _SINGLE_ATTN_BACKEND != "triton" and not _VERIFY_ATTN:
        return
    import vllm.v1.attention.backends.flash_attn as backend

    current = backend.flash_attn_varlen_func
    if _VERIFY_ATTN and not getattr(current, "_gguf_verify_attention", False):
        current = _make_verify_dispatch(current)
    if _SINGLE_ATTN_BACKEND == "triton":
        installed = getattr(current, "_gguf_single_attention_segments", None)
        if installed is not None and installed != _SINGLE_ATTN_SEGMENTS:
            raise RuntimeError("Restart the server to change attention segments")
        if installed is None:
            current = _make_single_decode_dispatch(current, _SINGLE_ATTN_SEGMENTS)
    backend.flash_attn_varlen_func = current
