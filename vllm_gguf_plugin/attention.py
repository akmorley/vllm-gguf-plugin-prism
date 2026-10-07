# SPDX-License-Identifier: Apache-2.0
"""Opt-in FA2 prefill and Triton single-request decode for the tested SM86 shape."""

import inspect
import os
from functools import wraps

import torch

_SINGLE_ATTN_BACKEND = os.environ.get("GGUF_PQ2_SINGLE_ATTN_BACKEND", "fa2")
_SINGLE_ATTN_SEGMENTS = int(os.environ.get("GGUF_PQ2_SINGLE_ATTN_SEGMENTS", "32"))
_SINGLE_ATTN_VERSION = "fa2-prefill-triton-single-decode-v1"
if _SINGLE_ATTN_BACKEND not in ("fa2", "triton"):
    raise ValueError("GGUF_PQ2_SINGLE_ATTN_BACKEND must be fa2 or triton")
if _SINGLE_ATTN_SEGMENTS not in (32, 64):
    raise ValueError("GGUF_PQ2_SINGLE_ATTN_SEGMENTS must be 32 or 64")


def _supports_single_decode(a):
    q, k, v = a["q"], a["k"], a["v"]
    optional = (
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
    if not (
        a["fa_version"] == 2
        and q.is_cuda
        and q.shape == (1, 24, 256)
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
        and a["max_seqlen_q"] == 1
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
        and all(a[key] is None for key in optional)
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


def install_single_decode_attention():
    """Install once when explicitly enabled; leave the default backend untouched."""
    if _SINGLE_ATTN_BACKEND != "triton":
        return
    import vllm.v1.attention.backends.flash_attn as backend

    current = backend.flash_attn_varlen_func
    installed = getattr(current, "_gguf_single_attention_segments", None)
    if installed is not None:
        if installed != _SINGLE_ATTN_SEGMENTS:
            raise RuntimeError("Restart the server to change attention segments")
        return
    backend.flash_attn_varlen_func = _make_single_decode_dispatch(
        current, _SINGLE_ATTN_SEGMENTS
    )
