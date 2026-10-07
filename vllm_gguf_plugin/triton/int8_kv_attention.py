# SPDX-License-Identifier: Apache-2.0
"""Split-KV decode and verification attention over vLLM's INT8 per-token-head KV cache.

vLLM's ``int8_per_token_head`` layout stores, per (block, slot, KV head), 256 int8 key values,
one fp32 key scale, 256 int8 value values and one fp32 value scale. Its Triton
``unified_attention`` does not split the KV sequence for multi-token queries and loads the int8
rows narrowly, so long-context decode and speculative verification were slower than BF16 FA2.

This kernel follows llama.cpp's quantised-KV vector kernel (``fattn-vec``): Q is quantised to
int8 once per row, Q.K runs on int8 tensor cores with int32 accumulation and is rescaled by the
query and key scales, and V is used as exact BF16 with its per-token scale folded into P.

One request's query tokens (1 for decode, up to 16 for speculative verification) and the query
heads sharing a KV head form the rows of one program, so the causal mask is applied per row and
no prefix/tail decomposition is needed. The KV sequence is split into a fixed number of segments
(CUDA-graph safe); a second kernel merges the segments by log-sum-exp.
"""

import torch
import triton
import triton.language as tl

MAX_QUERY = 16


@triton.jit
def _int8_kv_segment(
    q_ptr, sq0, sq1,
    k_ptr, kb, ks, kh,
    ksc_ptr, vsc_ptr, sb, ss, sh,
    table_ptr, st0,
    cu_ptr, used_ptr,
    acc_ptr, max_ptr, sum_ptr,
    sm_scale,
    GROUP: tl.constexpr, HEADS: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    ROWS: tl.constexpr, TILE: tl.constexpr, SEGMENTS: tl.constexpr, INT_QK: tl.constexpr,
    V_OFFSET: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    segment = tl.program_id(2)
    q_start = tl.load(cu_ptr + seq)
    q_len = tl.load(cu_ptr + seq + 1) - q_start
    seq_len = tl.load(used_ptr + seq)

    rows = tl.arange(0, ROWS)
    token = rows // GROUP
    head = kv_head * GROUP + rows % GROUP
    row_ok = token < q_len
    q_pos = seq_len - q_len + token
    dims = tl.arange(0, D)

    q = tl.load(q_ptr + (q_start + token)[:, None].to(tl.int64) * sq0 + head[:, None] * sq1
                + dims[None, :], mask=row_ok[:, None], other=0.0).to(tl.float32)
    if INT_QK:
        amax = tl.max(tl.abs(q), axis=1)
        q_scale = tl.where(amax > 0, amax / 127.0, 1.0)
        qf = q / q_scale[:, None]
        q_op = tl.where(qf >= 0, qf + 0.5, qf - 0.5).to(tl.int8)
        row_scale = q_scale * sm_scale
    else:
        q_op = q.to(tl.bfloat16)
        row_scale = tl.full((ROWS,), 1.0, tl.float32) * sm_scale

    span = tl.cdiv(tl.cdiv(seq_len, SEGMENTS), TILE) * TILE
    start = segment * span
    end = tl.minimum(start + span, seq_len)

    m = tl.full((ROWS,), -1.0e30, tl.float32)
    l = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    for j in range(start, end, TILE):
        pos = j + tl.arange(0, TILE)
        ok = pos < end
        block = tl.load(table_ptr + seq * st0 + pos // PAGE, mask=ok, other=0).to(tl.int64)
        slot = pos % PAGE
        base = block * kb + slot * ks + kv_head * kh
        base = tl.multiple_of(base, 8)
        k = tl.load(k_ptr + base[None, :] + dims[:, None], mask=ok[None, :], other=0)
        sbase = block * sb + slot * ss + kv_head * sh
        k_scale = tl.load(ksc_ptr + sbase, mask=ok, other=0.0)
        v_scale = tl.load(vsc_ptr + sbase, mask=ok, other=0.0)
        if INT_QK:
            s = tl.dot(q_op, k, out_dtype=tl.int32).to(tl.float32)
        else:
            s = tl.dot(q_op, k.to(tl.bfloat16))
        s = s * row_scale[:, None] * k_scale[None, :]
        visible = ok[None, :] & (pos[None, :] <= q_pos[:, None])
        s = tl.where(visible, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        v = tl.load(k_ptr + V_OFFSET + base[:, None] + dims[None, :], mask=ok[:, None], other=0)
        pv = (p * v_scale[None, :]).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(pv, v.to(tl.bfloat16))
        m = m_new

    out_row = ((q_start + token).to(tl.int64) * HEADS + head) * SEGMENTS + segment
    tl.store(max_ptr + out_row, m, mask=row_ok)
    tl.store(sum_ptr + out_row, l, mask=row_ok)
    tl.store(acc_ptr + out_row[:, None] * D + dims[None, :], acc, mask=row_ok[:, None])


@triton.jit
def _int8_kv_reduce(
    acc_ptr, max_ptr, sum_ptr, out_ptr, so0, so1,
    HEADS: tl.constexpr, D: tl.constexpr, SEGMENTS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    row = (token.to(tl.int64) * HEADS + head) * SEGMENTS
    segs = tl.arange(0, SEGMENTS)
    m = tl.load(max_ptr + row + segs)
    l = tl.load(sum_ptr + row + segs)
    w = tl.exp(m - tl.max(m, axis=0))
    acc = tl.load(acc_ptr + (row + segs)[:, None] * D + tl.arange(0, D)[None, :])
    total = tl.sum(l * w, axis=0)
    # Padded rows (CUDA-graph batch padding) see no keys; emit zeros rather than NaN.
    out = tl.sum(acc * w[:, None], axis=0) / tl.where(total > 0, total, 1.0)
    tl.store(out_ptr + token.to(tl.int64) * so0 + head * so1 + tl.arange(0, D),
             out.to(out_ptr.dtype.element_ty))


# (tile, warps, segments) by padded query rows, from a 3090 sweep at 32K/128K context
# (pq2-int8kv-kernels-20261007). Decode prefers small tiles and many segments; verification rows
# make the kernel partly compute-bound, so it prefers wide tiles and fewer segments.
GEOMETRY = {16: (16, 2, 64), 32: (32, 4, 32), 64: (64, 8, 16), 128: (32, 8, 16)}


def _geometry(max_query, group):
    rows = max(16, triton.next_power_of_2(max_query * group))
    return (rows, *GEOMETRY[rows])


def int8_kv_attention(q, key_cache, value_cache, k_scale_cache, v_scale_cache, block_table,
                      cu_seqlens_q, seqused_k, max_seqlen_q, softmax_scale, out,
                      segments=None, int_qk=False, geometry=None):
    """Causal attention for up to 16 query tokens per request over the INT8 per-token-head cache.

    ``key_cache``/``value_cache``: int8 ``[blocks, page, kv_heads, >= D]`` views (vLLM's
    ``_pth_key_value_caches``); scale caches: fp32 ``[blocks, page, kv_heads]``.
    """
    tokens, heads, dim = q.shape
    kv_heads = key_cache.shape[2]
    group = heads // kv_heads
    seqs = cu_seqlens_q.numel() - 1
    rows, tile, warps, default_segments = geometry or _geometry(max_seqlen_q, group)
    segments = segments or default_segments
    acc = torch.empty((tokens, heads, segments, dim), device=q.device, dtype=torch.float32)
    maximum = torch.empty((tokens, heads, segments), device=q.device, dtype=torch.float32)
    expsum = torch.empty_like(maximum)
    assert key_cache.stride() == value_cache.stride()
    # V is addressed from the K pointer at a constant byte offset so that alignment is provable.
    v_offset = value_cache.data_ptr() - key_cache.data_ptr()
    assert 0 < v_offset < key_cache.stride(2) and key_cache.stride(-1) == 1
    assert k_scale_cache.stride() == v_scale_cache.stride()
    _int8_kv_segment[(seqs, kv_heads, segments)](
        q, q.stride(0), q.stride(1),
        key_cache, key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        k_scale_cache, v_scale_cache,
        k_scale_cache.stride(0), k_scale_cache.stride(1), k_scale_cache.stride(2),
        block_table, block_table.stride(0), cu_seqlens_q, seqused_k,
        acc, maximum, expsum, softmax_scale,
        GROUP=group, HEADS=heads, D=dim, PAGE=key_cache.shape[1],
        ROWS=rows, TILE=tile, SEGMENTS=segments, INT_QK=int_qk, V_OFFSET=v_offset,
        num_warps=warps,
    )
    _int8_kv_reduce[(tokens, heads)](
        acc, maximum, expsum, out, out.stride(0), out.stride(1),
        HEADS=heads, D=dim, SEGMENTS=segments, num_warps=4,
    )
    return out
