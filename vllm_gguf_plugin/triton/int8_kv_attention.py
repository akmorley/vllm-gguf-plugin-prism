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
    start_ptr, end_ptr, row_ptr, used_ptr,
    acc_ptr, max_ptr, sum_ptr,
    sm_scale,
    GROUP: tl.constexpr, HEADS: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    ROWS: tl.constexpr, TILE: tl.constexpr, SEGMENTS: tl.constexpr, INT_QK: tl.constexpr,
    V_OFFSET: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    segment = tl.program_id(2)
    # Request `seq` owns query tokens [start, end) of q/out and compact scratch rows from `row`.
    q_start = tl.load(start_ptr + seq)
    q_len = tl.load(end_ptr + seq) - q_start
    first_row = tl.load(row_ptr + seq)
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

    out_row = ((first_row + token).to(tl.int64) * HEADS + head) * SEGMENTS + segment
    tl.store(max_ptr + out_row, m, mask=row_ok)
    tl.store(sum_ptr + out_row, l, mask=row_ok)
    tl.store(acc_ptr + out_row[:, None] * D + dims[None, :], acc, mask=row_ok[:, None])


@triton.jit
def _int8_kv_reduce(
    acc_ptr, max_ptr, sum_ptr, out_ptr, so0, so1, map_ptr,
    HEADS: tl.constexpr, D: tl.constexpr, SEGMENTS: tl.constexpr, HAS_MAP: tl.constexpr,
):
    compact = tl.program_id(0)
    head = tl.program_id(1)
    row = (compact.to(tl.int64) * HEADS + head) * SEGMENTS
    token = tl.load(map_ptr + compact) if HAS_MAP else compact
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
    ``_pth_key_value_caches``); scale caches: fp32 ``[blocks, page, kv_heads]``. Graph safe: the
    request layout comes from ``cu_seqlens_q`` on the device.
    """
    starts, ends = cu_seqlens_q[:-1], cu_seqlens_q[1:]
    return _launch(q, key_cache, value_cache, k_scale_cache, v_scale_cache, block_table,
                   starts, ends, starts, seqused_k, q.shape[0], None, max_seqlen_q,
                   softmax_scale, out, segments, int_qk, geometry)


def _launch(q, key_cache, value_cache, k_scale_cache, v_scale_cache, block_table, starts, ends,
            first_rows, seqused_k, rows_total, token_map, max_seqlen_q, softmax_scale, out,
            segments=None, int_qk=False, geometry=None):
    heads, dim = q.shape[1], q.shape[2]
    kv_heads = key_cache.shape[2]
    group = heads // kv_heads
    seqs = starts.numel()
    rows, tile, warps, default_segments = geometry or _geometry(max_seqlen_q, group)
    segments = segments or default_segments
    acc = torch.empty((rows_total, heads, segments, dim), device=q.device, dtype=torch.float32)
    maximum = torch.empty((rows_total, heads, segments), device=q.device, dtype=torch.float32)
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
        block_table, block_table.stride(0), starts, ends, first_rows, seqused_k,
        acc, maximum, expsum, softmax_scale,
        GROUP=group, HEADS=heads, D=dim, PAGE=key_cache.shape[1],
        ROWS=rows, TILE=tile, SEGMENTS=segments, INT_QK=int_qk, V_OFFSET=v_offset,
        num_warps=warps,
    )
    _int8_kv_reduce[(rows_total, heads)](
        acc, maximum, expsum, out, out.stride(0), out.stride(1),
        token_map if token_map is not None else acc,
        HEADS=heads, D=dim, SEGMENTS=segments, HAS_MAP=token_map is not None, num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------------------------
# Prefill: dequantise page-aligned key ranges into a BF16 scratch and run FA2 (llama.cpp converts
# quantised K/V to F16 before its tensor-core kernel for batch > 1). Ranges are bounded so the
# scratch stays small at any context length; partial results are merged by log-sum-exp.
# ---------------------------------------------------------------------------------------------


@triton.jit
def _dequant_pages(
    k_ptr, kb, ks, kh, ksc_ptr, vsc_ptr, sb, ss, sh,
    table_ptr, first, key_out, value_out,
    KV_HEADS: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, SLOTS: tl.constexpr,
    V_OFFSET: tl.constexpr,
):
    page = tl.program_id(0)
    chunk = tl.program_id(1)
    head = tl.program_id(2)
    block = tl.load(table_ptr + first + page).to(tl.int64)
    slot = chunk * SLOTS + tl.arange(0, SLOTS)
    ok = slot < PAGE
    dims = tl.arange(0, D)
    base = tl.multiple_of(block * kb + slot * ks + head * kh, 8)
    sbase = block * sb + slot * ss + head * sh
    k = tl.load(k_ptr + base[:, None] + dims[None, :], mask=ok[:, None], other=0).to(tl.float32)
    v = tl.load(k_ptr + V_OFFSET + base[:, None] + dims[None, :], mask=ok[:, None], other=0)
    k = k * tl.load(ksc_ptr + sbase, mask=ok, other=0.0)[:, None]
    v = v.to(tl.float32) * tl.load(vsc_ptr + sbase, mask=ok, other=0.0)[:, None]
    dst = ((page.to(tl.int64) * PAGE + slot) * KV_HEADS + head)[:, None] * D + dims[None, :]
    tl.store(key_out + dst, k.to(key_out.dtype.element_ty), mask=ok[:, None])
    tl.store(value_out + dst, v.to(value_out.dtype.element_ty), mask=ok[:, None])


def dequantize_pages(key_cache, value_cache, k_scale_cache, v_scale_cache, table_row, first,
                     count, dim, dtype):
    """BF16 ``[count, page, kv_heads, D]`` K and V for logical pages ``first .. first+count-1``."""
    blocks, page, kv_heads, _ = key_cache.shape
    keys = torch.empty((count, page, kv_heads, dim), device=key_cache.device, dtype=dtype)
    values = torch.empty_like(keys)
    slots = 32
    _dequant_pages[(count, triton.cdiv(page, slots), kv_heads)](
        key_cache, key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        k_scale_cache, v_scale_cache,
        k_scale_cache.stride(0), k_scale_cache.stride(1), k_scale_cache.stride(2),
        table_row, first, keys, values,
        KV_HEADS=kv_heads, D=dim, PAGE=page, SLOTS=slots,
        V_OFFSET=value_cache.data_ptr() - key_cache.data_ptr(), num_warps=4,
    )
    return keys, values


def int8_kv_prefill_attention(q, key_cache, value_cache, k_scale_cache, v_scale_cache,
                              block_table, seqused_k, max_seqlen_k, softmax_scale, out,
                              range_tokens=32768):
    """Causal attention for one request's prefill chunk (all of ``q``) over the INT8 cache.

    The chunk's queries are the last ``q.shape[0]`` positions of a sequence of ``max_seqlen_k``
    keys. Keys before the chunk are processed in page-aligned ranges without a mask; the final
    range (holding the chunk) uses FA2's bottom-right causal alignment.
    """
    n, heads, dim = q.shape
    length = max_seqlen_k
    page = key_cache.shape[1]
    span = max(page, range_tokens // page * page)
    boundary = (length - n) // page * page  # keys [0, boundary) are visible to every query
    ranges = [(a, min(a + span, boundary), False) for a in range(0, boundary, span)]
    ranges.append((boundary, length, True))
    cu = torch.tensor([0, n], device=q.device, dtype=torch.int32)
    row = block_table[0]
    merged = lse = None
    for start, end, causal in ranges:
        count = triton.cdiv(end, page) - start // page
        keys, values = dequantize_pages(key_cache, value_cache, k_scale_cache, v_scale_cache,
                                        row, start // page, count, dim, q.dtype)
        table = torch.arange(count, device=q.device, dtype=torch.int32)[None]
        used = torch.full((1,), end - start, device=q.device, dtype=torch.int32)
        part, part_lse = torch.ops._vllm_fa2_C.varlen_fwd(
            q, keys, values, None, cu, torch.zeros_like(cu), used, None, table, None,
            n, end - start, 0.0, softmax_scale, False, causal, -1, -1, 0.0, False, 0, None,
        )[:2]
        part_lse = part_lse.t()  # [n, heads]
        if merged is None:
            merged, lse = part.float(), part_lse
            continue
        top = torch.maximum(lse, part_lse)
        w1, w2 = torch.exp(lse - top), torch.exp(part_lse - top)
        merged = (merged * w1[..., None] + part.float() * w2[..., None]) / (w1 + w2)[..., None]
        lse = top + torch.log(w1 + w2)
    out.copy_(merged)
    return out


def int8_kv_mixed_attention(q, key_cache, value_cache, k_scale_cache, v_scale_cache, block_table,
                            cu_seqlens_q, seqused_k, softmax_scale, out, range_tokens=32768,
                            segments=None, int_qk=False):
    """Batches mixing prefill chunks with decode/verification tokens (not graph safe: reads the
    request layout on the host). Requests with <= MAX_QUERY tokens run the split-KV kernel
    together; each prefill chunk runs the dequantised-range FA2 path."""
    cu = cu_seqlens_q.tolist()
    lengths = seqused_k.tolist()
    short, prefill = [], []
    for s in range(len(cu) - 1):
        n = cu[s + 1] - cu[s]
        if 0 < n <= MAX_QUERY:
            short.append(s)
        elif n > MAX_QUERY:
            prefill.append(s)
    if short:
        device = q.device
        starts = [cu[s] for s in short]
        counts = [cu[s + 1] - cu[s] for s in short]
        first_rows = [sum(counts[:i]) for i in range(len(short))]
        token_map = [t for st, c in zip(starts, counts) for t in range(st, st + c)]
        to = lambda v: torch.tensor(v, device=device, dtype=torch.int32)
        index = to(short).long()
        _launch(q, key_cache, value_cache, k_scale_cache, v_scale_cache,
                block_table.index_select(0, index), to(starts), to([a + c for a, c in zip(starts, counts)]),
                to(first_rows), seqused_k.index_select(0, index), len(token_map), to(token_map),
                max(counts), softmax_scale, out, segments, int_qk)
    for s in prefill:
        lo, hi = cu[s], cu[s + 1]
        int8_kv_prefill_attention(q[lo:hi], key_cache, value_cache, k_scale_cache, v_scale_cache,
                                  block_table[s:s + 1], seqused_k[s:s + 1], lengths[s],
                                  softmax_scale, out[lo:hi], range_tokens=range_tokens)
    return out
