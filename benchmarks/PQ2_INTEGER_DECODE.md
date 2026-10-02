# Experimental PQ2 integer decode

`triton/pq2_int_gemv.py::pq2_int_gemv` accepts FP16/BF16/FP32 CUDA activations
with 1–16 tokens and packed PQ2 weights. Each 128-wide activation block is
quantized once using the existing Q8 quantizer; signed DP4A reduces integer
dots within each independently scaled weight block. Activation and FP16 weight
scales are applied before FP32 outer reduction. The fixed tile is 16 rows by
four scale blocks, four warps. Scratch is M*K INT8 bytes plus M*K/128 FP32
scales, excluding output and input-contiguity copies.

## Opt-in serving

Set `GGUF_PQ2_INT_GEMV=1` before launching a fresh server. It is disabled by
default and read once at import. Serving selection requires all of:

- Actual projection M in {1, 2, 4, 5, 8}, including graph padding.
- FP16 or BF16 activation dtype on an SM86 CUDA GPU.
- (N,K) in {(34816,5120), (5120,17408), (16384,5120), (14336,5120), (248320,5120)}.

Other cases retain existing dispatch. Integer selection takes precedence over
`GGUF_PQ2_BATCHED_GEMV=1` on eligible cases. M=16 stays on the original path
because its smallest measured gains are near noise. M=3/6/7 were not benchmarked
and are not selected. Normal warmup/capture compiles the selected variants;
there is no runtime autotuning. Changing flags after import or graph capture
does not change captured arithmetic. Use a fresh server with both flags zero
for the native reference.

## Kernel and numerical evidence (2026-10-01)

RTX 3090 Ti, SM86, physical GPU 1, 275 W, uncontrolled clocks, Torch
2.13.0+cu130 and Triton 3.7.1. Complete-call timings include activation
quantization and use warmed CUDA graph replay. The matrix covers five
projections, BF16/FP16, contiguous/strided weights, and M=1/2/4/5/8/16:
480 records, 120 cases, native/integer/floating-batched/MMQ comparisons.
No measured native regression; native/integer ratios are 1.49–1.70x at M=1
and 4.09–4.90x at M=4. Eighty reversed-order BF16 repeats confirm 1.58–1.77x
and 4.53–4.77x respectively. Synthetic relative RMS against the floating
reference is at most 0.008666. DP4A is present in generated PTX, with zero
reported spills; PTX evidence is not a SASS inspection.

The controlled eager model diagnostic forces eight native 64-token
continuations, two four-request batches, with 512 fixed contexts per mode.
Native repeat logits are exact. Integer probability metrics repeat identically:
mean TV 0.00435528, max TV 0.0533802, mean KL 0.000296661, max KL 0.00769265.
One argmax changes at a native tie. Fifty-four same-activation samples replay
native/integer outputs bitwise exactly and both floating/quantized FP64 references
within 1e-12. BF16 reconstructed-weight rounding differs from preserving FP16
weight scales, so differences include more than activation quantization alone.

These are experimental diagnostics, not task-quality acceptance. The
[serving report](results/pq2-int-serving-20261001/report.md) records fresh matched
concurrency-1/4/8 throughput gains of 1.54x/3.22x/1.75x, graph execution, and
greedy divergences across repeats. Wider serving/task evaluation, comparison
with llama.cpp quality, and an agreed Q8 quality criterion remain needed before
default adoption.

From the containing workspace, matrix reproduction is:

```bash
CUDA_VISIBLE_DEVICES=1 GGUF_PQ2_INT_GEMV=0 GGUF_PQ2_BATCHED_GEMV=0 \
  PYTHONPATH=vllm-gguf-plugin-prism .venv/bin/python \
  vllm-gguf-plugin-prism/benchmarks/benchmark_prism.py \
  --tokens 1 2 4 5 8 16 --dtypes bf16 fp16 --layouts contiguous strided \
  --methods native int_gemv batched_gemv mmq --rep-ms 100 --output matrix.csv
```

The containing workspace's `benchmarks/results/pq2-int-gemv-20261001` and
`pq2-int-numerics-20261001` retain full reports, raw data, source fingerprints,
and replayable captures. Those workspace artifacts are outside this Git repository.

## Packed loading and scheduling investigation

The [packed-layout report](results/pq2-packed-20261001/report.md) compares
raw byte/half loads, aligned code/scale buffers, simpler signed-byte expansion,
software pipelining, cache policy, and warp-lane mappings. The benchmark-only
`pq2_packed_experiment.py` module has no serving dispatch. Preserve the original
weights for existing prefill/fallbacks until those paths support a shared prepared
layout; do not add per-call preparation or silently duplicate resident weights.

## Integrated unpack arithmetic

The signed-byte spread/add/XOR expansion is now integrated into `_int_gemv`.
It preserves the existing fixed tile, Q8 groups, reduction order, and storage.
The [slice report](results/pq2-unpack-20261001/report.md) records 213 tests,
100 bitwise full-shape comparisons, real-model replay, and matched timing evidence.
The unpack-only slice uses raw storage; the subsequent opt-in shared layout is described below.
The [whitepaper review](BONSAI2_WHITEPAPER_REVIEW.md) and the slice's offline
32/128-element activation-group diagnostic explain why Q8 quality validation
remains necessary before default adoption.

## Shared prepared layout

Set `GGUF_PQ2_PREPARED=1` before starting a fresh SM86 server to prepare pure-PQ2 weights once during loading. Pair with `GGUF_PQ2_INT_GEMV=1` to reproduce the measured decode gains; both flags remain off by default. All prepared readers share one allocation containing packed codes followed by FP16 scales, preserving 34 bytes per 128 weights. Embeddings and floating prefill/fallbacks support the same layout. Ties preserve Parameter identity; mixed-type and unsupported-device tensors retain raw storage. Preparation finishes before graph capture and releases original shard references. Prepared APIs require the entire owned contiguous allocation.

vLLM compilation factors include the normalized layout mode and `planes-v1` format version, preventing raw/prepared graph-cache reuse. The [shared-layout report](results/pq2-prepared-20261002/report.md) records 405 consolidated passing tests, 97 final strict layout tests, 96 byte-identical matrix cases, and 108 byte-identical saved outputs. Matched short-prompt serving improves throughput 10.0%/6.6%/5.2% at concurrency 1/4/8, with TTFT rising 5.9%/5.0%/2.8%. Some prefill shapes regress by 3–10%; default promotion is premature. All eight greedy continuations match, but serving logprobs drift across repeats. Packed parameter bytes are identical across modes, with no second resident weight copy.
