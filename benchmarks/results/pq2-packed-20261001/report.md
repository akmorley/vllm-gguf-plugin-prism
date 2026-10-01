# Packed PQ2 loading, unpacking, and scheduling investigation — 2026-10-01

The strongest measured candidate separates aligned code/scale buffers, retains bytewise dot lanes, and simplifies signed-byte expansion. Across 24 dtype/stride cases it is **1.098–1.326x faster than the exact original integer kernel at the already-tuned tiles**, including activation quantization. Simpler unpacking without preparation improves **1.025–1.140x**. This investigation leaves inference kernels and serving dispatch unchanged; these are kernel gains, not measured additional serving gains.

## Experiments and matched measurements

Physical GPU 1, RTX 3090 Ti, SM86, 275 W, uncontrolled clocks; Torch 2.13.0+cu130, Triton 3.7.1. The previous tile investigation was committed as c153321 before this work. GPU 1 returned to 1 MiB used afterwards. No benchmark command sent stop requests or signals to the existing GPU 0/2 services. Their earlier 20956/22798 MiB allocations were still present after model replay, but the final check found all three devices at 1 MiB and no services on ports 8097/8098/8099. The exit cause of the earlier services was not established.

The first sweep covers 13 loading/unpacking/scheduling strategies on FFN gate/up (34816,5120) and FFN down (5120,17408), M=1/4/8: 78 variants with forward and reverse full-call timings. It measures prequantized timing separately, complete calls including quantization and scratch allocation, compiler registers/spills/shared memory, and SM86 SASS at M=4. Separate weight preparation is outside full-call timing because it must be reused, not repeated during inference.

A confirmation uses an independent random seed, BF16/FP16, contiguous/strided raw rows, and both timing orders: 24 cases, 48 records, four methods each. Baselines use the previously selected larger tiles, so these gains are **additional to tile tuning**. The final confirmation includes both the exact original integer kernel and the matched experimental byte baseline. All confirmation outputs are bitwise identical to their same-tile reference.

Contiguous BF16, mean of both timing orders, complete call:

| Projection | M | Tuned baseline us | Unpack-only us | Aligned buffers + unpack us | Combined speedup |
|---|---:|---:|---:|---:|---:|
| FFN gate/up | 1 | 139.07 | 125.67 | 108.33 | 1.284x |
| FFN gate/up | 4 | 199.66 | 187.10 | 162.88 | 1.226x |
| FFN gate/up | 8 | 289.71 | 280.14 | 254.42 | 1.139x |
| FFN down | 1 | 70.13 | 61.52 | 53.15 | 1.319x |
| FFN down | 4 | 99.51 | 93.18 | 83.37 | 1.194x |
| FFN down | 8 | 142.64 | 138.25 | 129.96 | 1.098x |

The first sweep and the confirmation are separate measurements and do not have identical absolute timings. Clocks are uncontrolled. No inference about end-to-end gains is made from these kernel ratios.

## Loading: aligned buffers outperform manual grouping

Raw PQ2 blocks contain a two-byte FP16 scale followed by 32 code bytes. Their 34-byte stride supports aligned 16-bit loads with suitable base/row alignment, but not uniform aligned 32-bit block loads. Preparation splits exactly those bytes into contiguous code and FP16-scale buffers without dequantization or arithmetic changes. Both buffers together still contain exactly 34 bytes per block.

The winning byte-oriented kernel compiles to wider loads from the aligned buffers. For representative FFN gate/up M=4, generated SASS contains five static `LDG.E.128` sites in the aligned-byte variant versus one in the interleaved-byte variant, with interleaved scalar/halfword sites removed. Static sites are instruction evidence, not memory-transaction counters.

Manually grouping two/four code bytes per dot lane often loses despite wider source elements. In that same case, grouping four bytes increases static butterfly-shuffle sites from 24 to 216 and registers from 125 in the aligned-byte winner to 158. Grouped raw-half loads have 536 shuffle sites. The initial manual-word pipeline increases them further to 600, with 186 registers and 47360 bytes of shared memory. Some M=8 grouped/pipelined variants spill; the aligned-byte winner does not.

## Unpacking: fewer operations with identical bytes

The new expansion first spreads each two-bit code into a separate byte. Every byte is in 0..3. Adding 127 cannot carry between bytes; XOR with 0x80 per byte maps 127/128/129/130 to signed -1/0/1/2. This replaces four individual subtractions and associated masks. Unsigned arithmetic is explicit in the implementation.

An exhaustive GPU test checks all 256 packed bytes. For representative interleaved FFN gate/up M=4, SASS `LOP3.LUT` sites fall from 594 to 402 and `IADD3` from 330 to 138, while `IDP.4A.S8.S8` stays at 256. The experiment therefore removes packing work without reducing or approximating the integer dot products.

## Warp scheduling: preserving lanes helps, but does not win

The follow-up isolates wider loads while retaining the original 32 dot lanes, instead of shrinking to 16/8 lanes and unpacking two/four bytes per lane. It covers five strategies across the same six projection/batch cases: 30 variants, both timing orders, with SASS at M=4.

Keeping lanes reduces the manual-word variant's representative shuffle sites from 216 to 24. Nevertheless, it loses to aligned bytewise dots in every case. Its SASS contains 69 scalar `LDG.E` sites and one `LDG.E.128`, compared with five scalar and five 128-bit sites in the aligned-byte winner. This distinguishes lane-exchange overhead from vectorized loading: eliminating the former is insufficient when the load layout loses the latter.

Two-stage software pipelining with loop unrolling and the tested `.cg` weight-cache policy provide no consistent improvement over the aligned-byte winner. Instruction inspection confirms asynchronous global/shared operations in pipelined variants, but they also increase shared-memory/register demands and sometimes spills. Do not assume that a wider element type, more pipeline stages, or a lower register count alone improves throughput.

## Preparation and integration costs

First preparation takes 0.535 ms for FFN gate/up and 0.293 ms for FFN down in this run. New logical payloads are 47,349,760 and 23,674,880 bytes respectively, exactly matching the originals. Those are cold observations, not distributions. The source is retained during benchmarking, so preparation allocates **one additional packed copy**. Allocator padding and scratch are excluded from logical sizes.

A persistent prepared representation is necessary. Per-call preparation would overwhelm the measured savings. Current prefill/fallback kernels require the original interleaved representation, so it cannot simply be discarded while retaining those paths. Production integration must either share the new layout across decode, prefill, and fallbacks, or explicitly budget the additional resident weights and reduced KV-cache capacity. Loading/preparation peaks and serving VRAM must be measured. The lower-risk intermediate improvement is unpacking alone, which needs no prepared buffers.

## Numerical and implementation validation

102 prototype tests passed: all packed byte values; raw byte/half and separate byte/half/word formats; signed/zero scales; zero activation blocks; masked output/block tails; row strides and odd base/stride rejection for raw half loads; empty outputs; BF16/FP16/FP32 arithmetic; and changing-input CUDA graph replay. Lane-preserving wider variants also match the current integer reference bitwise in their graph tests.

All 78 first-sweep variants and 30 lane variants match their same-tile BF16 reference bitwise. All 24 independent dtype/stride confirmation cases match bitwise as well. Using the original 16-row/four-block reduction tile, five variants replay all 54 saved real-model projections: **270 comparisons bitwise identical to saved integer outputs**. The larger tiles still have the separately documented reduction-order changes relative to that original tile; this result establishes packing/layout parity, not universal native floating-point compatibility or Q8 quality acceptance.

Ruff, Python syntax, and plugin whitespace checks pass. Hardware performance counters are unavailable (`ERR_NVGPUCTRPERM`) under current driver permissions. Findings use repeated timings, compiled resources, and actual SASS, not dynamic transaction/occupancy measurements. No driver settings or GPU power limits were changed.

## Reproduction and artifacts

From /home/amorley/vllm, using the repository virtualenv and an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=1 GGUF_PQ2_INT_GEMV=0 GGUF_PQ2_BATCHED_GEMV=0 \
  PYTHONPATH=vllm-gguf-plugin-prism .venv/bin/python \
  vllm-gguf-plugin-prism/benchmarks/investigate_pq2_packed.py --output new-packed
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=vllm-gguf-plugin-prism .venv/bin/python \
  vllm-gguf-plugin-prism/benchmarks/confirm_pq2_packed.py \
  --sweep new-packed/results.json --output new-packed/confirmation.json
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=vllm-gguf-plugin-prism .venv/bin/python \
  vllm-gguf-plugin-prism/benchmarks/replay_pq2_packed.py \
  --captures benchmarks/results/pq2-int-numerics-20261001 \
  --output new-packed/model-replay.json
```

The current investigation CLI includes all 16 strategies. The original sweep's 13 strategies exclude `half_keep`, `soa_half_keep`, and `soa_word_keep`. Reproduce the lane follow-up with `--variants byte_spread soa_byte_spread half_keep soa_half_keep soa_word_keep`. SASS extraction uses /usr/local/cuda-13.2/bin/cuobjdump; hardware-specific paths can be adjusted locally.

Workspace `results.json`, `confirmation.json`, `lane-sweep/results.json`, `model-replay.json`, initial prototype snapshot, .cubin/.sass files, counter-access output, logs and `summary.json` retain full evidence. The plugin repository retains prototypes, tests, benchmark/replay tools, this report, compact JSON evidence, and instruction excerpts. No prepared-weight serving integration or new default dispatch is included.
