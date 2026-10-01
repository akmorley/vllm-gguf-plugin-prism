# Remaining PQ2 matmul headroom — 2026-10-01

Yes: matmul remains the measured dominant target, and further tile changes produce repeatable gains. This investigation does not change inference kernels or serving dispatch.

The latest integer eight-request serving trace attributes 88.39% of summed kernel time to PQ2 matmul. Hadamard and activation quantization together account for only 2.10%, so fusing only those launches has limited upside in this capture. The original llama.cpp traces attribute 65–70% to PQ2 decode; percentages alone are not a fair speed comparison because other operations, fusion, and projection layouts differ. Absolute serving TPOT and projection times are more useful. The previous integer serving slice already reduced native TPOT from 38.36 to 24.44 ms at concurrency 1 and 152.95 to 43.43 ms at concurrency 4, but still leaves matmul headroom.

## Additional measurements

On physical GPU 1, RTX 3090 Ti, SM86, power limit 275 W, uncontrolled clocks, Torch 2.13.0+cu130, Triton 3.7.1:

- Sweep 24 row/block/warp configurations across two dominant FFN projections and M=1/4/8: 144 prequantized records.
- Confirm the winning configurations with full calls including quantization and scratch allocation, BF16/FP16, contiguous/strided packed rows, and reversed method order: 24 cases, 48 paired timing records.
- Full-call confirmed speedups range from 1.068x to 1.158x; all cases improve. The exploratory prequantized sweep's larger peak gain does not transfer completely to the confirmation.

Contiguous BF16, mean of both timing orders:

| Projection | M | Current us | Candidate us | Speedup | Rows / scale blocks / warps |
|---|---:|---:|---:|---:|---:|
| FFN gate/up | 1 | 153.67 | 140.50 | 1.094x | 32 / 8 / 8 |
| FFN gate/up | 4 | 222.81 | 197.63 | 1.127x | 64 / 8 / 8 |
| FFN gate/up | 8 | 322.58 | 295.63 | 1.091x | 32 / 8 / 8 |
| FFN down | 1 | 80.91 | 70.03 | 1.155x | 32 / 8 / 8 |
| FFN down | 4 | 111.07 | 99.75 | 1.113x | 32 / 8 / 8 |
| FFN down | 8 | 155.20 | 142.25 | 1.091x | 32 / 8 / 4 |

All selected sweep variants report zero spills. All 24 confirmation comparisons pass atol=0.02, rtol=0.002 versus the existing quantized kernel. Maximum relative RMS difference is 0.000025065. Changing the block tile changes FP32 reduction order and occasionally rounded outputs, so model probability diagnostics are still needed before selecting these tiles in serving. These timings do not establish additional end-to-end throughput gains. Existing opt-in integer dispatch and its fixed tile remain unchanged.

## Next implementation target

Promote measured shape/batch-specific tiles only after numerical and serving validation. For larger gains, investigate vectorized packed-weight loading/unpacking and a CUDA warp layout that reduces integer packing/reduction overhead; benchmark independently rather than assuming CUDA itself is faster. Target FFN gate/up and FFN down first. Retain every 128-weight scale boundary and quantizer semantics. Further Hadamard-only fusion is lower priority for the latest decode profile. Long-prompt prefill still needs its separate MMQ serving/quality work; this decode experiment does not address it.

## Reproduction

From `/home/amorley/vllm`:

```bash
CUDA_VISIBLE_DEVICES=1 GGUF_PQ2_INT_GEMV=0 GGUF_PQ2_BATCHED_GEMV=0 \
  PYTHONPATH=vllm-gguf-plugin-prism .venv/bin/python \
  vllm-gguf-plugin-prism/benchmarks/tune_pq2_int_gemv.py \
  --extended --tokens 1 4 8 --output extended.json
CUDA_VISIBLE_DEVICES=1 GGUF_PQ2_INT_GEMV=0 GGUF_PQ2_BATCHED_GEMV=0 \
  PYTHONPATH=vllm-gguf-plugin-prism .venv/bin/python \
  vllm-gguf-plugin-prism/benchmarks/confirm_pq2_int_tiles.py \
  --sweep extended.json --output confirmation.json
```

`extended.json`, `confirmation.json`, `summary.json`, and logs preserve measurements, compiler resources and source fingerprints. Ruff, syntax, and plugin whitespace checks pass. The benchmark comparisons exercise arithmetic and CUDA graph replay; production code is unchanged, so focused/model suites were not unnecessarily repeated. GPU 1 is released and existing GPU 0/2 servers remain unchanged.
