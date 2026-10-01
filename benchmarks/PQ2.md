# PQ2 projection benchmarks

Run from `/home/amorley/vllm` on an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=vllm-gguf-plugin-prism \
  .venv/bin/python vllm-gguf-plugin-prism/benchmarks/benchmark_prism.py \
  --tokens 1 2 4 5 8 16 32 128 512 1024 256 258 \
  --output benchmarks/results/pq2-kernels-20261001/matrix.csv
```

Defaults cover the five projection shapes identified in the baseline profile,
BF16 and FP16, contiguous packed rows and rows with twice the packed row stride.
Use `--tokens` to add observed CUDA graph padding or mixed scheduling sizes.
The baseline's 256/258-token mixed prefills are included in the example.
`--methods native` reduces measurement time; `dequant_dense` includes unpacking
on every call, while `dense` times an already-dequantized matrix. For strided
weights the diagnostic unpacker also includes a contiguous copy.

Every shape is validated against FP32 accumulation of dequantized weights rounded
to the activation dtype. Timings use warmed CUDA graph replay, keeping JIT and
library initialization outside the replay measurements. `first_call_ms` records
the first native invocation before validation; it can hit an existing on-disk
compiler cache. Set a new `TRITON_CACHE_DIR` to measure fresh compilation, and
retain process startup separately. Timing repetitions are controlled by
`--rep-ms`; repeat entire sweeps to assess clock/power and measurement variation.

CSV records latency, logical packed-weight bytes divided by native latency, and
peak temporary allocated bytes including output. Logical bandwidth counts one
matrix read even for the multi-token GEMV that actually rereads weights, so it is
not a hardware memory-bandwidth measurement. JSON records arguments, GPU,
software versions, registers, spills, and shared memory when exposed by Triton.
The temporary allocation measurement excludes inputs and the dense reference
retained for validation; it includes the contiguous copy for strided diagnostics.
Dense validation of the output head needs several GiB of extra memory.

These are synthetic packed matrices with all four code values and randomized
FP16 block scales. They validate kernel arithmetic and graph compatibility, not
model quality. Candidate integer kernels must additionally undergo logprob and
quality evaluation before default dispatch changes. The benchmark's methods can
be extended with candidate callables alongside the existing native reference.

The explicit `--methods native mmq` option compares the experimental PQ2 x Q8
prefill with the floating-point serving path. MMQ quantizes each 128-element
activation block once, using an FP32 max-absolute scale / 127 and nearest-even
rounding. Each block's INT32 dot product is scaled separately in FP32; weights
are not rounded to the activation dtype before accumulation. Its CSV includes
relative RMS error against the floating-point reference, with a synthetic
workload rejection threshold of 0.03. This threshold is a kernel diagnostic,
not a model-quality acceptance threshold. MMQ is never selected by serving
`pq2_matmul`. Test both BF16 and FP16 and repeat sweeps before dispatch decisions.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=vllm-gguf-plugin-prism \
  .venv/bin/python vllm-gguf-plugin-prism/benchmarks/benchmark_prism.py \
  --projections ffn_gate_up ffn_down --tokens 5 128 512 1024 \
  --methods native mmq --rep-ms 100 \
  --output benchmarks/results/pq2-mmq-20261001/matrix.csv
```
