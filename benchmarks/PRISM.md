# Prism validation and kernel measurements

The plugin rejects Prism tensor parallelism greater than one before constructing
model layers. Supporting it requires rank-aware signs and block-aligned sharding,
including GDN layout validation. Partial folded packed projections now raise an
error. Version 2 installs the forward transform on the tied output head while
sharing embedding weight parameters, keeping the head's quantization method.

PQ2 decode uses a packed GEMV kernel for up to four tokens and widths up to 32768.
Other batches use tiled GEMM. Both decode PQ2 blocks directly on the GPU without
allocating a dense weight matrix. CUDA Hadamard transforms fuse signs, butterflies
and normalization, accumulating in FP32. CPU transforms retain the eager reference.
Signs are cached by width, device and dtype and preloaded during model loading;
a missing execution copy during CUDA graph capture raises an explicit error.

On an RTX 3090 Ti, FP16 inputs and a 4096 x 5120 PQ2 matrix produced these isolated
measurements (milliseconds):

| Tokens | PQ2 native | Dequantize + dense matmul | Fused FWHT | Eager FWHT |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.0184 | 0.1627 | 0.0043 | 0.1948 |
| 128 | 0.1491 | 0.2016 | 0.0104 | 0.1763 |

These are kernel timings, not serving throughput or time-to-first-token results.
The fallback creates a 40 MiB dense weight for this matrix; native kernels avoid
that allocation. Tile choice was measured on this GPU and may need tuning on
other architectures. Accumulation order can cause small floating-point differences.

Run the benchmark on an idle GPU from the repository root:

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/benchmark_prism.py
```

Validation includes packed PQ2 reference comparisons in FP16/BF16/FP32, masked
tiles, explicit signs, roundtrips, CUDA graph capture, partial packed coverage and
shared tied-head parameters. The local version-1 Ternary-Bonsai-2-27B-PQ2_0 model
passed the existing eight-token and logprob reference correctness test with
bonsai2-reference.json.
The version-2 real-model test was skipped because this model has a separate head;
tied-head behavior is covered by a unit regression. The focused suite passed 95 tests; the real-model suite passed two tests and
skipped the version-2 metadata test.
