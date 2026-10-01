# Opt-in integer PQ2 serving — 2026-10-01

Default-off integer serving improves matched short-prompt throughput by 1.54x at concurrency 1, 3.22x at concurrency 4, and 1.75x at concurrency 8. Greedy diagnostics diverge from native, and candidate repeats are not token-identical on one prompt. This slice establishes experimental serving integration and graph execution, not quality acceptance or default adoption.

## Dispatch and validation

`GGUF_PQ2_INT_GEMV=1` is read once at import before warmup/capture. It selects the explicit integer API only for actual M in {1,2,4,5,8}, FP16/BF16 CUDA activations, SM86, and the five measured (N,K) projection shapes: (34816,5120), (5120,17408), (16384,5120), (14336,5120), (248320,5120). Actual M includes graph padding and any eligible small prefill projection. Other cases retain existing dispatch. Integer selection takes precedence over the floating experiment on eligible cases. Both flags zero restore native dispatch in a fresh process. M=16 remains native because some measured gains were near noise; M=3/6/7 are unmeasured.

Normal model warmup and capture compile selected variants; there is no runtime autotuning. Changing an environment flag after import/capture does not change captured arithmetic. The Q8 quantizer and DP4A kernel are unchanged from the previous slice.

All 151 focused GEMV/MMQ/Prism GPU/Hadamard/linear/integration tests passed, including selector fallbacks, precedence, and actual public dispatch with changing-input graph replay at every selected M in BF16/FP16. Six CPU numerical-harness tests passed. The controlled diagnostic worker now explicitly disables both serving flags for its native reference. Ruff, syntax, and plugin whitespace checks passed.

## Matched serving measurements

Fresh native and integer servers ran sequentially on physical GPU 1, RTX 3090 Ti, SM86, power limit 275 W, uncontrolled clocks. BF16, TP1, max model length 8192, max sequences 8, max batched tokens 2048, GPU utilization reservation 0.85, seed zero, prefix caching disabled, language model only. Both serve the same GGUF and tokenizer on localhost port 8099. Shared numeric token-ID prompts have no chat template or BOS. Eight requests per repeat, 128 input and 128 output tokens, ignored EOS, two warmup rounds and two measured repeats per concurrency. Prompt/output counts are verified. Sixteen measured requests per case give exploratory P95 estimates.

| Concurrency | Native throughput tok/s | Integer throughput tok/s | Gain | Native median TPOT ms | Integer median TPOT ms | Native median TTFT ms | Integer median TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.25 | 38.92 | 1.54x | 38.36 | 24.44 | 190.24 | 182.85 |
| 4 | 25.35 | 81.73 | 3.22x | 152.95 | 43.43 | 762.65 | 743.47 |
| 8 | 72.72 | 126.94 | 1.75x | 99.28 | 52.25 | 1471.18 | 1427.04 |

Native uses its GEMM crossover at M=8, so its eight-request baseline is already faster than its four-request baseline. Gains should not be extrapolated from the four-token kernel ratio. Throughput includes prefill and transport; TPOT excludes the first token. This run does not measure long-prompt prefill gains or re-benchmark llama.cpp.

Sampled peak GPU memory is 20370/20424/20610 MiB native and 20386/20440/20626 MiB integer at concurrency 1/4/8. The 16 MiB increase includes allocator/captured storage and is not an isolated scratch-memory measurement; one-second sampling can miss transient peaks. Both repeat throughputs are stable: native 25.26/25.25, 25.34/25.36, 72.69/72.74; integer 38.93/38.90, 81.67/81.80, 126.97/126.91.

The integer server runs under an inactive Nsight session during throughput collection; active tracing occurs only afterwards. Profiling overhead outside active collection was not independently quantified. Existing servers on GPUs 0 and 2 are preserved. GPU 1 returns to 1 MiB used after the isolated servers exit.

## Greedy numerical diagnostics

Eight fixed text prompts, two four-request batches, 64 greedy tokens per prompt and top-five logprobs, collected twice on each server. Native repeats have identical tokens but nonzero logprob variation (mean absolute generated-logprob delta 0.00128831, maximum 0.0722567). Candidate and its repeat diverge on prompt index 0 at zero-based token position 60; their aligned mean absolute logprob delta is 0.00302958, maximum 0.113023.

Native versus first integer run diverges on three prompts: indices 0/2/7 at positions 60/15/24. Native versus integer repeat diverges on two prompts: indices 2/7 at 15/24. Logprobs are compared only along shared histories, including the shared-context divergence prediction. First-run aligned generated-logprob change is mean 0.00602410, max 0.103759, across 419 contexts; repeat comparison is mean 0.00573240, max 0.103759, across 423 contexts. Later positions have different histories and cannot be treated as arithmetic comparisons.

Serving-repeat variation prevents attributing every difference solely to the candidate arithmetic. The previous synchronized eager fixed-context diagnostic gives a cleaner comparison: native logits repeat exactly, integer metrics repeat identically across 512 contexts, mean TV 0.436%, and one argmax change at a native tie. That is a different execution path and is not exact serving equivalence. Neither eight-prompt diagnostic establishes task accuracy. An agreed Q8 quality threshold, wider fixed task evaluation against native and llama.cpp, and wider serving workloads remain required before default adoption.

## CUDA graph evidence and remaining overhead

A separate eight-request, 128-input/64-output decode capture starts after every request emits its first token. The recorded interval contains 7719 integer GEMV launches and 7719 quantizer launches; 7680 of each have nonzero CUDA graph IDs. Integer grids 320/896/1024/2176/15520 correspond to all five measured projection families. The remaining launches include output-head work outside the decoder graph.

The trace attributes 88.39% of summed kernel time to PQ2 matmul, 5.35% to GDN/recurrent/convolution, 1.37% to Hadamard, 0.72% to activation quantization, and 0.061% to sampling/reductions. Name-based categories and this one capture are diagnostic, not a universal bottleneck claim. The small separate Hadamard/quantizer fraction limits the likely gain from simply fusing those launches; PQ2 projection work remains the measured dominant cost. Kernel idle fraction is 0.22%, including copies and synchronization. These traced timings are not the unprofiled throughput results.

## Reproduction and retained evidence

From `/home/amorley/vllm`:

```bash
.venv/bin/python benchmarks/pq2_integer_serving.py \
  --output benchmarks/results/new-int-serving
```

The script launches only its isolated servers, preserves existing healthy service on port 8099 by refusing to start, and terminates its own process groups on completion or failure. Exact server parameters are in commands.json. Manual runs require `.venv/bin` first in PATH, CUDA_VISIBLE_DEVICES=1, PYTHONPATH=vllm-gguf-plugin-prism, GGUF_PQ2_BATCHED_GEMV=0, and GGUF_PQ2_INT_GEMV=0/1. `pq2_capture.py --session pq2-int-vllm` selects the distinct profiler session.

Workspace artifacts include raw requests, workloads/prompts, telemetry, logs, repeated quality JSON, comparison.json, graph-check.json, source fingerprints, .nsys-rep/.sqlite trace and summary. The plugin repository retains this report and compact JSON evidence; full workspace artifacts are outside that Git repository. No default dispatch adoption is included.
