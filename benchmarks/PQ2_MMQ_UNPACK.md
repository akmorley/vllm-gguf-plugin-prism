# MMQ byte expansion and prefill validation — 2026-10-02

The opt-in PQ2 MMQ kernel now loads 32 packed bytes per scale block, expands their four codes, and reshapes the result into the same 128-element integer dot tile. This replaces loading packed bytes through the 128-element index grid. Quantization, code values, scales, tile sizes, integer dot products and floating accumulation order are unchanged. Raw and prepared readers share the change. The compilation factor is now `q8-128-m128-byte32-v2` so restarted servers compile the optimized implementation.

## Paired kernel measurements

Physical GPU 2 RTX 3090, 275 W; BF16; M=128/512; six actual projection shapes; raw and prepared weights. Complete calls include quantization and output allocation. Preparation is excluded. CUDA graph replay uses 100 ms per measurement with both execution orders; clocks are uncontrolled. The existing server is resident but receives no requests during kernel timing. Model projection dimensions are recorded in every row. Attention/GDN output is measured through the explicit API; serving MMQ dispatch does not select that shape.

| Layout | Cases | Minimum speedup | Median | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Raw | 12 | 1.125x | 1.190x | 1.319x |
| Prepared | 12 | 1.898x | 2.290x | 2.464x |

Every case matches the saved previous MMQ implementation byte for byte. These are kernel speedups over previous MMQ, not gains over native floating prefill or measured end-to-end gains from this new unpacking change. FP16/FP32 and masks are covered by correctness tests, but are not timed in this slice. The focused MMQ, dispatch and prepared suites pass 119 tests with six skips; the final cache-version checks pass three additional tests. Ruff checks pass.

## Recovered serving comparison

Before changing unpacking, completed the interrupted comparison using the existing MMQ server. Both modes use the same physical RTX 3090, raw weights, BF16, floating decode, prefix caching disabled, max sequences 8, max batched tokens 2048 and length 8192. Four requests generate exactly 32 tokens per workload, with warmup and two repeats. Summary values average each repeat's aggregate throughput and request-median TTFT. Native runs precede MMQ; neither clocks nor run order are randomized. Concurrent workloads include scheduling and queueing, and their TTFT is not a pure kernel prefill measurement.

| Input tokens | Concurrency | Native / MMQ tok/s | Native / MMQ TTFT ms | Throughput gain |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 1 | 14.02 / 16.61 | 1374.8 / 1020.7 | 1.184x |
| 1024 | 4 | 13.94 / 16.76 | 5373.2 / 3939.4 | 1.202x |
| 4096 | 1 | 4.76 / 6.27 | 5802.2 / 4181.9 | 1.317x |
| 4096 | 4 | 4.80 / 6.34 | 19291.4 / 14517.7 | 1.320x |

TTFT decreases 24.7–27.9%. Probe artifacts verify MMQ invocation; native has no MMQ calls. Serving artifacts and the summary are in `../pq2-mmq-gsm8k-20261002/serving`. The running servers used the previous unpacking implementation; these results must not be attributed to byte expansion.

## Quality diagnostic and adoption decision

Ran the first eight questions from the previously saved deterministic 128-question GSM8K sample against already-running native and MMQ servers, serially within each mode. Both use four fixed training shots, temperature zero, seed zero, a 1024-token output limit, raw weights and floating decode. Native runs on physical GPU 0 RTX 3090 Ti, MMQ on physical GPU 1 RTX 3090 Ti; quality-request durations are not a matched performance comparison. There are no unparsed or length-limited answers. Native answers 8/8 correctly; MMQ answers 7/8. Question 831 has gold 324: native returns 324 and MMQ returns 294. Two additional repeats per mode produce the same respective answers. Full responses, request bodies, tokenized prompts, original dataset hashes, dispatch probes and repeated failing responses are saved in `../pq2-mmq-gsm8k-20261002/paired-smoke`.

This small diagnostic does not estimate overall task accuracy, but identifies a reproducible quality difference. MMQ remains default-off. The unpacking change preserves MMQ arithmetic and does not address the Q8 approximation. Next investigate this failing context and selective projection dispatch or finer activation groups, then run the full paired quality workload and fresh optimized-server serving comparisons. No fresh llama.cpp comparison is claimed.

## Reproduction

From the containing workspace:

```bash
PYTHONPATH=vllm-gguf-plugin-prism CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/pq2_mmq_unpack_bench.py --reference benchmarks/results/pq2-mmq-unpack-20261002/reference-kernel.py --output /tmp/pq2-mmq-unpack-repeat.json
PYTHONPATH=vllm-gguf-plugin-prism CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest vllm-gguf-plugin-prism/tests/test_pq2_mmq.py vllm-gguf-plugin-prism/tests/test_pq2_mmq_dispatch.py vllm-gguf-plugin-prism/tests/test_pq2_prepared.py -q
```

`matrix.json`, `summary.json` and `reference-kernel.py` retain timings, dimensions and source hashes. The paired-quality runner `benchmarks/pq2_mmq_live_quality.py --help` takes explicit native/MMQ server URLs, saved prompt source, and a new output directory; it never starts or stops those servers.
