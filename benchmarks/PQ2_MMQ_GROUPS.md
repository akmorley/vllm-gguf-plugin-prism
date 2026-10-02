# Smaller MMQ activation groups — 2026-10-02

The default-off MMQ experiment now supports Q8 activation groups of 32, 64 or 128 values. Smaller groups reduce activation quantization error at the cost of extra integer dot products. A fresh 64-group server recovered the known GSM8K and GPQA regression cases in two repeats each. This is a targeted diagnostic, not evidence for default adoption or a full task-quality score.

## Implementation and controls

`pq2_mmq(..., activation_group=64)` selects the explicit API variant. For serving, set both `GGUF_PQ2_MMQ=1` and `GGUF_PQ2_MMQ_GROUP=64` before startup (32 is also supported). The default group remains 128 and MMQ remains disabled by default. Existing dtype/SM86/projection/token-count guards remain in effect. Unsupported group values fail validation.

Subgroups retain the original 128-weight PQ2 block scale. Quantization still uses FP32 maximum-absolute scales, round-to-nearest-even signed INT8 values, and integer tensor-core dots; accumulation is reset at each subgroup. Code bytes are loaded once per subgroup and expanded into signed values. Weight storage and preparation remain unchanged. Temporary storage is M*K INT8 bytes plus M*K/group FP32 scales: 1.03125/1.0625/1.125 bytes per activation for group 128/64/32. These are buffer sizes, not measured peak memory.

Both group value and `q8-groups-m128-byte32-v3` enter compilation factors. Group-aware probes verify the selected variant. This prevents an AOT graph compiled for one group being reused with another.

## Numerical and kernel evidence

Fifty-four saved real-activation projection probes from `pq2-int-numerics-20261001` are replayed against their saved FP64 dequantized references. They contain short/decode activation rows and sampled output rows, so they are diagnostics rather than representative long-prefill task evaluation. Mean RMSE below is unweighted across those probes; layer/output units vary. Smaller groups do not improve every probe.

| Group | Mean projection RMSE | Improvement vs 128 | Improved probes | Median latency vs 128 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.004343 | reference | — | 1.000x |
| 64 | 0.004134 | 4.8% | 38/54 | 1.102x |
| 32 | 0.003897 | 10.3% | 50/54 | 1.524x |

Twenty timing cases cover all five measured model projections, raw/prepared weights, BF16, M=128/512. Complete calls include activation quantization and output allocation, exclude preparation, and use 100 ms CUDA graph replay in forward/reverse group orders. FFN and QKV run on physical GPU 2 RTX 3090; GDN and output head on GPU 1 RTX 3090 Ti. Compare group ratios within each case, not absolute timings across GPUs. Both GPUs have 275 W caps and uncontrolled clocks. Existing servers remain resident. The quality driver was suspended while timing the second matrix to avoid concurrent control-server requests.

Group 64 costs 5.5–24.0% more latency, median 10.2%; group 32 costs 33.6–64.6% more, median 52.4%. Default group-128 outputs match the preserved previous implementation byte for byte in all 20 timed cases. This slice does not measure fresh matched end-to-end speed for group 64 or establish that the smaller group retains native-serving speed gains.

## Model diagnostic

`model-trial/commands.json` records exact commands, requests, source artifact paths and budgets. A fresh candidate runs on physical GPU 2 RTX 3090 with BF16, raw weights, floating decode, MMQ enabled, group 64, no prefix caching, max sequences 1, max batched tokens 2048, and max length 8192. The native and group-128 controls are already-running RTX 3090 Ti servers on GPUs 0/1 from the previous slice; the group-128 server predates byte expansion. Byte-identical default-kernel checks support arithmetic continuity, but this remains a diagnostic across different server generations/devices, not a strictly matched benchmark.

Each mode receives the exact saved requests, serially, with temperature/seed zero and two repeats per question. GSM8K row 831 keeps the 1024-token budget. GPQA Diamond row 66 uses 4096 tokens for every mode, increased from the earlier smoke's 2048 to diagnose truncation. The candidate probe records 768 MMQ invocations, all group 64; native records zero.

The group-64 candidate and native controls return correct final answers in both repeats of both questions. For GSM8K, native/group64 return 324 while group128 returns 294 twice. For GPQA, native/group64 return the correct letter C twice, while group128 exhausts all 4096 tokens without a parsed final answer in both repeats. Results on these selected failing cases do not estimate full GSM8K/GPQA accuracy. No acceptance threshold has been established.

## Validation and reproduction

Focused MMQ, dispatch, cache and prepared-layout suites pass 223 tests with six skips. Coverage includes all three groups, BF16/FP16/FP32, raw and prepared layouts, strided sources, zero activations/scales, masked rows/tiles, graph capture/replay with changed activations, invalid groups and runtime-M compilation reuse. Ruff checks/format checks pass.

From the containing workspace:

```bash
PYTHONPATH=vllm-gguf-plugin-prism CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/pq2_mmq_groups.py --output /tmp/pq2-mmq-groups-repeat.json --reference benchmarks/results/pq2-mmq-groups-20261002/reference-group128.py --captures benchmarks/results/pq2-int-numerics-20261001
PYTHONPATH=vllm-gguf-plugin-prism CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest vllm-gguf-plugin-prism/tests/test_pq2_mmq.py vllm-gguf-plugin-prism/tests/test_pq2_mmq_dispatch.py vllm-gguf-plugin-prism/tests/test_pq2_prepared.py -q
```

`matrix.json`, `gdn-head-matrix.json`, `summary.json`, `sources.json` and the saved reference retain timings, numerical results and hashes. Raw model replies and dispatch evidence are under `model-trial/`. The candidate trial server shuts down after evaluation; existing native/group128 controls remain running.

Next run wider paired GSM8K and full GPQA Diamond quality comparisons at identical output budgets, then fresh matched same-GPU serving measurements for group64 versus native/group128. Investigate subgroup scheduling and selective projection use if group64 quality holds but the latency cost is too large. Preserve default-off dispatch until those results and quality acceptance criteria are available. No fresh llama.cpp comparison is claimed.
