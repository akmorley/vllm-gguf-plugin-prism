# Serving performance

Run from the plugin repository with the Python environment that contains vLLM,
the GGUF plugin extension, and the Prism-enabled GGUF package:

```sh
/home/amorley/vllm/.venv/bin/python benchmarks/benchmark_serving.py --gpu 0
```

The model defaults to
`/home/amorley/vllm/models/Ternary-Bonsai-2-27B-PQ2_0.gguf` and the tokenizer to
`/home/amorley/vllm/models/qwen38-tokenizer`. Override them with `--model` and
`--tokenizer`. Choose an idle GPU using its physical `nvidia-smi` index or UUID.
The script pins the server to that UUID, uses TP=1, binds to localhost port 8097,
and shuts down its server process group on completion or failure.

The default sweep uses 128/1024/3072 input tokens, 128 output tokens, concurrency
1/4/8, and 32 requests per workload. Requests stream through `/v1/completions`.
Each workload has at least two warmup requests (raised to its concurrency when
needed), fixed-length random prompts, seed zero, and ignores EOS to keep decode length consistent. Prefix caching is disabled to
avoid making later workloads artificially faster through cache reuse. CUDA
graphs and normal server compilation remain enabled by default. The server
sequence limit defaults to the largest workload concurrency so it does not
capture unnecessarily large batches. Set `--max-num-seqs` to a fixed deployment
value when comparing different concurrency sweeps.

A smaller run:

```sh
/home/amorley/vllm/.venv/bin/python benchmarks/benchmark_serving.py \
  --gpu 0 --input-lengths 128 1024 --concurrency 1 4 \
  --requests 8 --output-length 64 --repeats 2
```

Add `--enforce-eager` to measure eager execution separately. Use the same lengths,
concurrency, request count, GPU and memory utilization when comparing runs.
`--result-dir` must name a new directory; results are never overwritten.
`--startup-timeout` defaults to 900 seconds and `--workload-timeout` to 1800 seconds.
Server startup/compilation and warmups are excluded from throughput measurements.
First-use CUDA module compilation can take several minutes; follow `server.log`
for progress. The harness adds the Python executable directory to subprocess
`PATH` so environment-provided build tools are available without activation.

Results are written under `benchmarks/results/<UTC timestamp>/` by default:

- `summary.csv` and `summary.json`: completed requests, requests/s, output tokens/s,
  total tokens/s, mean/p95 time to first token (TTFT), time per output token (TPOT),
  end-to-end latency, and sampled peak device memory.
- One vLLM result JSON per workload: detailed request timings, output token counts,
  generated text, errors, and latency percentiles including inter-token latency.
  Each `*-command.json` records the exact benchmark command and effective warmups.
- `*-memory.json`: device-memory samples approximately every 0.5 seconds, including
  the workload warmups; `startup-memory.json` covers loading and graph capture.
- `metadata.json`: model/tokenizer paths, workload settings, GPU identity, package
  versions and exact server command. `server.log` and per-workload logs retain
  startup and benchmark diagnostics.

Concurrency is a client-side cap with an unlimited offered request rate. TTFT and
end-to-end latency reflect active requests; raw benchmark results also retain
client queue information where supported by the installed vLLM version. Random
prompts measure controlled serving workloads, not representative answer quality.

GPU memory is total device usage reported by `nvidia-smi`, including the server's
reserved KV cache and any other processes. It is sampled, so brief allocation
peaks may be missed. It is not PyTorch's peak active tensor allocation. Run on an
otherwise idle GPU and compare with the same `--gpu-memory-utilization` (default
0.85). Partial request failures fail the run; completed workloads remain saved.
