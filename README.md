# vLLM GGUF Quantization Plugin

This plugin provides out-of-tree GGUF quantization support for vLLM after
in-tree support deprecation
([vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583)).

## Installation

### Prerequisites

- CUDA toolkit or ROCm toolkit

We recommend [uv](https://docs.astral.sh/uv/) for package management. If you
don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone this repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-gguf-plugin
   cd vllm-gguf-plugin
   ```

2. If vLLM is not already installed, install it first:

   ```bash
   uv pip install vllm --torch-backend=auto
   ```

3. Build and install the plugin against the PyTorch installation used by
   vLLM:

   ```bash
   uv pip install -e . --no-build-isolation
   ```

   Disabling build isolation ensures that the CUDA extension is compiled
   against the same PyTorch installation used by vLLM at runtime.

## Development

After completing the editable source installation above, install and run the
development tooling:

```bash
uv pip install -e .[dev] --torch-backend=auto
pre-commit install
pre-commit run --all-files
```

The same hooks also run in GitHub Actions on every push and pull request.

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B
```

Qwen 3.5 MTP speculative decoding loads the `nextn` block embedded in the same
GGUF; it does not download separate Hugging Face MTP weights:

```bash
vllm serve unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M \
  --tokenizer Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

For a GGUF without a `nextn` block, omit `--speculative-config`; the backbone
loads normally without MTP.

## Tested model coverage

The plugin uses vLLM's model implementations and a generic GGUF weight
adapter, so model compatibility is broader than a fixed allowlist. The models
below are covered by the repository's generation tests and are the best-known
starting points:

| Modality | Model family | Tested GGUF quantization |
| --- | --- | --- |
| Text | Qwen 2.5 | Q6_K |
| Text | Qwen 3 | Q8_0 |
| Text | Phi 3.5 | IQ4_XS |
| Text | GPT-2 | Q4_K_M |
| Text | StableLM | Q4_K_M |
| Text | Gemma 3 | Q4_0 |
| Text | OLMoE | Q4_0 |
| Vision-language | Gemma 3 | Q4_0 backbone with F16 projector |
| Vision-language | Gemma 4 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.5 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.6 | UD-IQ2_XXS backbone with BF16 projector |
| Image generation | Z-Image-Turbo | Q4_0 |
| Image generation | FLUX.2-klein | Q8_0 |

Other vLLM-supported architectures may work when their GGUF tensor names map
to the corresponding Hugging Face model. A model appearing in vLLM's general
supported-model list does not by itself guarantee GGUF compatibility. When
reporting an unsupported model, include the model repository, quantization,
plugin and vLLM versions, and the complete weight-mapping error.

## Prism serving benchmark

Use the serving benchmark guide (`benchmarks/plugin-archive/benchmarks/SERVING.md` in the
containing workspace) to measure the local
27B model across prompt lengths and request concurrency. The harness saves
throughput, latency percentiles, GPU memory samples, and raw vLLM results.

The experimental PQ2 integer decode guide
(`benchmarks/plugin-archive/benchmarks/PQ2_INTEGER_DECODE.md` in the containing workspace)
documents the default-off `GGUF_PQ2_INT_GEMV=1` experiment, measured dispatch
rules, kernel results, and numerical acceptance work.

## PQ2 single-request decode defaults

The optimized PQ2 decode path now defaults to the prepared single-token byte
kernel (`GGUF_PQ2_SINGLE_GEMV=1`) and `shape-tuned-v2` geometry
(`GGUF_PQ2_INT_GEMV_VARIANT=shape-tuned-v2`). These choices apply when the
existing integer-decode path is enabled. For the tested optimized serving
configuration, use `GGUF_PQ2_INT_GEMV=1`, `GGUF_PQ2_PREPARED=1`, and
`GGUF_PQ2_MMQ=1` (MMQ group 128). The two new default flags need no overrides.

The single-token specialization selects only the six measured projection
shapes on SM86 with BF16/FP16, prepared weights, PRMT decode, and disabled
chained and fused-gate paths. Other inputs retain their existing dispatch.
Batch-eight chaining and the BM8 floating output projection are included from
the validated optimized configuration. Resolved settings and the kernel
version enter the compilation cache identity.

To restore the prior single-token implementation and geometry, set
`GGUF_PQ2_SINGLE_GEMV=0` and `GGUF_PQ2_INT_GEMV_VARIANT=shape-tuned`.
The experimental attention split override is not enabled by this promotion.

Full-model measurements with 4096 input and 2048 generated tokens showed
18.22% and 17.62% higher generation throughput on the two RTX 3090 Ti GPUs.

## Optional single-request Triton attention

To keep FlashAttention 2 for prefill and use Triton for the measured
single-request decode shape, set these variables before starting the server:

```bash
GGUF_PQ2_SINGLE_ATTN_BACKEND=triton \
GGUF_PQ2_SINGLE_ATTN_SEGMENTS=32 \
vllm serve MODEL.gguf --attention-backend FLASH_ATTN [other options]
```

The option is disabled by default (`GGUF_PQ2_SINGLE_ATTN_BACKEND=fa2`).
It supports BF16 queries with 24 query heads, four KV heads and head dimension
256 on SM86. Prefill, batched decode, other GPUs/shapes/dtypes, sliding-window
attention, softcaps, auxiliary features and context parallelism retain FA2.
The existing paged KV layout is preserved, including 784-token hybrid pages.
Softmax scratch is allocated per call and retained by CUDA graph capture. The resolved backend, segment
count and implementation version enter vLLM's compilation cache identity.
Restart the server when changing settings.

A 64-segment option is also available; 32 is the tested starting point.
Keep the `FLASH_ATTN` vLLM backend selected: this setting changes only eligible
FA2 calls and does not replace other configured attention backends. This is an
experimental option: numerical and graph checks passed, but greedy generated
text can change. Language-quality evaluation is required before promotion.
