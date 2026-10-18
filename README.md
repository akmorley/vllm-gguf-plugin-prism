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

## Ternary Bonsai (PQ2) models

This plugin can run PrismML's Ternary Bonsai models, which store most weights as one of three values (-1, 0 or +1)
in the `PQ2_0` GGUF format. A 27 billion parameter model of this kind fits in about 7 GB of GPU memory. Running it
fast needs special code, and that code is what this section describes.

### Turning the fast code on

Set these three variables before starting the server:

```bash
export GGUF_PQ2_INT_GEMV=1 GGUF_PQ2_PREPARED=1 GGUF_PQ2_MMQ=1
vllm serve Ternary-Bonsai-2-27B-PQ2_0.gguf --tokenizer <tokenizer directory> \
  --dtype bfloat16 --kv-cache-dtype int8_per_token_head
```

Without them the model still loads and runs, but through slower general-purpose code.

The fast paths are written for NVIDIA GPUs of the Ampere family with compute capability 8.6 (the GeForce RTX 30
series, the RTX A4000 and similar). They also need the model to run in `bfloat16` or `float16`. On any other GPU or
data type the plugin quietly uses the general code instead.

On one RTX 3090 Ti limited to 300 W, a single request generated about 94 tokens per second, and a 4096-token prompt
was processed at about 2000 tokens per second. Numbers depend heavily on the card and its power limit.

### Choosing how aggressive to be

One variable selects the whole set of fast paths:

| `GGUF_PQ2_KERNELS` | What you get |
|---|---|
| `tuned` (default) | Everything, including a few paths whose results differ from the older code by tiny rounding amounts, and one that changes results by about 0.7% per layer (see below). |
| `exact` | Only the fast paths that give bit-for-bit the same results as the older code. Use this when you need to compare outputs exactly. |
| `legacy` | None of the newer fast paths. |

Each fast path also has its own switch, which overrides the setting above when it is set to `0` or `1`. They are
mainly useful for finding the cause of a problem or for comparing speeds.

| Switch | What it controls | Same results as before? |
|---|---|---|
| `GGUF_PQ2_MMQ_TUNED_TILES` | Picks the work-split sizes for long prompts by shape | Yes |
| `GGUF_PQ2_PREFILL_ROWS` | Processes several blocks per step in the prompt preparation steps | Yes |
| `GGUF_PQ2_MMA_PREFILL` | Prompt processing on the GPU's integer matrix units | Yes |
| `GGUF_PQ2_SMALL_MMQ_RT` | Reuses data between neighbouring rows for batches of 2 to 16 tokens | Yes |
| `GGUF_PQ2_SMALL_MMQ_RING` | Streaming kernel for the narrow layers at batches of 2 to 8 tokens | Mostly (see next row) |
| `GGUF_PQ2_SMALL_MMQ_RING_EXACT` | With the line above, keep only the cases that give identical results | Yes |
| `GGUF_PQ2_M1_CUDA` | Hand-written kernel for generating one token at a time | Differs by rounding only |
| `GGUF_PQ2_MMQ_OUTPUT` | Runs the 5120 x 6144 output layers on integer activations during prompt processing | About 0.7% per layer |
| `GGUF_PQ2_MMQ_GLUON` | Prompt processing with the Gluon (Triton) kernel instead of the CUDA one, 7-9% faster | Yes |
| `GGUF_PQ2_SMALL_GLUON` | Gluon (Triton) kernels for batches of 2 to 16 tokens, replacing the CUDA and older Triton ones | Mostly; `exact` keeps only the identical cases |
| `GGUF_PQ2_M1_TRITON` | Triton kernel for generating one token at a time, replacing the hand-written CUDA one | Differs by rounding only |
| `GGUF_SMALL_GEMM_TRITON` | Triton kernel for small unquantized layers (Qwen3.5 `in_proj_ba`) at batches of up to 16 tokens, instead of cuBLAS | Differs by rounding only |

With the default `tuned` set on an RTX 3090 or 3090 Ti, every PQ2 matrix multiply at 1 to 16 tokens and in prompt
processing runs on Triton / Gluon kernels; the CUDA kernels remain as fallbacks for other GPUs, other shapes and larger
batches. Compared with the previous defaults (the CUDA kernels) this is 4.8% faster at one request, 20% at four, 16% at
eight, 35-41% at 12 to 16 concurrent requests and 7.5-8.5% faster at prompt processing (Ternary Bonsai 2 27B PQ2_0, RTX
3090 Ti at 300 W, `--max-num-seqs 16`), with GSM8K answers identical on 127 of 128 questions.

"Rounding only" means the same numbers are added in a different order, so results can differ in the last few digits
and are always the same from run to run. The output-layer switch changes results by a little more. We checked it on
grade-school maths questions, science questions and perplexity, and saw no loss, but the science test is noisy, so
use `GGUF_PQ2_KERNELS=exact` if you need the older results.

These older switches still work: `GGUF_PQ2_SINGLE_GEMV` (the previous one-token kernel, `=0` to use the general
one), `GGUF_PQ2_SMALL_MMQ` (integer matrix units for batches of 2 to 64 tokens), `GGUF_PQ2_SMALL_MMQ_OUTPUT`
(integer version of the 5120 x 6144 layer at batches of 2 to 8 tokens, about 0.7% change) and
`GGUF_PQ2_VERIFY_ATTN` (faster attention while checking drafted tokens). Several earlier experimental switches were
removed and are ignored if set.

The values that are in effect, and the version of the kernels, are included in vLLM's compile cache key, so changing a
switch never reuses stale compiled code. If you compare two settings bit for bit, also set
`VLLM_DISABLE_COMPILE_CACHE=1`: a cached compile and a fresh compile of the same model can give slightly different
numbers.

### Using less memory for the model's running state

Most layers of this model keep a running summary of the text so far (they are called Gated DeltaNet layers). vLLM
stores it in 32-bit floating point by default. Adding `--mamba-ssm-cache-dtype bfloat16` halves that memory and made
generation a few percent faster (about 9% with eight requests at once). In our tests, perplexity over 32,000-token
passages and a set of short maths questions did not change. We have not checked long reasoning tasks, so test it on
your own workload first.

### Faster generation with a draft model (speculative decoding)

The DFlash2 draft model can generate several tokens ahead for the main model to check. On RTX 3090 and 3090 Ti cards
the best setting we measured is five drafted tokens with an 8-bit (FP8) draft model:

```bash
vllm serve Ternary-Bonsai-2-27B-PQ2_0.gguf --tokenizer <tokenizer directory> \
  --speculative-config '{"method": "dflash", "model": "<bonsai-dflash2 directory>", "num_speculative_tokens": 5, "quantization": "fp8"}'
```

The FP8 draft needs a vLLM build that supports it (the `dflash-quantized-context-kv` branch of the `vllm-prism`
repository). With any other vLLM, leave out `"quantization"` and the draft runs in 16-bit, about 10% slower.

For a single request this gave roughly 1.9 to 2.3 times the speed of normal generation, depending on the prompt
(averages of about 210 tokens per second at a 1,000-token prompt and 180 to 195 at 4,000 tokens, against 94 to 97
without a draft). The gain shrinks as more requests run at once: about 1.8 times with two, 1.4 times with four, and
at eight requests normal generation is faster, so do not use drafting for heavily shared servers.

### Optional: Triton attention for single requests

By default the plugin uses vLLM's FlashAttention 2 for attention. For one request at a time you can swap in a Triton
version for the generation step only:

```bash
GGUF_PQ2_SINGLE_ATTN_BACKEND=triton \
GGUF_PQ2_SINGLE_ATTN_SEGMENTS=32 \
vllm serve MODEL.gguf --attention-backend FLASH_ATTN [other options]
```

This is off by default and experimental. It only applies to models with 24 query heads, 4 key/value heads and a head
size of 256, in `bfloat16` on compute capability 8.6 GPUs. Everything else, including prompt processing and batches,
keeps using FlashAttention 2. The numerical checks passed, but the generated text can change, and its quality has not
been evaluated, so check your results before relying on it. Keep `--attention-backend FLASH_ATTN` set, and restart the
server after changing these variables. A 64-segment setting also exists; 32 is the tested starting point.

## License

This package is released under the Apache License 2.0; see `LICENSE`. Some CUDA files were copied or adapted from
llama.cpp, which is MIT licensed. They are listed, with the MIT text, in `THIRD_PARTY_NOTICES.md`.
