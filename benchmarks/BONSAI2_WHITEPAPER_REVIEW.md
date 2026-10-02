# Bonsai 2 whitepaper review — 2026-10-01

Reviewed the September 2026 [official Bonsai 2 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf), all 14 PDF pages, against the current plugin and recorded experiments. Download SHA256: `aea10331ede3b34c34c21d1a45b80fd0fd6e231b3e8db7bd6346e20fcb8402c4`.

The review supports continuing packed matmul optimization. It does not supply numerical acceptance criteria for our integer activation experiment or justify changing serving defaults.

## Implementation implications

- **Preserve the rotation contract.** Section 2.4 and Appendix A.2 specify sign flips before a normalized, 1024-wide Hadamard transform. Our forward path follows that order; inverse embedding transforms require the reversed order. Buffer preparation must preserve codes, FP16 scales, row identity, and transform metadata. Existing GDN permutation handling remains part of the loader contract.
- **Keep weights packed throughout serving.** Sections 3 and A.1 motivate executing the packed representation directly. Our separate code/scale experiment keeps 34 bytes per 128 weights, but retaining its source doubles that matrix's resident packed storage. A production layout needs shared support across decode, prefill, and fallbacks, or a measured explicit memory budget. Kernel timing alone cannot approve this tradeoff.
- **Treat activation quantization as a separate quality change.** Appendix A.1 describes higher-precision activations without specifying internal rounding, group sizes, or scale precision. The local llama.cpp CUDA source explicitly uses `vec_dot_pq2_0_q8_1`, selecting four independently scaled 32-element Q8 chunks per 128-weight PQ2 block. Our quantizer uses one FP32 activation scale across 128 elements; this is a concrete arithmetic difference requiring comparison. Bitwise packed-layout replay proves preservation of our existing integer result, not equivalence to the floating model or the paper's evaluation stack.
- **Retain selected full-precision state tensors.** Section 2.2 excludes recurrent-state and normalization tensors from rotation and ternary quantization. Do not broaden packed optimizations to those tensors based solely on matrix shape.

## Evidence limits

The paper's throughput protocol is warmed batch-one `tg128`/`pp512`, depth zero, three repeats, text-only llama.cpp/CUDA. Our HTTP serving concurrency results include a different workload and execution stack; compare them with matched local llama.cpp measurements rather than the paper's GPU table. The paper contains no RTX 3090 Ti row.

Its task evaluations use H100 vLLM with TP=2/DP=4. Our plugin currently rejects Prism TP>1, and the paper does not identify the exact low-bit serving implementation or activation arithmetic. We cannot claim to reproduce that evaluation configuration.

The 98.2% retention headline covers the 20-task xhigh aggregate. Terminal-Bench and SWE-bench are reported separately and retain about 75%; medium reasoning has a larger aggregate gap. These results make long reasoning and agentic tasks useful validation targets before adopting our additional Q8 approximation. Greedy continuation checks and fixed-context TV/KL diagnostics remain complementary evidence.

The live model card and PDF also differ in exact artifact sizes and some evaluation figures. Pin checkpoint/build hashes and measure actual resident bytes instead of importing headline storage claims into our benchmarks.

## Next-slice recommendation

First integrate and benchmark the simpler unpack arithmetic with the current storage layout, preserving the established integer reduction order. It offers measured gains without introducing a second resident weight representation. Keep the integer serving flag default-off pending task evaluation. Then pursue one shared aligned layout across projection paths and repeat serving/VRAM measurements.

The paper's rotation-overhead roadmap concerns its measured runtime. Our latest trace attributes 88.39% to PQ2 matmul and 2.10% to Hadamard plus quantization, so matmul remains the supported priority here. Denser PTQ1 storage is a later experiment: the paper itself reports that cheaper unpacking can outperform reduced traffic on Ampere.

This review changes no kernels or serving dispatch. Local supporting evidence is in [the packed investigation](results/pq2-packed-20261001/report.md) and [the integer serving report](results/pq2-int-serving-20261001/report.md).
