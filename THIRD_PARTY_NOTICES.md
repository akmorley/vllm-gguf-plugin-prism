# Third-party notices

This package is licensed under the Apache License 2.0 (see `LICENSE`). Some files were copied or adapted from other
projects and keep their original licenses, as described below.

## llama.cpp / ggml (MIT License)

The following files in `vllm_gguf_plugin/csrc/gguf/` were copied from, or adapted from, the CUDA sources of
[llama.cpp](https://github.com/ggerganov/llama.cpp) (tag b2899, now maintained at
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)). Each file carries a comment naming its origin.

- `ggml-common.h`
- `dequantize.cuh`
- `mmq.cuh`
- `mmvq.cuh`
- `moe_vec.cuh`
- `vecdotq.cuh`
- `moe.cuh` (adapted from `mmq.cuh` above)

The GGUF file format and the quantization type definitions that the Python code in this package reads and implements
(Q4_0, Q8_0, the K-quants, the IQ types, and so on) are also defined by llama.cpp and the `gguf` Python package.

The PQ2_0 weight format used by the Ternary Bonsai models (128-weight groups of 2-bit codes with an FP16 scale) is
defined by PrismML's fork of llama.cpp, which is also MIT licensed. The PQ2 kernels in
`vllm_gguf_plugin/csrc/pq2/` and `vllm_gguf_plugin/triton/pq2_*.py` read that format but were written for this
package and contain no llama.cpp source code.

```
MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
