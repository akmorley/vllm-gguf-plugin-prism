// SPDX-License-Identifier: Apache-2.0
// Single-token (M = 1) PQ2 x Q8 projection on the existing row-major prepared layout: DP4A on the raw 2-bit planes.
//
// y[n] = sum_g ws[n,g] * xs[g] * ( sum_{i in g} code[n,i] * q[i]  -  qsum[g] )
//
// Plane s of a packed 32-bit weight word is (w >> 2s) & 0x03030303 (four int8 lanes holding codes 0..2); activations
// are quantized in plane order (q[g*128 + s*32 + byte] = Q8(x[g*128 + 4*byte + s]), see _quantize_planes_sum), so each
// plane's four activation bytes are one 32-bit word and no weight decode (PRMT selector) is needed: ~3 integer ops per
// weight byte instead of ~10, which makes the kernel memory-bound. Eight lanes read one 128 B row window (4 groups) with
// LDG.128, four rows per warp instruction; the activation vector, group scales and sums are staged once per CTA in shared
// memory; the next U windows of weights are prefetched into registers; the two lanes of a group combine through a shuffle.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }

template <typename T, int RB, int U, int WARPS, bool SSM, bool PRE>
__global__ void __launch_bounds__(WARPS * 32) pq2_gemv_m1_kernel(const int8_t* __restrict__ qp, const float* __restrict__ xs,
                                                  const int* __restrict__ qsum, const uint8_t* __restrict__ codes,
                                                  const __half* __restrict__ wscale, T* __restrict__ y, int N, int K) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int G = K >> 7, W = G >> 2;
  constexpr int ROWS = WARPS * 4 * RB;
  int8_t* sq = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + K);
  int* sqs = reinterpret_cast<int*>(smem + K + G * 4);
  __half* sws = reinterpret_cast<__half*>(smem + K + G * 8);       // [ROWS][G] when SSM

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int rsel = lane >> 3, part = lane & 7, gsel = part >> 1, half = part & 1;
  const int cta_row0 = blockIdx.x * ROWS;
  const int row_base = cta_row0 + warp * (4 * RB);
  const uint8_t* cp[RB];
  const __half* wp[RB];
  int row[RB];
  bool ok[RB];
#pragma unroll
  for (int r = 0; r < RB; ++r) {
    row[r] = row_base + r * 4 + rsel;
    ok[r] = row[r] < N;
    const int rr = ok[r] ? row[r] : 0;
    cp[r] = codes + (size_t)rr * (K >> 2) + part * 16;
    wp[r] = wscale + (size_t)rr * G + gsel;
  }
  // prologue weight loads (before the staging barrier)
  uint4 wv[U][RB];
  float ws[U][RB];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int w = u < W ? u : W - 1;
#pragma unroll
    for (int r = 0; r < RB; ++r) {
      wv[u][r] = __ldg(reinterpret_cast<const uint4*>(cp[r] + (size_t)w * 128));
      if (!SSM) ws[u][r] = __half2float(__ldg(wp[r] + w * 4));
    }
  }
  for (int i = threadIdx.x * 16; i < K; i += blockDim.x * 16)
    *reinterpret_cast<uint4*>(sq + i) = *reinterpret_cast<const uint4*>(qp + i);
  for (int i = threadIdx.x; i < G; i += blockDim.x) { sxs[i] = xs[i]; sqs[i] = qsum[i]; }
  if (SSM) {
    // rows are contiguous G-halves in global memory: coalesced 4 B pieces
    for (int i = threadIdx.x; i < ROWS * (G >> 1); i += blockDim.x) {
      const int r = i / (G >> 1), c = i % (G >> 1);
      const int rr = min(cta_row0 + r, N - 1);
      reinterpret_cast<uint32_t*>(sws)[r * (G >> 1) + c] = __ldg(reinterpret_cast<const uint32_t*>(wscale + (size_t)rr * G) + c);
    }
  }
  __syncthreads();

  float acc[RB];
#pragma unroll
  for (int r = 0; r < RB; ++r) acc[r] = 0.f;
  const int srow0 = warp * 4 * RB + rsel;                          // this lane's local row for r = 0

  for (int w0 = 0; w0 < W; w0 += U) {
    uint4 nx[U][RB];
    float nws[U][RB];
    if (PRE) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int w = w0 + U + u < W ? w0 + U + u : W - 1;
#pragma unroll
        for (int r = 0; r < RB; ++r) {
          nx[u][r] = __ldg(reinterpret_cast<const uint4*>(cp[r] + (size_t)w * 128));
          if (!SSM) nws[u][r] = __half2float(__ldg(wp[r] + w * 4));
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (w0 + u >= W) break;
      const int g = (w0 + u) * 4 + gsel;
      const int8_t* qg = sq + g * 128 + half * 16;
      const uint4 q0 = *reinterpret_cast<const uint4*>(qg);
      const uint4 q1 = *reinterpret_cast<const uint4*>(qg + 32);
      const uint4 q2 = *reinterpret_cast<const uint4*>(qg + 64);
      const uint4 q3 = *reinterpret_cast<const uint4*>(qg + 96);
      const float x = sxs[g];
      const int qs = sqs[g];
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const uint32_t wd[4] = {wv[u][r].x, wv[u][r].y, wv[u][r].z, wv[u][r].w};
        const uint32_t qa[4] = {q0.x, q0.y, q0.z, q0.w}, qb[4] = {q1.x, q1.y, q1.z, q1.w};
        const uint32_t qc[4] = {q2.x, q2.y, q2.z, q2.w}, qd[4] = {q3.x, q3.y, q3.z, q3.w};
        int dot = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          dot = __dp4a((int)( wd[j]        & 0x03030303u), (int)qa[j], dot);
          dot = __dp4a((int)((wd[j] >> 2)  & 0x03030303u), (int)qb[j], dot);
          dot = __dp4a((int)((wd[j] >> 4)  & 0x03030303u), (int)qc[j], dot);
          dot = __dp4a((int)((wd[j] >> 6)  & 0x03030303u), (int)qd[j], dot);
        }
        dot += __shfl_xor_sync(0xffffffffu, dot, 1);
        const float wsv = SSM ? __half2float(sws[(srow0 + r * 4) * G + g]) : ws[u][r];
        if (half == 0) acc[r] += (float)(dot - qs) * x * wsv;
      }
    }
    if (PRE) {
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int r = 0; r < RB; ++r) { wv[u][r] = nx[u][r]; if (!SSM) ws[u][r] = nws[u][r]; }
    } else {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int w = w0 + U + u < W ? w0 + U + u : W - 1;
#pragma unroll
        for (int r = 0; r < RB; ++r) {
          wv[u][r] = __ldg(reinterpret_cast<const uint4*>(cp[r] + (size_t)w * 128));
          if (!SSM) ws[u][r] = __half2float(__ldg(wp[r] + w * 4));
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RB; ++r) {
    float v = acc[r];
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    if (part == 0 && ok[r]) y[row[r]] = from_float<T>(v);
  }
}

template <typename T, int RB, int U, int WARPS, bool SSM, bool PRE>
static int launch_m1(const void* q, const void* xs, const void* qs, const void* codes, const void* ws, void* y, int N,
                     int K, cudaStream_t s) {
  const int G = K >> 7;
  if (G % 4) return -2;
  constexpr int ROWS = WARPS * 4 * RB;
  const int smem = K + G * 8 + (SSM ? ROWS * G * 2 : 0);
  if (smem > 98304) return -5;
  auto kern = pq2_gemv_m1_kernel<T, RB, U, WARPS, SSM, PRE>;
  static const bool ok = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304) == cudaSuccess;
  if (!ok) return -4;
  kern<<<(N + ROWS - 1) / ROWS, WARPS * 32, smem, s>>>((const int8_t*)q, (const float*)xs, (const int*)qs,
                                                      (const uint8_t*)codes, (const __half*)ws, (T*)y, N, K);
  return (int)cudaGetLastError();
}

template <typename T>
static int m1_dispatch(int cfg, const void* q, const void* xs, const void* qs, const void* codes, const void* ws, void* y,
                       int N, int K, cudaStream_t s) {
  switch (cfg) {  // RB, U, WARPS, SSM, PRE (selected on RTX 3090 Ti with real weights)
    case 0: return launch_m1<T, 2, 1, 4, true, true>(q, xs, qs, codes, ws, y, N, K, s);    // gate/up 34816 x 5120
    case 1: return launch_m1<T, 2, 1, 8, false, true>(q, xs, qs, codes, ws, y, N, K, s);   // in_proj / qkv
    case 2: return launch_m1<T, 2, 2, 8, false, true>(q, xs, qs, codes, ws, y, N, K, s);   // down 5120 x 17408
    case 3: return launch_m1<T, 1, 2, 8, false, true>(q, xs, qs, codes, ws, y, N, K, s);   // out 5120 x 6144
    case 4: return launch_m1<T, 2, 1, 4, true, false>(q, xs, qs, codes, ws, y, N, K, s);   // lm_head 248320 x 5120
  }
  return -3;
}

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// y[1, N] (BF16/FP16, written in place) from one plane-ordered Q8 activation row and prepared PQ2 planes.
void pq2_gemv_m1(Tensor q, Tensor xs, Tensor qsum, Tensor weight, Tensor y, int64_t cfg) {
  const int32_t device_idx = q.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device_idx);
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &raw_stream));
  const int K = (int)q.numel(), N = (int)weight.sizes()[0];
  STD_TORCH_CHECK(K % 512 == 0, "pq2_gemv_m1: K must be a multiple of 512");
  STD_TORCH_CHECK(weight.sizes()[1] == (int64_t)(K / 128) * 34, "pq2_gemv_m1: bad PQ2 weight shape");
  STD_TORCH_CHECK(y.numel() == N, "pq2_gemv_m1: bad output shape");
  STD_TORCH_CHECK(y.scalar_type() == ScalarType::BFloat16 || y.scalar_type() == ScalarType::Half,
                  "pq2_gemv_m1: output must be BF16 or FP16");
  const char* codes = (const char*)weight.data_ptr();
  const void* wscale = codes + (size_t)N * K / 4;
  const int err = y.scalar_type() == ScalarType::BFloat16
      ? m1_dispatch<__nv_bfloat16>((int)cfg, q.data_ptr(), xs.data_ptr(), qsum.data_ptr(), codes, wscale, y.data_ptr(), N,
                                   K, (cudaStream_t)raw_stream)
      : m1_dispatch<__half>((int)cfg, q.data_ptr(), xs.data_ptr(), qsum.data_ptr(), codes, wscale, y.data_ptr(), N, K,
                            (cudaStream_t)raw_stream);
  STD_TORCH_CHECK(err == 0, "pq2_gemv_m1: launch failed with code ", err);
}
