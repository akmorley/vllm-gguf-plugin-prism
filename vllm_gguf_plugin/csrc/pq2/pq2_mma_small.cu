// SPDX-License-Identifier: Apache-2.0
// PQ2 x Q8 small-batch projection with direct-fragment INT8 tensor-core MMA (SM80+/SM86).
//
// Y[t, n] = sum_g ws[n,g] * xs[t,g] * (sum_{i in g} code[n,i] * q[t,i] - qsum[t,g]),
// where code in {0,1,2,(3)} is the raw 2-bit PQ2 code (ternary value = code - 1).
//
// mma.sync.m16n8k32.s8.s8.s32: A = 16 weight rows x 32 k, B = 32 k x 8 tokens.
// Per 128-weight group a row holds 32 code bytes (prepared code plane, row-major).
// Thread (gid = lane/4, tid = lane%4) loads 8 contiguous bytes per row: bytes 8*tid .. 8*tid+7,
// as two 32-bit words w0 (A k-slots tid*4+j) and w1 (A k-slots 16+tid*4+j).
// Plane s of those bytes is (w >> 2s) & 0x03030303, i.e. real k = 4*byte + s. One MMA k-step
// per plane. Activations are stored per group in the matching order (see quantize kernel):
//   position s*32 + L, where logical slot L maps to byte(L) = 8*(L%16/4) + 4*(L/16) + L%4.
// No shared memory, no barriers; INT32 group sums equal the DP4A kernel's exactly.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }

// NT = number of 8-token column tiles (1: M<=8, 2: M<=16). UNROLL groups in flight per warp.
template <typename T, int NT, int UNROLL>
__global__ void __launch_bounds__(128) pq2_mma_small_kernel(
    const int8_t* __restrict__ q,        // [M, K] plane-ordered
    const float* __restrict__ xs,        // [M, K/128]
    const int* __restrict__ qsum,        // [M, K/128]
    const uint8_t* __restrict__ codes,   // [N, K/4]
    const __half* __restrict__ wscale,   // [N, K/128]
    T* __restrict__ y,                   // [M, N]            (split == 1)
    float* __restrict__ partial,         // [split, M, N]     (split > 1)
    int M, int N, int K, int split) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gid = lane >> 2, tid = lane & 3;
  const int row_tile = blockIdx.x * (blockDim.x >> 5) + warp;
  const int row0 = row_tile * 16 + gid, row1 = row0 + 8;
  if (row_tile * 16 >= N) return;
  const int groups = K >> 7;
  const int per = groups / split;
  const int g_begin = blockIdx.y * per, g_end = g_begin + per;
  const bool ok0 = row0 < N, ok1 = row1 < N;
  const uint8_t* c0 = codes + (size_t)(ok0 ? row0 : 0) * (K >> 2) + tid * 8;
  const uint8_t* c1 = codes + (size_t)(ok1 ? row1 : 0) * (K >> 2) + tid * 8;

  float acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;

  // Per-thread activation pointers for its B column in each 8-token tile, and its C tokens.
  const int8_t* qb[NT];
  bool tok_ok[NT], c_ok[NT][2];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tok = t * 8 + gid;
    tok_ok[t] = tok < M;
    qb[t] = q + (size_t)(tok_ok[t] ? tok : 0) * K + tid * 4;
    c_ok[t][0] = t * 8 + tid * 2 < M;
    c_ok[t][1] = t * 8 + tid * 2 + 1 < M;
  }
  for (int g0 = g_begin; g0 < g_end; g0 += UNROLL) {
    // Phase 1: issue every load for UNROLL groups (weights, scales, activations, sums).
    uint2 w0[UNROLL], w1[UNROLL];
    float ws0[UNROLL], ws1[UNROLL];
    uint32_t b[UNROLL][NT][8];
    float x[UNROLL][NT][2];
    int qs[UNROLL][NT][2];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int g = g0 + u < g_end ? g0 + u : g_end - 1;  // clamp: tail repeats are skipped below
      w0[u] = ok0 ? __ldg(reinterpret_cast<const uint2*>(c0 + g * 32)) : make_uint2(0x55555555u, 0x55555555u);
      w1[u] = ok1 ? __ldg(reinterpret_cast<const uint2*>(c1 + g * 32)) : make_uint2(0x55555555u, 0x55555555u);
      ws0[u] = ok0 ? __half2float(__ldg(wscale + (size_t)row0 * groups + g)) : 0.f;
      ws1[u] = ok1 ? __half2float(__ldg(wscale + (size_t)row1 * groups + g)) : 0.f;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          b[u][t][2 * s] = tok_ok[t] ? __ldg(reinterpret_cast<const uint32_t*>(qb[t] + g * 128 + s * 32)) : 0u;
          b[u][t][2 * s + 1] = tok_ok[t] ? __ldg(reinterpret_cast<const uint32_t*>(qb[t] + g * 128 + s * 32 + 16)) : 0u;
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int tk = c_ok[t][j] ? t * 8 + tid * 2 + j : 0;
          x[u][t][j] = c_ok[t][j] ? __ldg(xs + (size_t)tk * groups + g) : 0.f;
          qs[u][t][j] = c_ok[t][j] ? __ldg(qsum + (size_t)tk * groups + g) : 0;
        }
      }
    }
    // Phase 2: tensor-core math and per-group scaling.
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      if (g0 + u >= g_end) break;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int c[4] = {0, 0, 0, 0};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint32_t a[4] = {(w0[u].x >> (2 * s)) & 0x03030303u, (w1[u].x >> (2 * s)) & 0x03030303u,
                                 (w0[u].y >> (2 * s)) & 0x03030303u, (w1[u].y >> (2 * s)) & 0x03030303u};
          mma_s8(c, a, b[u][t][2 * s], b[u][t][2 * s + 1]);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          acc[t][j] += (float)(c[j] - qs[u][t][j]) * ws0[u] * x[u][t][j];
          acc[t][2 + j] += (float)(c[2 + j] - qs[u][t][j]) * ws1[u] * x[u][t][j];
        }
      }
    }
  }
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int tk = t * 8 + tid * 2 + j;
      if (tk >= M) continue;
      if (split == 1) {
        if (ok0) y[(size_t)tk * N + row0] = from_float<T>(acc[t][j]);
        if (ok1) y[(size_t)tk * N + row1] = from_float<T>(acc[t][2 + j]);
      } else {
        float* p = partial + ((size_t)blockIdx.y * M + tk) * N;
        if (ok0) p[row0] = acc[t][j];
        if (ok1) p[row1] = acc[t][2 + j];
      }
    }
}

template <typename T, int NT, int UNROLL>
static int launch_t(const void* q, const void* xs, const void* qsum, const void* codes, const void* ws, void* y,
                    void* partial, int M, int N, int K, int split, int warps, cudaStream_t stream) {
  const int tiles = (N + 15) / 16;
  dim3 grid((tiles + warps - 1) / warps, split), block(32 * warps);
  pq2_mma_small_kernel<T, NT, UNROLL><<<grid, block, 0, stream>>>(
      (const int8_t*)q, (const float*)xs, (const int*)qsum, (const uint8_t*)codes, (const __half*)ws, (T*)y,
      (float*)partial, M, N, K, split);
  return (int)cudaGetLastError();
}

template <typename T, int NT>
static int launch_u(int unroll, const void* q, const void* xs, const void* qsum, const void* codes, const void* ws,
                    void* y, void* partial, int M, int N, int K, int split, int warps, cudaStream_t s) {
  switch (unroll) {
    case 1: return launch_t<T, NT, 1>(q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
    case 2: return launch_t<T, NT, 2>(q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
    case 4: return launch_t<T, NT, 4>(q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
    case 8: return launch_t<T, NT, 8>(q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
  }
  return -1;
}

static int pq2_mma_small_dispatch(const void* q, const void* xs, const void* qsum, const void* codes,
                                    const void* ws, void* y, void* partial, int M, int N, int K, int bf16,
                                    int split, int warps, int unroll, void* stream) {
  if (M < 1 || M > 16 || (K & 127) || ((K >> 7) % split)) return -2;
  cudaStream_t s = (cudaStream_t)stream;
  const bool two = M > 8;
  if (bf16)
    return two ? launch_u<__nv_bfloat16, 2>(unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s)
               : launch_u<__nv_bfloat16, 1>(unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
  return two ? launch_u<__half, 2>(unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s)
             : launch_u<__half, 1>(unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
}

#ifdef PQ2_STANDALONE
extern "C" int pq2_mma_small_launch(const void* q, const void* xs, const void* qsum, const void* codes,
                                    const void* ws, void* y, void* partial, int M, int N, int K, int bf16,
                                    int split, int warps, int unroll, void* stream) {
  return pq2_mma_small_dispatch(q, xs, qsum, codes, ws, y, partial, M, N, K, bf16, split, warps, unroll, stream);
}
#else
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// y[M, N] (BF16/FP16, written in place) from plane-ordered Q8 activations and prepared PQ2 planes.
void pq2_mma_small(Tensor q, Tensor xs, Tensor qsum, Tensor weight, Tensor y, int64_t warps, int64_t unroll) {
  const int32_t device_idx = q.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device_idx);
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &raw_stream));
  const int M = (int)q.sizes()[0], K = (int)q.sizes()[1], N = (int)weight.sizes()[0];
  STD_TORCH_CHECK(weight.sizes()[1] == (int64_t)(K / 128) * 34, "pq2_mma_small: bad PQ2 weight shape");
  STD_TORCH_CHECK(y.sizes()[0] == M && y.sizes()[1] == N, "pq2_mma_small: bad output shape");
  STD_TORCH_CHECK(y.scalar_type() == ScalarType::BFloat16 || y.scalar_type() == ScalarType::Half,
                  "pq2_mma_small: output must be BF16 or FP16");
  const char* codes = (const char*)weight.data_ptr();
  const int err = pq2_mma_small_dispatch(q.data_ptr(), xs.data_ptr(), qsum.data_ptr(), codes,
                                         codes + (size_t)N * K / 4, y.data_ptr(), nullptr, M, N, K,
                                         y.scalar_type() == ScalarType::BFloat16, 1, (int)warps, (int)unroll,
                                         raw_stream);
  STD_TORCH_CHECK(err == 0, "pq2_mma_small: launch failed with code ", err);
}
#endif
