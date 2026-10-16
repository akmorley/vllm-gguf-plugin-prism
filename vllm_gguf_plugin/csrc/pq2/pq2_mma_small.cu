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
// RT 16-row weight tiles per warp share each group's activation fragments (RT = 1: one tile per warp);
// per-element arithmetic is identical for every RT.
#include "pq2_common.cuh"

// NT = number of 8-token column tiles (1: M<=8, 2: M<=16). UNROLL groups in flight per warp.
// RT = 16-row weight tiles per warp: the activation fragments of each group are loaded once and
// reused for RT tiles (RT = 1 is the original kernel; per-element arithmetic is identical).
template <typename T, int NT, int UNROLL, int RT>
__global__ void __launch_bounds__(128) pq2_mma_small_kernel(
    const int8_t* __restrict__ q, const float* __restrict__ xs, const int* __restrict__ qsum,
    const uint8_t* __restrict__ codes, const __half* __restrict__ wscale, T* __restrict__ y,
    float* __restrict__ partial, int M, int N, int K, int split) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gid = lane >> 2, tid = lane & 3;
  const int first_tile = (blockIdx.x * (blockDim.x >> 5) + warp) * RT;
  if (first_tile * 16 >= N) return;
  const int groups = K >> 7;
  const int per = groups / split;
  const int g_begin = blockIdx.y * per, g_end = g_begin + per;
  int row0[RT], row1[RT];
  bool ok0[RT], ok1[RT];
  const uint8_t *c0[RT], *c1[RT];
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    row0[r] = (first_tile + r) * 16 + gid;
    row1[r] = row0[r] + 8;
    ok0[r] = row0[r] < N;
    ok1[r] = row1[r] < N;
    c0[r] = codes + (size_t)(ok0[r] ? row0[r] : 0) * (K >> 2) + tid * 8;
    c1[r] = codes + (size_t)(ok1[r] ? row1[r] : 0) * (K >> 2) + tid * 8;
  }
  float acc[RT][NT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[r][t][i] = 0.f;

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
    uint2 w0[UNROLL][RT], w1[UNROLL][RT];
    float ws0[UNROLL][RT], ws1[UNROLL][RT];
    uint32_t b[UNROLL][NT][8];
    float x[UNROLL][NT][2];
    int qs[UNROLL][NT][2];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int g = g0 + u < g_end ? g0 + u : g_end - 1;
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        w0[u][r] = ok0[r] ? __ldg(reinterpret_cast<const uint2*>(c0[r] + g * 32)) : make_uint2(0x55555555u, 0x55555555u);
        w1[u][r] = ok1[r] ? __ldg(reinterpret_cast<const uint2*>(c1[r] + g * 32)) : make_uint2(0x55555555u, 0x55555555u);
        ws0[u][r] = ok0[r] ? __half2float(__ldg(wscale + (size_t)row0[r] * groups + g)) : 0.f;
        ws1[u][r] = ok1[r] ? __half2float(__ldg(wscale + (size_t)row1[r] * groups + g)) : 0.f;
      }
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
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      if (g0 + u >= g_end) break;
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          int c[4] = {0, 0, 0, 0};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t a[4] = {(w0[u][r].x >> (2 * s)) & 0x03030303u, (w1[u][r].x >> (2 * s)) & 0x03030303u,
                                   (w0[u][r].y >> (2 * s)) & 0x03030303u, (w1[u][r].y >> (2 * s)) & 0x03030303u};
            mma_s8(c, a, b[u][t][2 * s], b[u][t][2 * s + 1]);
          }
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            acc[r][t][j] += (float)(c[j] - qs[u][t][j]) * ws0[u][r] * x[u][t][j];
            acc[r][t][2 + j] += (float)(c[2 + j] - qs[u][t][j]) * ws1[u][r] * x[u][t][j];
          }
        }
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int tk = t * 8 + tid * 2 + j;
        if (tk >= M) continue;
        if (split == 1) {
          if (ok0[r]) y[(size_t)tk * N + row0[r]] = from_float<T>(acc[r][t][j]);
          if (ok1[r]) y[(size_t)tk * N + row1[r]] = from_float<T>(acc[r][t][2 + j]);
        } else {
          float* p = partial + ((size_t)blockIdx.y * M + tk) * N;
          if (ok0[r]) p[row0[r]] = acc[r][t][j];
          if (ok1[r]) p[row1[r]] = acc[r][t][2 + j];
        }
      }
}

template <typename T, int NT, int UNROLL, int RT>
static int launch_t(const void* q, const void* xs, const void* qsum, const void* codes, const void* ws, void* y,
                    void* partial, int M, int N, int K, int split, int warps, cudaStream_t stream) {
  const int tiles = (N + 15) / 16;
  const int warp_total = (tiles + RT - 1) / RT;
  dim3 grid((warp_total + warps - 1) / warps, split), block(32 * warps);
  pq2_mma_small_kernel<T, NT, UNROLL, RT><<<grid, block, 0, stream>>>(
      (const int8_t*)q, (const float*)xs, (const int*)qsum, (const uint8_t*)codes, (const __half*)ws, (T*)y,
      (float*)partial, M, N, K, split);
  return (int)cudaGetLastError();
}

template <typename T, int NT, int RT>
static int launch_u(int unroll, const void* q, const void* xs, const void* qsum, const void* codes, const void* ws,
                    void* y, void* partial, int M, int N, int K, int split, int warps, cudaStream_t s) {
  switch (unroll) {
    case 1: return launch_t<T, NT, 1, RT>(q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
    case 2: return launch_t<T, NT, 2, RT>(q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
    case 4: return launch_t<T, NT, 4, RT>(q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
    case 8: return launch_t<T, NT, 8, RT>(q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
  }
  return -1;
}

template <typename T, int NT>
static int launch_r(int rt, int unroll, const void* q, const void* xs, const void* qsum, const void* codes,
                    const void* ws, void* y, void* partial, int M, int N, int K, int split, int warps,
                    cudaStream_t s) {
  switch (rt) {
    case 1: return launch_u<T, NT, 1>(unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
    case 2: return launch_u<T, NT, 2>(unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
    case 4: return launch_u<T, NT, 4>(unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
  }
  return -1;
}

static int pq2_mma_small_dispatch(const void* q, const void* xs, const void* qsum, const void* codes,
                                    const void* ws, void* y, void* partial, int M, int N, int K, int bf16,
                                    int split, int warps, int unroll, int rt, void* stream) {
  if (M < 1 || M > 16 || (K & 127) || ((K >> 7) % split)) return -2;
  cudaStream_t s = (cudaStream_t)stream;
  const bool two = M > 8;
  if (bf16)
    return two ? launch_r<__nv_bfloat16, 2>(rt, unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s)
               : launch_r<__nv_bfloat16, 1>(rt, unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
  return two ? launch_r<__half, 2>(rt, unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s)
             : launch_r<__half, 1>(rt, unroll, q, xs, qsum, codes, ws, y, partial, M, N, K, split, warps, s);
}

#ifdef PQ2_STANDALONE
extern "C" int pq2_mma_small_launch(const void* q, const void* xs, const void* qsum, const void* codes,
                                    const void* ws, void* y, void* partial, int M, int N, int K, int bf16,
                                    int split, int warps, int unroll, void* stream) {
  return pq2_mma_small_dispatch(q, xs, qsum, codes, ws, y, partial, M, N, K, bf16, split, warps, unroll, 1, stream);
}
#else
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// y[M, N] (BF16/FP16, written in place) from plane-ordered Q8 activations and prepared PQ2 planes.
void pq2_mma_small(Tensor q, Tensor xs, Tensor qsum, Tensor weight, Tensor y, int64_t warps, int64_t unroll,
                   int64_t rt) {
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
                                         y.scalar_type() == ScalarType::BFloat16, 1, (int)warps, (int)unroll, (int)rt,
                                         raw_stream);
  STD_TORCH_CHECK(err == 0, "pq2_mma_small: launch failed with code ", err);
}
#endif
