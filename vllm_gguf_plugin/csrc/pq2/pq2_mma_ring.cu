// SPDX-License-Identifier: Apache-2.0
// Small-batch (M <= 16) PQ2 x Q8 projection on the existing row-major prepared layout.
//
// Y[t, n] = sum_g ws[n,g] * xs[t,g] * (sum_{i in g} code[n,i] * q[t,i] - qsum[t,g])
//
// A CTA owns BMW weight rows and the whole (<= 16 token) activation tile. Weights (SG groups per stage,
// SG*32 B contiguous per row) and activations are staged in shared memory with a cp.async ring, so every
// DRAM request is a >= 64 B row segment instead of one 32 B piece per row (the access pattern that limits the
// direct-fragment kernel in pq2_mma_small.cu to ~420 GB/s on K = 5120 projections). INT8 tensor-core MMA works on
// the raw 2-bit planes (shift + mask, -qsum per group). Optional split-K: every split writes FP32 partials and the
// last CTA to finish a row block sums them in fixed order (deterministic regardless of arrival order) and writes the
// output; counters self-reset. Activations use the prefill fragment layout written by _quantize_prefill.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <int B> __device__ __forceinline__ void cp_async(void* dst, const void* src) {
  if (B == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
  else if (B == 8) asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
  else asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }

template <int WR, int RT, int NT, int SG, int ST>
struct Cfg {
  static constexpr int NTH = WR * 32, BMW = WR * RT * 16, BT = NT * 8;
  static constexpr int WSTRIDE = SG * 32 + (SG >= 4 ? 32 : 0);   // padded: conflict-free A-fragment LDS.64
  static constexpr int QSTRIDE = SG * 128 + 16;
  static constexpr int WC = BMW * WSTRIDE, QB = BT * QSTRIDE, WSB = BMW * SG * 2, XSB = BT * SG * 4, QSB = BT * SG * 4;
  static constexpr int STAGE = ((WC + QB + WSB + XSB + QSB + 15) / 16) * 16;
  static constexpr int SMEM = STAGE * ST;
};

template <typename T, int WR, int RT, int NT, int SG, int ST>
__global__ void __launch_bounds__(WR * 32) ring_kernel(
    const int8_t* __restrict__ q, const float* __restrict__ xs, const int* __restrict__ qsum,
    const uint8_t* __restrict__ codes, const __half* __restrict__ wscale, T* __restrict__ y,
    float* __restrict__ partial, unsigned* __restrict__ counters, int M, int N, int K, int split) {
  using C = Cfg<WR, RT, NT, SG, ST>;
  extern __shared__ __align__(16) uint8_t smem[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gid = lane >> 2, tid = lane & 3;
  const int row0 = blockIdx.x * C::BMW;
  const int G = K >> 7;
  const int per = G / split, g_begin = blockIdx.y * per, iters = per / SG;

  auto load_stage = [&](int s, int it) {
    uint8_t* Wc = smem + s * C::STAGE;
    uint8_t* Q = Wc + C::WC;
    uint8_t* Ws = Q + C::QB;
    uint8_t* Xs = Ws + C::WSB;
    uint8_t* Qs = Xs + C::XSB;
    const int g0 = g_begin + it * SG;
    constexpr int WCH = SG * 32 / 16;
    for (int i = threadIdx.x; i < C::BMW * WCH; i += C::NTH) {
      const int r = i / WCH, c = i % WCH;
      cp_async<16>(Wc + r * C::WSTRIDE + c * 16, codes + (size_t)min(row0 + r, N - 1) * (K >> 2) + g0 * 32 + c * 16);
    }
    constexpr int QCH = SG * 8;
    for (int i = threadIdx.x; i < C::BT * QCH; i += C::NTH) {
      const int t = i / QCH, c = i % QCH;
      cp_async<16>(Q + t * C::QSTRIDE + c * 16, q + (size_t)min(t, M - 1) * K + g0 * 128 + c * 16);
    }
    for (int i = threadIdx.x; i < C::BMW; i += C::NTH)
      cp_async<SG * 2>(Ws + i * SG * 2, wscale + (size_t)min(row0 + i, N - 1) * G + g0);
    for (int i = threadIdx.x; i < C::BT; i += C::NTH) {
      cp_async<SG * 4>(Xs + i * SG * 4, xs + (size_t)min(i, M - 1) * G + g0);
      cp_async<SG * 4>(Qs + i * SG * 4, qsum + (size_t)min(i, M - 1) * G + g0);
    }
  };

  float acc[RT][NT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[r][n][i] = 0.f;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) { if (s < iters) load_stage(s, s); cp_commit(); }

  for (int it = 0; it < iters; ++it) {
    cp_wait<ST - 2>();
    __syncthreads();
    { const int nit = it + ST - 1; if (nit < iters) load_stage(nit % ST, nit); cp_commit(); }
    const uint8_t* Wc = smem + (it % ST) * C::STAGE;
    const uint8_t* Q = Wc + C::WC;
    const __half* Ws = reinterpret_cast<const __half*>(Q + C::QB);
    const float* Xs = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(Ws) + C::WSB);
    const int* Qs = reinterpret_cast<const int*>(reinterpret_cast<const uint8_t*>(Xs) + C::XSB);
#pragma unroll
    for (int gl = 0; gl < SG; ++gl) {
      uint32_t b[NT][8];
      float x0[NT], x1[NT];
      int q0[NT], q1[NT];
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const uint4* bp = reinterpret_cast<const uint4*>(Q + (n * 8 + gid) * C::QSTRIDE + gl * 128 + tid * 32);
        const uint4 u0 = bp[0], u1 = bp[1];
        b[n][0] = u0.x; b[n][1] = u0.y; b[n][2] = u0.z; b[n][3] = u0.w;
        b[n][4] = u1.x; b[n][5] = u1.y; b[n][6] = u1.z; b[n][7] = u1.w;
        const int tc = n * 8 + tid * 2;
        x0[n] = Xs[tc * SG + gl]; x1[n] = Xs[(tc + 1) * SG + gl];
        q0[n] = 0x4B400000 - Qs[tc * SG + gl]; q1[n] = 0x4B400000 - Qs[(tc + 1) * SG + gl];
      }
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const int rl = (warp * RT + r) * 16 + gid;
        const uint2 w0 = *reinterpret_cast<const uint2*>(Wc + rl * C::WSTRIDE + gl * 32 + tid * 8);
        const uint2 w1 = *reinterpret_cast<const uint2*>(Wc + (rl + 8) * C::WSTRIDE + gl * 32 + tid * 8);
        const float ws0 = __half2float(Ws[rl * SG + gl]), ws1 = __half2float(Ws[(rl + 8) * SG + gl]);
        uint32_t a[4][4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          a[s][0] = (w0.x >> (2 * s)) & 0x03030303u; a[s][1] = (w1.x >> (2 * s)) & 0x03030303u;
          a[s][2] = (w0.y >> (2 * s)) & 0x03030303u; a[s][3] = (w1.y >> (2 * s)) & 0x03030303u;
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          int c[4] = {0, 0, 0, 0};
#pragma unroll
          for (int s = 0; s < 4; ++s) mma_s8(c, a[s], b[n][2 * s], b[n][2 * s + 1]);
          constexpr float MAGIC = 12582912.0f;
          acc[r][n][0] += (__int_as_float(c[0] + q0[n]) - MAGIC) * ws0 * x0[n];
          acc[r][n][1] += (__int_as_float(c[1] + q1[n]) - MAGIC) * ws0 * x1[n];
          acc[r][n][2] += (__int_as_float(c[2] + q0[n]) - MAGIC) * ws1 * x0[n];
          acc[r][n][3] += (__int_as_float(c[3] + q1[n]) - MAGIC) * ws1 * x1[n];
        }
      }
    }
  }
  cp_wait<0>();
  const int wrow = row0 + warp * RT * 16;
  if (split == 1) {
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int tk = n * 8 + tid * 2 + j, rr = wrow + r * 16 + gid;
          if (tk < M) {
            if (rr < N) y[(size_t)tk * N + rr] = from_float<T>(acc[r][n][j]);
            if (rr + 8 < N) y[(size_t)tk * N + rr + 8] = from_float<T>(acc[r][n][2 + j]);
          }
        }
    return;
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int tk = n * 8 + tid * 2 + j, rr = wrow + r * 16 + gid;
        if (tk < M) {
          float* p = partial + ((size_t)blockIdx.y * M + tk) * N;
          if (rr < N) p[rr] = acc[r][n][j];
          if (rr + 8 < N) p[rr + 8] = acc[r][n][2 + j];
        }
      }
  __threadfence();
  __syncthreads();
  __shared__ unsigned is_last;
  if (threadIdx.x == 0) is_last = (atomicAdd(&counters[blockIdx.x], 1u) == (unsigned)(split - 1));
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  for (int i = threadIdx.x; i < C::BMW * M; i += C::NTH) {
    const int tk = i / C::BMW, row = row0 + i % C::BMW;
    if (row >= N) continue;
    float a = __ldcg(partial + (size_t)tk * N + row);
    for (int z = 1; z < split; ++z) a += __ldcg(partial + ((size_t)z * M + tk) * N + row);
    y[(size_t)tk * N + row] = from_float<T>(a);
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = 0;
}

template <typename T, int WR, int RT, int NT, int SG, int ST>
static int launch(const void* q, const void* xs, const void* qs, const void* codes, const void* ws, void* y,
                  void* partial, void* counters, int M, int N, int K, int split, cudaStream_t stream) {
  using C = Cfg<WR, RT, NT, SG, ST>;
  if ((K >> 7) % (split * SG) || M > C::BT) return -2;
  auto kern = ring_kernel<T, WR, RT, NT, SG, ST>;
  static const bool ok = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) == cudaSuccess;
  if (!ok) return -4;
  dim3 grid((N + C::BMW - 1) / C::BMW, split), block(C::NTH);
  kern<<<grid, block, C::SMEM, stream>>>((const int8_t*)q, (const float*)xs, (const int*)qs, (const uint8_t*)codes,
                                 (const __half*)ws, (T*)y, (float*)partial, (unsigned*)counters, M, N, K, split);
  return (int)cudaGetLastError();
}

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

template <typename T>
static int ring_dispatch(int cfg, const void* q, const void* xs, const void* qs, const void* codes, const void* ws,
                         void* y, void* partial, void* counters, int M, int N, int K, int split, cudaStream_t s) {
#define R(i, WR, RT, NT, SG, ST) \
  case i: return launch<T, WR, RT, NT, SG, ST>(q, xs, qs, codes, ws, y, partial, counters, M, N, K, split, s);
  switch (cfg) {
    R(0, 4, 1, 1, 4, 3) R(1, 4, 2, 1, 4, 3) R(2, 2, 1, 1, 4, 4) R(3, 4, 1, 1, 2, 4) R(4, 2, 2, 1, 4, 3)
    R(5, 4, 1, 1, 4, 2) R(6, 8, 1, 1, 4, 3) R(7, 2, 1, 1, 2, 3) R(8, 4, 1, 1, 4, 4) R(9, 8, 1, 1, 2, 3)
    R(10, 4, 1, 2, 4, 3) R(11, 4, 2, 2, 4, 2) R(12, 2, 1, 2, 4, 3)
  }
#undef R
  return -3;
}

// y[M, N] (BF16/FP16, written in place). cfg selects the CTA tile (see ring_dispatch); `partial` is an FP32
// [>= split * M * N] workspace and `counters` a zero-initialised int32 array (>= ceil(N / rows per CTA)) that the
// kernel resets itself; both are only touched when split > 1.
void pq2_mma_ring(Tensor q, Tensor xs, Tensor qsum, Tensor weight, Tensor y, Tensor partial, Tensor counters,
                  int64_t cfg, int64_t split) {
  const int32_t device_idx = q.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device_idx);
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &raw_stream));
  const int M = (int)q.sizes()[0], K = (int)q.sizes()[1], N = (int)weight.sizes()[0];
  STD_TORCH_CHECK(M >= 1 && M <= 16, "pq2_mma_ring: M must be 1..16");
  STD_TORCH_CHECK(K % 128 == 0 && weight.sizes()[1] == (int64_t)(K / 128) * 34, "pq2_mma_ring: bad PQ2 weight shape");
  STD_TORCH_CHECK(y.sizes()[0] == M && y.sizes()[1] == N, "pq2_mma_ring: bad output shape");
  STD_TORCH_CHECK(split >= 1 && (split == 1 || (partial.numel() >= split * M * N && counters.numel() > 0)),
                  "pq2_mma_ring: split-K workspace too small");
  STD_TORCH_CHECK(y.scalar_type() == ScalarType::BFloat16 || y.scalar_type() == ScalarType::Half,
                  "pq2_mma_ring: output must be BF16 or FP16");
  const char* codes = (const char*)weight.data_ptr();
  const void* wscale = codes + (size_t)N * K / 4;
  const int err = y.scalar_type() == ScalarType::BFloat16
      ? ring_dispatch<__nv_bfloat16>((int)cfg, q.data_ptr(), xs.data_ptr(), qsum.data_ptr(), codes, wscale,
                                     y.data_ptr(), partial.data_ptr(), counters.data_ptr(), M, N, K, (int)split,
                                     (cudaStream_t)raw_stream)
      : ring_dispatch<__half>((int)cfg, q.data_ptr(), xs.data_ptr(), qsum.data_ptr(), codes, wscale, y.data_ptr(),
                              partial.data_ptr(), counters.data_ptr(), M, N, K, (int)split, (cudaStream_t)raw_stream);
  STD_TORCH_CHECK(err == 0, "pq2_mma_ring: launch failed with code ", err);
}
