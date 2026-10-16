// SPDX-License-Identifier: Apache-2.0
// PQ2 x Q8 prefill projection (M >= 64): INT8 tensor-core MMA on the raw 2-bit planes.
//
// Y[t, n] = sum_g ws[n,g] * xs[t,g] * (sum_{i in g} code[n,i] * q[t,i] - qsum[t,g])
//
// Same integer arithmetic and float epilogue order as the Triton _mmq kernel (bit-exact with it), but
// the weight codes are never unpacked: plane s of a packed byte is (w >> 2s) & 0x03030303, one MMA
// k-step per plane, and the -1 ternary offset is applied once per group as -qsum. Weights are the MMA A
// operand (16 rows x 32 k), activations the B operand. Activations use the prefill fragment layout
// written by quantize_prefill (pq2_mmq.py): per token per group, lane `tid` of the MMA owns the 32
// contiguous bytes [s*8 + h*4 + j] = q[4 * (8*tid + 4*h + j) + s], so a B fragment is two LDS.128.
// CTA tile: (WR*RT*16 weight rows) x (WT*NT*8 tokens); cp.async multi-stage pipeline in shared memory.
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
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }

template <int WR, int WT, int RT, int NT, int SG, int ST>
struct Cfg {
  static constexpr int NWARP = WR * WT, NTH = NWARP * 32;
  static constexpr int BMW = WR * RT * 16, BT = WT * NT * 8;
  static constexpr int WSTRIDE = SG * 32, QSTRIDE = SG * 128 + 16;
  static constexpr int WC_BYTES = BMW * WSTRIDE, Q_BYTES = BT * QSTRIDE;
  static constexpr int WS_BYTES = BMW * SG * 2, XS_BYTES = BT * SG * 4, QS_BYTES = BT * SG * 4;
  static constexpr int STAGE = ((WC_BYTES + Q_BYTES + WS_BYTES + XS_BYTES + QS_BYTES + 15) / 16) * 16;
  static constexpr int SMEM = STAGE * ST;
};

template <typename T, int WR, int WT, int RT, int NT, int SG, int ST, int MINB>
__global__ void __launch_bounds__(WR * WT * 32, MINB) pq2_mma_prefill_kernel(
    const int8_t* __restrict__ q, const float* __restrict__ xs, const int* __restrict__ qsum,
    const uint8_t* __restrict__ codes, const __half* __restrict__ wscale, T* __restrict__ y, int M, int N, int K) {
  using C = Cfg<WR, WT, RT, NT, SG, ST>;
  extern __shared__ __align__(16) uint8_t smem[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gid = lane >> 2, tid = lane & 3;
  const int wr = warp % WR, wt = warp / WR;
  const int row0 = blockIdx.x * C::BMW, tok0 = blockIdx.y * C::BT;
  const int G = K >> 7, iters = G / SG;

  auto stage_ptr = [&](int s) { return smem + s * C::STAGE; };
  auto load_stage = [&](int s, int it) {
    uint8_t* base = stage_ptr(s);
    uint8_t* Wc = base;
    uint8_t* Q = Wc + C::WC_BYTES;
    uint8_t* Ws = Q + C::Q_BYTES;
    uint8_t* Xs = Ws + C::WS_BYTES;
    uint8_t* Qs = Xs + C::XS_BYTES;
    const int g0 = it * SG;
    constexpr int WCH = C::WSTRIDE / 16;
    for (int i = threadIdx.x; i < C::BMW * WCH; i += C::NTH) {
      const int r = i / WCH, c = i % WCH;
      const int gr = min(row0 + r, N - 1);
      cp_async16(Wc + r * C::WSTRIDE + c * 16, codes + (size_t)gr * (K >> 2) + g0 * 32 + c * 16);
    }
    constexpr int QCH = SG * 8;
    for (int i = threadIdx.x; i < C::BT * QCH; i += C::NTH) {
      const int t = i / QCH, c = i % QCH;
      const int gt = min(tok0 + t, M - 1);
      cp_async16(Q + t * C::QSTRIDE + c * 16, q + (size_t)gt * K + g0 * 128 + c * 16);
    }
    for (int i = threadIdx.x; i < C::BMW; i += C::NTH) {
      const int gr = min(row0 + i, N - 1);
      static_assert(SG == 1 || SG == 2, "scale copy supports SG 1 or 2");
      if (SG == 2) cp_async4(Ws + i * SG * 2, wscale + (size_t)gr * G + g0);
      else *reinterpret_cast<__half*>(Ws + i * 2) = wscale[(size_t)gr * G + g0];
    }
    for (int i = threadIdx.x; i < C::BT; i += C::NTH) {
      const int gt = min(tok0 + i, M - 1);
      if (SG == 2) {
        cp_async8(Xs + i * SG * 4, xs + (size_t)gt * G + g0);
        cp_async8(Qs + i * SG * 4, qsum + (size_t)gt * G + g0);
      } else {
        *reinterpret_cast<float*>(Xs + i * 4) = xs[(size_t)gt * G + g0];
        *reinterpret_cast<int*>(Qs + i * 4) = qsum[(size_t)gt * G + g0];
      }
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
  for (int s = 0; s < ST - 1; ++s) {
    if (s < iters) load_stage(s, s);
    cp_commit();
  }

  for (int it = 0; it < iters; ++it) {
    cp_wait<ST - 2>();
    __syncthreads();
    {
      const int nit = it + ST - 1;
      if (nit < iters) load_stage(nit % ST, nit);
      cp_commit();
    }
    const uint8_t* base = stage_ptr(it % ST);
    const uint8_t* Wc = base;
    const uint8_t* Q = Wc + C::WC_BYTES;
    const __half* Ws = reinterpret_cast<const __half*>(Q + C::Q_BYTES);
    const float* Xs = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(Ws) + C::WS_BYTES);
    const int* Qs = reinterpret_cast<const int*>(reinterpret_cast<const uint8_t*>(Xs) + C::XS_BYTES);
#pragma unroll
    for (int gl = 0; gl < SG; ++gl) {
      uint32_t a[RT][4][4];
      float ws0[RT], ws1[RT];
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const int rl = (wr * RT + r) * 16 + gid;
        const uint2 w0 = *reinterpret_cast<const uint2*>(Wc + rl * C::WSTRIDE + gl * 32 + tid * 8);
        const uint2 w1 = *reinterpret_cast<const uint2*>(Wc + (rl + 8) * C::WSTRIDE + gl * 32 + tid * 8);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          a[r][s][0] = (w0.x >> (2 * s)) & 0x03030303u;
          a[r][s][1] = (w1.x >> (2 * s)) & 0x03030303u;
          a[r][s][2] = (w0.y >> (2 * s)) & 0x03030303u;
          a[r][s][3] = (w1.y >> (2 * s)) & 0x03030303u;
        }
        ws0[r] = __half2float(Ws[rl * SG + gl]);
        ws1[r] = __half2float(Ws[(rl + 8) * SG + gl]);
      }
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const int tl = (wt * NT + n) * 8;
        uint32_t b[8];
        {
          const uint4* bp = reinterpret_cast<const uint4*>(Q + (tl + gid) * C::QSTRIDE + gl * 128 + tid * 32);
          const uint4 u0 = bp[0], u1 = bp[1];
          b[0] = u0.x; b[1] = u0.y; b[2] = u0.z; b[3] = u0.w;
          b[4] = u1.x; b[5] = u1.y; b[6] = u1.z; b[7] = u1.w;
        }
        const int tc0 = tl + tid * 2;
        const float x0 = Xs[tc0 * SG + gl], x1 = Xs[(tc0 + 1) * SG + gl];
        const int q0 = 0x4B400000 - Qs[tc0 * SG + gl], q1 = 0x4B400000 - Qs[(tc0 + 1) * SG + gl];
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          int c[4] = {0, 0, 0, 0};
#pragma unroll
          for (int s = 0; s < 4; ++s) mma_s8(c, a[r][s], b[2 * s], b[2 * s + 1]);
          constexpr float MAGIC = 12582912.0f;
          acc[r][n][0] = fmaf((__int_as_float(c[0] + q0) - MAGIC) * x0, ws0[r], acc[r][n][0]);
          acc[r][n][1] = fmaf((__int_as_float(c[1] + q1) - MAGIC) * x1, ws0[r], acc[r][n][1]);
          acc[r][n][2] = fmaf((__int_as_float(c[2] + q0) - MAGIC) * x0, ws1[r], acc[r][n][2]);
          acc[r][n][3] = fmaf((__int_as_float(c[3] + q1) - MAGIC) * x1, ws1[r], acc[r][n][3]);
        }
      }
    }
  }
  cp_wait<0>();
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    const int rr = row0 + (wr * RT + r) * 16 + gid;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int tc = tok0 + (wt * NT + n) * 8 + tid * 2;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        if (tc + j < M) {
          if (rr < N) y[(size_t)(tc + j) * N + rr] = from_float<T>(acc[r][n][j]);
          if (rr + 8 < N) y[(size_t)(tc + j) * N + rr + 8] = from_float<T>(acc[r][n][2 + j]);
        }
      }
    }
  }
}

template <typename T, int WR, int WT, int RT, int NT, int SG, int ST, int MINB = 1>
static int launch_cfg(const void* q, const void* xs, const void* qsum, const void* codes, const void* ws, void* y,
                      int M, int N, int K, cudaStream_t s) {
  using C = Cfg<WR, WT, RT, NT, SG, ST>;
  auto kern = pq2_mma_prefill_kernel<T, WR, WT, RT, NT, SG, ST, MINB>;
  static const bool configured = [] {
    return cudaFuncSetAttribute(pq2_mma_prefill_kernel<T, WR, WT, RT, NT, SG, ST, MINB>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) == cudaSuccess;
  }();
  if (!configured) return -4;
  dim3 grid((N + C::BMW - 1) / C::BMW, (M + C::BT - 1) / C::BT), block(C::NTH);
  kern<<<grid, block, C::SMEM, s>>>((const int8_t*)q, (const float*)xs, (const int*)qsum, (const uint8_t*)codes,
                                    (const __half*)ws, (T*)y, M, N, K);
  return (int)cudaGetLastError();
}

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

template <typename T>
static int pq2_prefill_dispatch(int cfg, const void* q, const void* xs, const void* qsum, const void* codes,
                                const void* ws, void* y, int M, int N, int K, cudaStream_t s) {
  switch (cfg) {
    case 0: return launch_cfg<T, 4, 2, 2, 8, 2, 2>(q, xs, qsum, codes, ws, y, M, N, K, s);  // 128 rows x 128 tokens
    case 1: return launch_cfg<T, 4, 2, 4, 4, 2, 2>(q, xs, qsum, codes, ws, y, M, N, K, s);  // 256 rows x 64 tokens
  }
  return -3;
}

// y[M, N] (BF16/FP16, written in place) from prefill-layout Q8 activations and prepared PQ2 planes.
// cfg 0: 128 x 128 tile (wide outputs); cfg 1: 256 x 64 tile (N = 5120 outputs). K must be a multiple of 256.
void pq2_mma_prefill(Tensor q, Tensor xs, Tensor qsum, Tensor weight, Tensor y, int64_t cfg) {
  const int32_t device_idx = q.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device_idx);
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &raw_stream));
  const int M = (int)q.sizes()[0], K = (int)q.sizes()[1], N = (int)weight.sizes()[0];
  STD_TORCH_CHECK(K % 256 == 0, "pq2_mma_prefill: K must be a multiple of 256");
  STD_TORCH_CHECK(weight.sizes()[1] == (int64_t)(K / 128) * 34, "pq2_mma_prefill: bad PQ2 weight shape");
  STD_TORCH_CHECK(y.sizes()[0] == M && y.sizes()[1] == N, "pq2_mma_prefill: bad output shape");
  STD_TORCH_CHECK(y.scalar_type() == ScalarType::BFloat16 || y.scalar_type() == ScalarType::Half,
                  "pq2_mma_prefill: output must be BF16 or FP16");
  const char* codes = (const char*)weight.data_ptr();
  const void* wscale = codes + (size_t)N * K / 4;
  const int err = y.scalar_type() == ScalarType::BFloat16
                      ? pq2_prefill_dispatch<__nv_bfloat16>((int)cfg, q.data_ptr(), xs.data_ptr(), qsum.data_ptr(),
                                                            codes, wscale, y.data_ptr(), M, N, K,
                                                            (cudaStream_t)raw_stream)
                      : pq2_prefill_dispatch<__half>((int)cfg, q.data_ptr(), xs.data_ptr(), qsum.data_ptr(),
                                                     codes, wscale, y.data_ptr(), M, N, K,
                                                     (cudaStream_t)raw_stream);
  STD_TORCH_CHECK(err == 0, "pq2_mma_prefill: launch failed with code ", err);
}
