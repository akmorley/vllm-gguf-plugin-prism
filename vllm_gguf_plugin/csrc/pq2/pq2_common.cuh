// SPDX-License-Identifier: Apache-2.0
// Device helpers shared by the PQ2 INT8 tensor-core and DP4A kernels.
//
// All of them work on the raw 2-bit planes of the prepared layout: plane s of a packed 32-bit weight word is
// (w >> 2s) & 0x03030303, activations are quantized to INT8 in plane/fragment order, and the ternary -1 offset is
// applied once per 128-weight group as -qsum.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// One m16n8k32 INT8 tensor-core step: c += a (16x32, row) * b (32x8, col).
__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// cp.async of B bytes (4, 8 or 16) global -> shared; 16-byte copies bypass L1.
template <int B>
__device__ __forceinline__ void cp_async(void* dst, const void* src) {
  static_assert(B == 4 || B == 8 || B == 16, "cp.async copies 4, 8 or 16 bytes");
  const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  if constexpr (B == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
  else if constexpr (B == 8) asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(d), "l"(src));
  else asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) { cp_async<16>(dst, src); }
__device__ __forceinline__ void cp_async8(void* dst, const void* src) { cp_async<8>(dst, src); }
__device__ __forceinline__ void cp_async4(void* dst, const void* src) { cp_async<4>(dst, src); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// float -> output type. Written with the explicit intrinsics because the torch build disables the implicit
// half/bfloat16 conversions.
template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }
