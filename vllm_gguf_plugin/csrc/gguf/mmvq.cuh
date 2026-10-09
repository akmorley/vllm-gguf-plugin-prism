// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;
    const auto vec = blockIdx.y;

    if (row >= nrows || vec >= nvecs) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;


    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = vec*(nrows_y/QK8_1) + i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    }

    if (threadIdx.x == 0) {
        dst[vec*nrows + row] = tmp;
    }
}

// Multi-vector dot products for the K-quants that dominate GGUF checkpoints. The generic
// path calls vec_dot once per vector and so re-extracts the weight bits ncols_y times,
// which makes 2-8 vector GEMVs compute-bound; these unpack each weight int once and
// reuse it for every vector. Same arithmetic as vec_dot_q5_K/q6_K_q8_1 (vecdotq.cuh).
struct mmvq_no_multi_dot {};

struct mmvq_q5_K_multi_dot {
    template <int ncols_y>
    static __device__ __forceinline__ void dot(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ y, const int y_stride,
        const int iqs, float (&out)[ncols_y]) {
        const block_q5_K * bq5_K = (const block_q5_K *) vbq;

        const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
        const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
        const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));
        const int vl0 = ql[0];
        const int vl1 = ql[4];
        const int vh0 = qh[0] >> bq8_offset;
        const int vh1 = qh[4] >> bq8_offset;

        const uint16_t * scales = (const uint16_t *)bq5_K->scales;
        uint16_t aux[2];
        const int j = bq8_offset/2;
        if (j < 2) {
            aux[0] = scales[j+0] & 0x3f3f;
            aux[1] = scales[j+2] & 0x3f3f;
        } else {
            aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
            aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
        }
        const uint8_t * sc = (const uint8_t *)aux;
        const uint8_t * m  = sc + 2;

        int v0[QR5_K], v1[QR5_K];
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            v0[i] = ((vl0 >> (4*i)) & 0x0F0F0F0F) | (((vh0 >> i) << 4) & 0x10101010);
            v1[i] = ((vl1 >> (4*i)) & 0x0F0F0F0F) | (((vh1 >> i) << 4) & 0x10101010);
        }
        const float2 dm5f = __half22float2(bq5_K->dm);

#pragma unroll
        for (int c = 0; c < ncols_y; ++c) {
            const block_q8_1 * bq8 = y + c*y_stride + bq8_offset;
            float sumf_d = 0.0f;
            float sumf_m = 0.0f;
#pragma unroll
            for (int i = 0; i < QR5_K; ++i) {
                const int * q8 = (const int *)bq8[i].qs + ((iqs/2)%4);
                const int u0 = q8[0];
                const int u1 = q8[4];
                const float d8 = __low2float(bq8[i].ds);
                const int dot1 = __dp4a(v0[i], u0, __dp4a(v1[i], u1, 0));
                const int dot2 = __dp4a(0x01010101, u0, __dp4a(0x01010101, u1, 0));
                sumf_d += d8 * (dot1 * sc[i]);
                sumf_m += d8 * (dot2 * m[i]);
            }
            out[c] += dm5f.x*sumf_d - dm5f.y*sumf_m;
        }
    }
};

struct mmvq_q6_K_multi_dot {
    template <int ncols_y>
    static __device__ __forceinline__ void dot(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ y, const int y_stride,
        const int iqs, float (&out)[ncols_y]) {
        const block_q6_K * bq6_K = (const block_q6_K *) vbq;

        const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
        const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
        const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

        const int vl = get_int_from_uint8(bq6_K->ql, iqs);
        const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;
        const int8_t * scales = bq6_K->scales + scale_offset;

        int vi[QR6_K], sc[QR6_K];
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            sc[i] = scales[4*i];
            const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
            vi[i] = __vsubss4((vil | vih), 0x20202020);
        }
        const float d = __half2float(bq6_K->d);

#pragma unroll
        for (int c = 0; c < ncols_y; ++c) {
            const block_q8_1 * bq8 = y + c*y_stride + bq8_offset;
            float sumf = 0.0f;
#pragma unroll
            for (int i = 0; i < QR6_K; ++i) {
                const int u = get_int_from_int8_aligned(bq8[2*i].qs, iqs % QI8_1);
                sumf += __low2float(bq8[2*i].ds) * (__dp4a(vi[i], u, 0) * sc[i]);
            }
            out[c] += d*sumf;
        }
    }
};

// Multi-vector variant after later llama.cpp (ggml-cuda/mmvq.cu): one CUDA block computes
// rows_per_block rows for all ncols_y vectors, so each weight block is read from memory once
// and reused from registers/L1 for every vector. Several warps split the K dimension and are
// reduced through shared memory. Unlike llama.cpp (whose tensors are padded) rows past nrows
// are never read.
template <int ncols_y> struct mmvq_multi_cfg {
    static constexpr int nwarps = ncols_y <= 4 ? 4 : 2;
    static constexpr int rows_per_block = ncols_y == 1 ? 1 : 2;
};

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int nwarps, int rows_per_block, typename multi_dot = mmvq_no_multi_dot>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q_multi(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows) {
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const int row0 = rows_per_block*blockIdx.x;
    const int blocks_per_row = ncols / qk;
    const int blocks_per_col_y = (ncols + 512 - 1) / 512 * 512 / QK8_1;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    float tmp[ncols_y][rows_per_block] = {{0.0f}};

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            if (rows_per_block > 1 && row0 + i >= nrows) {
                break;
            }
            if constexpr (std::is_same<multi_dot, mmvq_no_multi_dot>::value) {
#pragma unroll
                for (int j = 0; j < ncols_y; ++j) {
                    tmp[j][i] += vec_dot_q_cuda(&x[(row0 + i)*blocks_per_row + kbx], &y[j*blocks_per_col_y + kby], kqs);
                }
            } else {
                float acc[ncols_y];
#pragma unroll
                for (int j = 0; j < ncols_y; ++j) {
                    acc[j] = 0.0f;
                }
                multi_dot::template dot<ncols_y>(&x[(row0 + i)*blocks_per_row + kbx], &y[kby], blocks_per_col_y, kqs, acc);
#pragma unroll
                for (int j = 0; j < ncols_y; ++j) {
                    tmp[j][i] += acc[j];
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps > 1 ? nwarps - 1 : 1][ncols_y][rows_per_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_block; ++i) {
                tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
#pragma unroll
            for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
                tmp[j][i] += VLLM_SHFL_XOR_SYNC(tmp[j][i], mask);
            }
        }
        if (threadIdx.x < rows_per_block && row0 + threadIdx.x < nrows) {
            dst[j*nrows + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int nwarps, int rows_per_block>
static void launch_mul_mat_vec_q_cfg(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_nums((nrows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);
    using multi_dot = std::conditional_t<std::is_same<block_q_t, block_q5_K>::value, mmvq_q5_K_multi_dot,
                      std::conditional_t<std::is_same<block_q_t, block_q6_K>::value, mmvq_q6_K_multi_dot,
                                         mmvq_no_multi_dot>>;
    mul_mat_vec_q_multi<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, ncols_y, nwarps, rows_per_block, multi_dot>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

// Tall matrices with short rows (the vocabulary head: about 248K rows of 2560) spend most of a
// 4-warp block on the cross-warp reduction; packing 4 rows per block is 28% faster at 1 vector
// and 10-25% at 3-8 (3090 Ti, Q6_K). Other shapes were within noise of the defaults.
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y>
static void launch_mul_mat_vec_q_multi(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    if (nrows >= 65536) {
        constexpr int nwarps = ncols_y <= 2 ? 2 : 1;
        launch_mul_mat_vec_q_cfg<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, ncols_y, nwarps, 4>(vx, vy, dst, ncols, nrows, stream);
        return;
    }
    using cfg = mmvq_multi_cfg<ncols_y>;
    launch_mul_mat_vec_q_cfg<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, ncols_y, cfg::nwarps, cfg::rows_per_block>(vx, vy, dst, ncols, nrows, stream);
}

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static void launch_mul_mat_vec_q(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    switch (nvecs) {
        case 1: launch_mul_mat_vec_q_multi<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 1>(vx, vy, dst, ncols, nrows, stream); return;
        case 2: launch_mul_mat_vec_q_multi<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 2>(vx, vy, dst, ncols, nrows, stream); return;
        case 3: launch_mul_mat_vec_q_multi<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 3>(vx, vy, dst, ncols, nrows, stream); return;
        case 4: launch_mul_mat_vec_q_multi<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 4>(vx, vy, dst, ncols, nrows, stream); return;
        case 5: launch_mul_mat_vec_q_multi<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 5>(vx, vy, dst, ncols, nrows, stream); return;
        case 6: launch_mul_mat_vec_q_multi<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 6>(vx, vy, dst, ncols, nrows, stream); return;
        case 7: launch_mul_mat_vec_q_multi<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 7>(vx, vy, dst, ncols, nrows, stream); return;
        case 8: launch_mul_mat_vec_q_multi<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 8>(vx, vy, dst, ncols, nrows, stream); return;
        default: break;
    }
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q4_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q4_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q5_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q5_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q2_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q3_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq3_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_m_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq4_nl_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq4_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    launch_mul_mat_vec_q<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}
