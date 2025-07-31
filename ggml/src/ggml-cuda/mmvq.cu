#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"
#include "mmv.cuh"

#include <cstdint>

#define ROWS_PER_BLOCK_1 1
#define ROWS_PER_BLOCK_2 2
#define ROWS_PER_BLOCK_4 4

#define DEFAULT_REDUCE_WARP_SIZE0 16
#define DEFAULT_REDUCE_WARP_SIZE1 32

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA2) || defined(RDNA3) || defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

static constexpr __host__ __device__ int calc_nwarps(int ncols_dst,  mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    return 1;
}

int calc_rows_per_block(const int nrows_x, int ncols_dst) {
    switch (ncols_dst) {
        case 1:
            if (nrows_x >= 27648) {
                return ROWS_PER_BLOCK_4;
            }
            if (nrows_x >= 5120) {
                return ROWS_PER_BLOCK_2;
            }
            return ROWS_PER_BLOCK_1;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
            return ROWS_PER_BLOCK_2;
        default:
            return ROWS_PER_BLOCK_1;
    }
    return ROWS_PER_BLOCK_1;
}

template <int nwarps, int ncols_dst, int warp_size, int rows_per_cuda_block, op_another opins>
__device__ __forceinline__ void reduce_write(float * __restrict__ dst, float *tmp, const int row0, const int sample_dst, 
        const int stride_sample_dst, const int channel_dst, const int stride_channel_dst, const int stride_col_dst, float *nextsrc0, float *nextsrc1) {
    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j*rows_per_cuda_block+i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    //dst += sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) {
                tmp[j*rows_per_cuda_block+i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j*rows_per_cuda_block+i] = warp_reduce_sum<warp_size>(tmp[j*rows_per_cuda_block+i]);
        }

        if (threadIdx.x < rows_per_cuda_block &&
            (rows_per_cuda_block == 1 || row0 + int(threadIdx.x) < stride_col_dst)) {
            //dst[j * stride_col_dst + threadIdx.x] = tmp[j*rows_per_cuda_block+threadIdx.x];
            //dst[j * stride_col_dst + threadIdx.x] = nextsrc1 == nullptr ? tmp[j*rows_per_cuda_block+threadIdx.x] : tmp[j*rows_per_cuda_block+threadIdx.x] + nextsrc1[sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row0 + j * stride_col_dst + threadIdx.x];
            opins(dst, sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row0 + j * stride_col_dst + threadIdx.x, tmp[j*rows_per_cuda_block+threadIdx.x], nextsrc0, nextsrc1);
        }
    }
}

#define mul_mat_vec_q_common_params const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, float * __restrict__ dst, \
        const int ncols_x, const int nchannels_y, const int stride_row_x, const int stride_col_y, const int stride_col_dst, \
        const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst, \
        const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst

#define mul_mat_vec_q_common_variable constexpr int qk  = ggml_cuda_type_traits<type>::qk; \
    constexpr int qi  = ggml_cuda_type_traits<type>::qi; \
    constexpr int vdr = get_vdr_mmvq(type); \
    constexpr mmvq_parameter_table_id table_id = get_device_table_id(); \
    constexpr int nwarps = calc_nwarps(ncols_dst, table_id); \
    const     int tid = warp_size*threadIdx.y + threadIdx.x; \
    const     int row0 = rows_per_cuda_block*blockIdx.x; \
    const     int blocks_per_row_x = ncols_x / qk; \
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi; \
    const int channel_dst = blockIdx.y; \
    const int channel_x   = ncols_dst == 1 && ids ? ids[channel_dst] : channel_dst / channel_ratio; \
    const int channel_y   = ncols_dst == 1 && ids ? channel_dst % nchannels_y : channel_dst; \
    const int sample_dst  = blockIdx.z; \
    const int sample_x    = sample_dst / sample_ratio; \
    const int sample_y    = sample_dst; \
    float tmp[ncols_dst*rows_per_cuda_block] = {0.0f}; \
    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y; \
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

template <ggml_type type, int ncols_dst, int warp_size, int rows_per_cuda_block, op_another opins>
// tell the compiler to use as many registers as it wants, see nwarps definition below
//__launch_bounds__(calc_nwarps(ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(mul_mat_vec_q_common_params, float *nextsrc0, float *nextsrc1) {
    mul_mat_vec_q_common_variable

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

        for (int j = 0; j < ncols_dst; ++j) {
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j*rows_per_cuda_block+i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
            }
        }
    }

    reduce_write<nwarps, ncols_dst, warp_size, rows_per_cuda_block, opins>(dst, tmp, row0, sample_dst, stride_sample_dst, channel_dst, stride_channel_dst, stride_col_dst, nextsrc0, nextsrc1);
}

template <ggml_type type, int ncols_dst, int warp_size, int rows_per_cuda_block, op_another opins>
static __global__ void mul_mat_vec_q_Q2_K(mul_mat_vec_q_common_params, float *nextsrc0, float *nextsrc1){
    mul_mat_vec_q_common_variable

    v4u32 yBase;
    yBase.x  = (unsigned) (unsigned long long) (y);
    yBase.y  = (unsigned) ((unsigned long long) (y) >> 32);
    yBase.zw = -1u;
    const int qs_offset_blkq81 = 4; // qs在block_q8_1的偏移字节数
    const int qs_offset_blkq2k = 16; // qs在block_q2_K的偏移字节数
    const int dm_offset_blkq2k = 80; // dm在block_q2_K的偏移字节数
    const int sc_offset_blkq2k = 0; // scales在block_q2_K的偏移字节数

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));
        const int bq8_offset = QR2_K * (kqs / QI8_1);

        for (int j = 0; j < ncols_dst; ++j) {
            int    u[QR2_K];
            float d8[QR2_K];

#pragma unroll
            for (int i = 0; i < QR2_K; ++ i) {
                //d8[i] = __low2float(y[j*stride_col_y + kby + bq8_offset + i].ds);
                int tmp = (__ivcorex_ml_mem_load_i32(yBase, (j*stride_col_y + kby + bq8_offset + i) * sizeof(block_q8_1), 0, 0));
                //u[i]  = get_int_b4(y[j*stride_col_y + kby + bq8_offset + i].qs, kqs % QI8_1);
                u[i] = __ivcorex_ml_mem_load_i32(yBase, (j*stride_col_y + kby + bq8_offset + i) * sizeof(block_q8_1), kqs % QI8_1 * 4 + qs_offset_blkq81, 0);
                d8[i] = __low2float(*((ggml_half2 *)(&tmp)));
            }

            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const block_q2_K * bq2_K = (const block_q2_K *) vx + (kbx_offset + i*stride_row_x);
                v4u32 bq2kBase;
                bq2kBase.x  = (unsigned) (unsigned long long) (bq2_K);
                bq2kBase.y  = (unsigned) ((unsigned long long) (bq2_K) >> 32);
                bq2kBase.zw = -1u;
                
                const int scale_offset = kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1/2);
                
                int s_c = scale_offset / 4;
                int s_y = scale_offset % 4;

                int32_t knn[2];
                knn[0] = __ivcorex_ml_mem_load_i32(bq2kBase, kbx*sizeof(block_q2_K) + s_c*4, sc_offset_blkq2k + 0, 0);
                knn[1] = __ivcorex_ml_mem_load_i32(bq2kBase, kbx*sizeof(block_q2_K) + s_c*4, sc_offset_blkq2k + 4, 0);
                const int v = __ivcorex_ml_mem_load_i32(bq2kBase, kbx*sizeof(block_q2_K) + kqs*4, qs_offset_blkq2k, 0);
                int dmtmp = __ivcorex_ml_mem_load_i32(bq2kBase, kbx*sizeof(block_q2_K), dm_offset_blkq2k, 0);
                
                float sumf_d = 0.0f;
                float sumf_m = 0.0f;
                int sc = 0;
#pragma unroll
                for (int i = 0; i < QR2_K/2; ++i) {
                    sc = (*(uint32_t *)(knn))>> ((s_y + 2 * i) * 8) & 0x000000ff;

                    const int vi = (v >> (2*i)) & 0x03030303;

                    sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

                    // fill int with 4x m
                    int m = sc >> 4;
                    m |= m <<  8;
                    m |= m << 16;
                    sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
                }
#pragma unroll
                for (int i = QR2_K/2; i < QR2_K; ++i) {
                    sc = (*(uint32_t *)(knn+1)) >> ((s_y + 2 * (i - 2)) * 8) & 0x000000ff;

                    const int vi = (v >> (2*i)) & 0x03030303;

                    sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

                    // fill int with 4x m
                    int m = sc >> 4;
                    m |= m <<  8;
                    m |= m << 16;
                    sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
                }

                const float2 dm2f = __half22float2(*((ggml_half2 *)(&dmtmp)));

                tmp[j*rows_per_cuda_block+i] += (dm2f.x*sumf_d - dm2f.y*sumf_m);
            }
        }
    }

    reduce_write<nwarps, ncols_dst, warp_size, rows_per_cuda_block, opins>(dst, tmp, row0, sample_dst, stride_sample_dst, channel_dst, stride_channel_dst, stride_col_dst, nextsrc0, nextsrc1);
}

template <ggml_type type, int ncols_dst, int warp_size, int rows_per_cuda_block, op_another opins>
static __global__ void mul_mat_vec_q_Q3_K(mul_mat_vec_q_common_params, float *nextsrc0, float *nextsrc1){
    mul_mat_vec_q_common_variable

    const int qs_offset_blkq81 = 4; // qs在block_q8_1的偏移字节数
    const int d_offset_blkq3k = 108; // d在block_q3_K的偏移字节数
    const int hmask_offset_blkq3k = 0; // hmask在block_q3_K的偏移字节数
    const int qs_offset_blkq3k = 32; // qs在block_q3_K的偏移字节数
    const int scales_offset_blkq3k = 96; // scales在block_q3_K的偏移字节数

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));
        const int bq8_offset = QR3_K * (kqs / (QI3_K/2));
        v4u32 yBase;
        yBase.x  = (unsigned) (unsigned long long) (y+(kby + bq8_offset));
        yBase.y  = (unsigned) ((unsigned long long) (y+(kby + bq8_offset)) >> 32);
        yBase.zw = -1u;

        for (int j = 0; j < ncols_dst; ++j) {
            int u[QR3_K];
            float d8[QR3_K];

#pragma unroll
            for (int i = 0; i < QR3_K; ++i) {
                int tmp = __ivcorex_ml_mem_load_i32(yBase, (j*stride_col_y + i) * sizeof(block_q8_1), 0, 0);
                u[i] = __ivcorex_ml_mem_load_i32(yBase, (j*stride_col_y + i) * sizeof(block_q8_1), kqs % QI8_1 * 4 + qs_offset_blkq81, 0);
                
                d8[i] = __low2float(*((ggml_half2 *)(&tmp)));
            }

            for (int i = 0; i < rows_per_cuda_block; ++i) {
                {
                    //const block_q3_K * bq3_K = (const block_q3_K *) vx + (kbx_offset + i*stride_row_x + kbx);
                    const block_q3_K * bq3_K = (const block_q3_K *) vx + (kbx_offset + i*stride_row_x);
                    v4u32 bq3_KBase;
                    bq3_KBase.x  = (unsigned) (unsigned long long) (bq3_K);
                    bq3_KBase.y  = (unsigned) ((unsigned long long) (bq3_K) >> 32);
                    bq3_KBase.zw = -1u;

                    short sclow_i_00 = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K), scales_offset_blkq3k, 0);
                    short sclow_i_01 = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K), scales_offset_blkq3k + 2, 0);
                    short sclow_i_10 = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K), scales_offset_blkq3k + 4, 0);
                    short sclow_i_11 = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K), scales_offset_blkq3k + 4 + 2, 0);
                    short schigh_i0 = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K), scales_offset_blkq3k + 8, 0);
                    short schigh_i1 = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K), scales_offset_blkq3k + 8 + 2, 0);
                    
                    short vl0 = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K) + kqs*4, qs_offset_blkq3k, 0);
                    short vl1 = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K) + kqs*4, qs_offset_blkq3k + 2, 0);

                    short vh0 = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K) + (kqs % (QI3_K / 2))*4, hmask_offset_blkq3k, 0);
                    short vh1 = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K) + (kqs % (QI3_K / 2))*4, hmask_offset_blkq3k + 2, 0);

                    short dtmp = __ivcorex_ml_mem_load_i16(bq3_KBase, kbx*sizeof(block_q3_K), d_offset_blkq3k, 0);


                    float sumf = 0.0f;
                    const int scale_offset = kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1/2);
                    int s_y = scale_offset % 4;

                    int32_t sclow_i[2];
                    sclow_i[0] = (*((uint16_t *) (&sclow_i_00))) << 0;
                    sclow_i[0] |= (uint32_t((*(uint16_t *) (&sclow_i_01)))) << 16;

                    sclow_i[1] = (*((uint16_t *) (&sclow_i_10))) << 0;
                    sclow_i[1] |= (uint32_t((*(uint16_t *) (&sclow_i_11)))) << 16;
                    
                    int schigh_i = (*((uint16_t *) (&schigh_i0))) << 0;
                    schigh_i |= (uint32_t((*(uint16_t *) (&schigh_i1)))) << 16;

                    int vl = (*((uint16_t *) (&vl0))) << 0;
                    vl |= (uint32_t((*(uint16_t *) (&vl1)))) << 16;

                    int vh = (*((uint16_t *) (&vh0))) << 0;
                    vh |= (uint32_t((*(uint16_t *) (&vh1)))) << 16;
                    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
                    vh = ~vh >> bq8_offset;

                    uint32_t sclowtmp = *((uint32_t *)(sclow_i+0));
                    uint32_t schightmp = *((uint32_t *)(&schigh_i));
#pragma unroll
                    for (int i = 0; i < QR3_K/2; ++i) {
                        const int isc = scale_offset + 2*i;

                        //const int isc_low = isc % (QK_K/32);
                        const int sc_shift_low = 4 * (isc / (QK_K/32));
                        const int sc_low  = ((uint8_t((sclowtmp>>(s_y+2*i)*8)&0x000000ff)) >> sc_shift_low) & 0xF;

                        //const int isc_high = isc % (QK_K/64);
                        const int sc_shift_high = 2 * (isc / (QK_K/64));
                        const int sc_high = (((uint8_t((schightmp>>(s_y+i%2*2)*8)&0x000000ff)) >> sc_shift_high) & 3) << 4;

                        const int sc = (sc_low | sc_high) - 32;

                        const int vil = (vl >> (2*i)) & 0x03030303;

                        const int vih = ((vh >> i) << 2) & 0x04040404;

                        const int vi = __vsubss4(vil, vih);

                        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
                    }

                    sclowtmp = *((uint32_t *)(sclow_i+1));
#pragma unroll
                    for (int i = QR3_K/2; i < QR3_K; ++i) {
                        const int isc = scale_offset + 2*i;

                        //const int isc_low = isc % (QK_K/32);
                        const int sc_shift_low = 4 * (isc / (QK_K/32));
                        const int sc_low  = ((uint8_t((sclowtmp>>(s_y+2*(i-2))*8)&0x000000ff)) >> sc_shift_low) & 0xF;

                        //const int isc_high = isc % (QK_K/64);
                        const int sc_shift_high = 2 * (isc / (QK_K/64));
                        const int sc_high = (((uint8_t((schightmp>>(s_y+(i-2)%2*2)*8)&0x000000ff)) >> sc_shift_high) & 3) << 4;

                        const int sc = (sc_low | sc_high) - 32;

                        const int vil = (vl >> (2*i)) & 0x03030303;

                        const int vih = ((vh >> i) << 2) & 0x04040404;

                        const int vi = __vsubss4(vil, vih);

                        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
                    }

                    const float d = *((ggml_half *)(&dtmp));
                    tmp[j*rows_per_cuda_block+i] += d * sumf;
                }
            }
        }
    }

    reduce_write<nwarps, ncols_dst, warp_size, rows_per_cuda_block, opins>(dst, tmp, row0, sample_dst, stride_sample_dst, channel_dst, stride_channel_dst, stride_col_dst, nextsrc0, nextsrc1);
}

__device__ __forceinline__ float vec_dot_q4_K_q8_1_new(const void * __restrict__ vbq, const int & kbx2, const int & kbx, const int & iqs, const int *u, const int *tmpd8, const int bq8_offset) {
    const int qs_offset_bq4k = 16;
    const int dm_offset_bq4k = 0;
    const int sc_offset_bq4k = 4;

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx2;
    v4u32 bq4KBase;
    bq4KBase.x  = (unsigned) (unsigned long long) (bq4_K);
    bq4KBase.y  = (unsigned) ((unsigned long long) (bq4_K) >> 32);
    bq4KBase.zw = -1u;

    const int j = bq8_offset/2;
    short tmpsc0_ = __ivcorex_ml_mem_load_i16(bq4KBase, kbx*sizeof(block_q4_K) + (j + 0)*2, sc_offset_bq4k, 0);
    short tmpsc1_ = __ivcorex_ml_mem_load_i16(bq4KBase, kbx*sizeof(block_q4_K) + (j + 2)*2, sc_offset_bq4k, 0);
    short tmpsc2_ = __ivcorex_ml_mem_load_i16(bq4KBase, kbx*sizeof(block_q4_K) + (j - 2)*2, sc_offset_bq4k, 0);
    int tmpdm_ = __ivcorex_ml_mem_load_i32(bq4KBase, kbx*sizeof(block_q4_K), dm_offset_bq4k, 0);

    int v[2];
    v[0] = __ivcorex_ml_mem_load_i32(bq4KBase, kbx*sizeof(block_q4_K) + (16 * bq8_offset + 4 * ((iqs/2)%4)), qs_offset_bq4k + 0, 0);
    v[1] = __ivcorex_ml_mem_load_i32(bq4KBase, kbx*sizeof(block_q4_K) + (16 * bq8_offset + 4 * ((iqs/2)%4)), qs_offset_bq4k + 16, 0);

    float d8[QR4_K];
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        d8[i] = __low2float(*((ggml_half2 *)(&tmpd8[i])));
    }

    uint16_t aux[2];
    uint16_t tmpsc0 = *((uint16_t *)(&tmpsc0_));
    uint16_t tmpsc1 = *((uint16_t *)(&tmpsc1_));
    uint16_t tmpsc2 = *((uint16_t *)(&tmpsc2_));

    if (j < 2) {
        aux[0] = tmpsc0 & 0x3f3f;
        aux[1] = tmpsc1 & 0x3f3f;
    } else {
        aux[0] = ((tmpsc1 >> 0) & 0x0f0f) | ((tmpsc2 & 0xc0c0) >> 2);
        aux[1] = ((tmpsc1 >> 4) & 0x0f0f) | ((tmpsc0 & 0xc0c0) >> 2);
    }
    
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    ggml_half2 tmpdm = *((ggml_half2 *)(&tmpdm_));

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, tmpdm, d8);
}

template <ggml_type type, int ncols_dst, int warp_size, int rows_per_cuda_block, op_another opins>
static __global__ void mul_mat_vec_q_Q4_K(mul_mat_vec_q_common_params, float *nextsrc0, float *nextsrc1) {
    mul_mat_vec_q_common_variable

    const int ds_offset_blkq81 = 0;
    const int qs_offset_blkq81 = 4;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));
        const int bq8_offset = QR4_K * ((kqs/2) / (QI8_1/2));

        for (int j = 0; j < ncols_dst; ++j) {
            int    u[2*QR4_K];
            int tmpd8[QR4_K];

            v4u32 yBase;
            yBase.x  = (unsigned) (unsigned long long) ((y + j*stride_col_y + kby) + bq8_offset);
            yBase.y  = (unsigned) ((unsigned long long) ((y + j*stride_col_y + kby) + bq8_offset) >> 32);
            yBase.zw = -1u;

#pragma unroll
            for (int it = 0; it < QR4_K; ++it) {
                tmpd8[it]  = __ivcorex_ml_mem_load_i32(yBase, 0, it* sizeof(block_q8_1) + ds_offset_blkq81, 0);

                u[2*it+0] = __ivcorex_ml_mem_load_i32(yBase, 0 + ((kqs/2)%4) * 4 + 0, it* sizeof(block_q8_1) + qs_offset_blkq81, 0);
                u[2*it+1] = __ivcorex_ml_mem_load_i32(yBase, 0 + ((kqs/2)%4) * 4 + 16, it* sizeof(block_q8_1) + qs_offset_blkq81, 0);
            }

            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j*rows_per_cuda_block+i] += vec_dot_q4_K_q8_1_new(vx, kbx_offset + i*stride_row_x, kbx, kqs, u, tmpd8, bq8_offset);
            }
        }
    }

    reduce_write<nwarps, ncols_dst, warp_size, rows_per_cuda_block, opins>(dst, tmp, row0, sample_dst, stride_sample_dst, channel_dst, stride_channel_dst, stride_col_dst, nextsrc0, nextsrc1);
}

template <ggml_type type, int ncols_dst, int warp_size, int rows_per_cuda_block, op_another opins>
static __global__ void mul_mat_vec_q_Q6_K(mul_mat_vec_q_common_params, float *nextsrc0, float *nextsrc1) {
    mul_mat_vec_q_common_variable

    const int ds_offset_blkq81 = 0;
    const int qs_offset_blkq81 = 4;
    const int ql_offset_blkq6k = 0;
    const int qh_offset_blkq6k = 128;
    const int sc_offset_blkq6k = 192;
    const int d_offset_blkq6k = 208;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));
        const int bq8_offset = 2 * QR6_K * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/4);

        for (int j = 0; j < ncols_dst; ++j) {            
            int    u[QR6_K];
            float d8[QR6_K];
            int tmpd8_[QR6_K];

            v4u32 yBase;
            yBase.x  = (unsigned) (unsigned long long) (y+j*stride_col_y + kby + bq8_offset);
            yBase.y  = (unsigned) ((unsigned long long) (y+j*stride_col_y + kby + bq8_offset) >> 32);
            yBase.zw = -1u;

#pragma unroll
            for (int i = 0; i < QR6_K; ++i) {
                tmpd8_[i] = __ivcorex_ml_mem_load_i32(yBase, 0, (2 * i )* sizeof(block_q8_1) + ds_offset_blkq81,0);
                u[i]  = __ivcorex_ml_mem_load_i32(yBase, kqs % QI8_1 * 4, (2 * i )* sizeof(block_q8_1) + qs_offset_blkq81, 0);
            }

            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const block_q6_K * bq6_K = (const block_q6_K *) vx + (kbx_offset + i*stride_row_x );
                v4u32 bq6KBase;
                bq6KBase.x  = (unsigned) (unsigned long long) (bq6_K);
                bq6KBase.y  = (unsigned) ((unsigned long long) (bq6_K) >> 32);
                bq6KBase.zw = -1u;

                short vl0 = __ivcorex_ml_mem_load_i16(bq6KBase, kbx*sizeof(block_q6_K) + kqs*4, ql_offset_blkq6k, 0);
                short vl1 = __ivcorex_ml_mem_load_i16(bq6KBase, kbx*sizeof(block_q6_K) + kqs*4 + 2, ql_offset_blkq6k, 0);

                short vh0 = __ivcorex_ml_mem_load_i16(bq6KBase, kbx*sizeof(block_q6_K) +((QI6_K / 4) * (kqs / (QI6_K / 2)) + kqs % (QI6_K / 4))*4, qh_offset_blkq6k, 0);
                short vh1 = __ivcorex_ml_mem_load_i16(bq6KBase, kbx*sizeof(block_q6_K) + ((QI6_K / 4) * (kqs / (QI6_K / 2)) + kqs % (QI6_K / 4))*4 + 2, qh_offset_blkq6k, 0);

                char tmpsc[QR6_K];
                const int scale_offset = (QI6_K/4) * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/8);

#pragma unroll
                for (int i = 0; i < QR6_K; ++i) {
                    tmpsc[i] = __ivcorex_ml_mem_load_i8(bq6KBase, kbx*sizeof(block_q6_K) + scale_offset, sc_offset_blkq6k + 4*i, 0);
                }

                short tmpd_ = __ivcorex_ml_mem_load_i16(bq6KBase, kbx*sizeof(block_q6_K), d_offset_blkq6k, 0);

                const int vh_shift = 2 * ((kqs % (QI6_K/2)) / (QI6_K/4));
                float sumf = 0.0f;

#pragma unroll
                for (int i = 0; i < QR6_K; ++i) {
                    d8[i] = __low2float(*((ggml_half2 *)(&tmpd8_[i])));
                }
                
                int vl  = (*((uint16_t *) (&vl0))) << 0;
                vl |= (uint32_t((*(uint16_t *) (&vl1)))) << 16;
                int vh  = (*((uint16_t *) (&vh0))) << 0;
                vh |= (uint32_t((*(uint16_t *) (&vh1)))) << 16;
                vh = vh >> vh_shift;

#pragma unroll
                for (int i = 0; i < QR6_K; ++i) {
                    const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

                    const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

                    const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

                    const int sc = tmpsc[i];

                    sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
                }

                ggml_half tmpd = *((ggml_half *)(&tmpd_));
                tmp[j*rows_per_cuda_block+i] += (float(tmpd))*sumf;
            }
        }
    }

    reduce_write<nwarps, ncols_dst, warp_size, rows_per_cuda_block, opins>(dst, tmp, row0, sample_dst, stride_sample_dst, channel_dst, stride_channel_dst, stride_col_dst, nextsrc0, nextsrc1);
}

template <ggml_type type, int ncols_dst, int warp_size, int rows_per_cuda_block, op_another opins>
static __global__ void mul_mat_vec_q_Q8_0(mul_mat_vec_q_common_params, float *nextsrc0, float *nextsrc1) {
    mul_mat_vec_q_common_variable

    const int qs_offset_blkq81 = 4; // qs在block_q8_1的偏移字节数
    const int qs_offset_blkq80 = 2; // qs在block_q8_0的偏移字节数
    const int ds_offset_blkq81 = 0; // ds在block_q8_1的偏移字节数

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

        for (int j = 0; j < ncols_dst; ++j) {
            int u[VDR_Q8_0_Q8_1_MMVQ];

            v4u32 yBase;
            yBase.x  = (unsigned) (unsigned long long) (y+j*stride_col_y);
            yBase.y  = (unsigned) ((unsigned long long) (y+j*stride_col_y) >> 32);
            yBase.zw = -1u;

            short dstmp = __ivcorex_ml_mem_load_i16(yBase, kby * sizeof(block_q8_1), ds_offset_blkq81, 0);

#pragma unroll
            for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
                u[i] = __ivcorex_ml_mem_load_i32(yBase, kby * sizeof(block_q8_1) + (kqs + i) * 4, qs_offset_blkq81, 0);
            }
            
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const block_q8_0 * bq8_0 = (const block_q8_0 *) vx + (kbx_offset + i*stride_row_x);
                v4u32 blkq80Base;
                blkq80Base.x  = (unsigned) (unsigned long long) (bq8_0);
                blkq80Base.y  = (unsigned) ((unsigned long long) (bq8_0) >> 32);
                blkq80Base.zw = -1u;

                short tmpv[VDR_Q8_0_Q8_1_MMVQ][2];
#pragma unroll
                for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
                    tmpv[i][0] = __ivcorex_ml_mem_load_i16(blkq80Base, kbx*sizeof(block_q8_0) + (kqs + i)*4, qs_offset_blkq80, 0);
                    tmpv[i][1] = __ivcorex_ml_mem_load_i16(blkq80Base, kbx*sizeof(block_q8_0) + (kqs + i)*4, qs_offset_blkq80 + 2, 0);
                }

                short dtmp = __ivcorex_ml_mem_load_i16(blkq80Base, kbx*sizeof(block_q8_0), 0, 0);

                int v[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
                for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
                    v[i]  = (*((uint16_t *) (&tmpv[i][0]))) << 0;
                    v[i] |= (uint32_t((*(uint16_t *) (&tmpv[i][1])))) << 16;
                }

                tmp[j*rows_per_cuda_block+i] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, *((ggml_half *)(&dtmp)), *((half *)(&dstmp)));
            }
        }
    }

    reduce_write<nwarps, ncols_dst, warp_size, rows_per_cuda_block, opins>(dst, tmp, row0, sample_dst, stride_sample_dst, channel_dst, stride_channel_dst, stride_col_dst, nextsrc0, nextsrc1);
}

static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_y, const int nsamples_y,
        const int warp_size, const mmvq_parameter_table_id table_id, int rows_per_block) {
    const int64_t nblocks = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks, nchannels_y, nsamples_y);
    const dim3 block_dims(warp_size, calc_nwarps(ncols_dst, table_id), 1);
    return {block_nums, block_dims};
}

template <ggml_type type, int warpsize, int rows_per_block, int c_ncols_dst, op_another opins>
static void mul_mat_vec_q_switch_type_for_kernel(
        const std::pair<dim3, dim3> &dims, cudaStream_t stream, mul_mat_vec_q_common_params, float *nextsrc0, float *nextsrc1) {

    if (type == GGML_TYPE_Q2_K) {
        mul_mat_vec_q_Q2_K<type, c_ncols_dst, warpsize, rows_per_block, opins><<<dims.first, dims.second, 0, stream>>>(
            vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst, channel_ratio,
            stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
            stride_sample_dst, nextsrc0, nextsrc1);
    } else if (type == GGML_TYPE_Q3_K) {
        mul_mat_vec_q_Q3_K<type, c_ncols_dst, warpsize, rows_per_block, opins><<<dims.first, dims.second, 0, stream>>>(
            vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst, channel_ratio,
            stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
            stride_sample_dst, nextsrc0, nextsrc1);
    } else if (type == GGML_TYPE_Q4_K) {
        mul_mat_vec_q_Q4_K<type, c_ncols_dst, warpsize, rows_per_block, opins><<<dims.first, dims.second, 0, stream>>>(
            vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst, channel_ratio,
            stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
            stride_sample_dst, nextsrc0, nextsrc1);
#if defined(__x86_64__) || defined(__i386__) // no mul_mat_vec_q_Q6_K on ARM because it will cause garbled text
    } else if (type == GGML_TYPE_Q6_K) {
        mul_mat_vec_q_Q6_K<type, c_ncols_dst, warpsize, rows_per_block, opins><<<dims.first, dims.second, 0, stream>>>(
            vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst, channel_ratio,
            stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
            stride_sample_dst, nextsrc0, nextsrc1);
#endif
    } else if (type == GGML_TYPE_Q8_0) {
        mul_mat_vec_q_Q8_0<type, c_ncols_dst, warpsize, rows_per_block, opins><<<dims.first, dims.second, 0, stream>>>(
            vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst, channel_ratio,
            stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
            stride_sample_dst, nextsrc0, nextsrc1);
    } else {
        mul_mat_vec_q<type, c_ncols_dst, warpsize, rows_per_block, opins><<<dims.first, dims.second, 0, stream>>>(
            vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst, channel_ratio,
            stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
            stride_sample_dst, nextsrc0, nextsrc1);
    }
}

template <ggml_type type, int warpsize, int rows_per_block, int c_ncols_dst>
static void mul_mat_vec_q_switch_nextop(
        const std::pair<dim3, dim3> &dims, cudaStream_t stream, mul_mat_vec_q_common_params, ggml_tensor * nextdst) {

    float * nextsrc0 = nullptr;
    float * nextsrc1 = nullptr;
    
    if (nextdst != nullptr) {
        nextsrc0 = (float *) nextdst->src[0]->data;
        nextsrc1 = (float *) nextdst->src[1]->data;
    }

    if (nextdst == nullptr || (nextdst && nextdst->op == GGML_OP_ADD)){
        mul_mat_vec_q_switch_type_for_kernel<type, warpsize, rows_per_block, c_ncols_dst, op_add_another>(dims, stream, vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextsrc0, nextsrc1);
    } else if (nextdst && nextdst->op == GGML_OP_MUL) {
        mul_mat_vec_q_switch_type_for_kernel<type, warpsize, rows_per_block, c_ncols_dst, op_mul_another>(dims, stream, vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextsrc0, nextsrc1);
    } else {
        GGML_ABORT("fatal error");
    }
}

template <ggml_type type, int warpsize, int rows_per_block>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream, ggml_tensor * nextdst) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const int channel_ratio = nchannels_dst / nchannels_x;
    const int sample_ratio  = nsamples_dst  / nsamples_x;

    const int device = ggml_cuda_get_device();
    const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[device].cc);

    GGML_ASSERT(!ids || ncols_dst == 1);
    switch (ncols_dst) {
        case 1:
        {
            constexpr int c_ncols_dst = 1;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warpsize, table_id, rows_per_block);
            mul_mat_vec_q_switch_nextop<type, warpsize, rows_per_block, c_ncols_dst>(dims, stream, vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextdst);            
            break;
        }
        case 2:
        {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warpsize, table_id, rows_per_block);
            mul_mat_vec_q_switch_nextop<type, warpsize, rows_per_block, c_ncols_dst>(dims, stream, vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextdst);            
            break;
        }
        case 3:
        {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warpsize, table_id, rows_per_block);
            mul_mat_vec_q_switch_nextop<type, warpsize, rows_per_block, c_ncols_dst>(dims, stream, vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextdst);            
            break;
        }
        case 4:
        {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warpsize, table_id, rows_per_block);
            mul_mat_vec_q_switch_nextop<type, warpsize, rows_per_block, c_ncols_dst>(dims, stream, vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextdst); 
            break;
        }
        case 5:
        {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warpsize, table_id, rows_per_block);
            mul_mat_vec_q_switch_nextop<type, warpsize, rows_per_block, c_ncols_dst>(dims, stream, vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextdst); 
            break;
        }
        case 6:
        {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warpsize, table_id, rows_per_block);
            mul_mat_vec_q_switch_nextop<type, warpsize, rows_per_block, c_ncols_dst>(dims, stream, vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextdst); 
            break;
        }
        case 7:
        {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warpsize, table_id, rows_per_block);
            mul_mat_vec_q_switch_nextop<type, warpsize, rows_per_block, c_ncols_dst>(dims, stream, vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextdst); 
            break;
        }
        case 8:
        {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warpsize, table_id, rows_per_block);
            mul_mat_vec_q_switch_nextop<type, warpsize, rows_per_block, c_ncols_dst>(dims, stream, vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextdst); 
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template <ggml_type type, int warpsize>
static void mul_mat_vec_q_switch_rows_per_blk(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream, ggml_tensor * nextdst) {

    const int rows_per_block_dynamic = calc_rows_per_block(nrows_x, ncols_dst);

    switch (rows_per_block_dynamic)
    {
    case ROWS_PER_BLOCK_4:
        mul_mat_vec_q_switch_ncols_dst<type, warpsize, ROWS_PER_BLOCK_4>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
        break;
    case ROWS_PER_BLOCK_2:
        mul_mat_vec_q_switch_ncols_dst<type, warpsize, ROWS_PER_BLOCK_2>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
        break;
    default:
        mul_mat_vec_q_switch_ncols_dst<type, warpsize, ROWS_PER_BLOCK_1>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
        break;
    }
}

template <ggml_type type>
static void mul_mat_vec_q_switch_warpsize(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream, ggml_tensor * nextdst) {
    int warp_size_dynamic = DEFAULT_REDUCE_WARP_SIZE0;
    if (
        //ncols_dst == 1 &&
        (
            (nrows_x < 1024 && ncols_x < 1024 && nchannels_dst == 1) 
            || (nrows_x <= 5120 && ncols_x / nrows_x >= 6)
        )
       )
    {
        warp_size_dynamic = DEFAULT_REDUCE_WARP_SIZE1;
    }

    switch (warp_size_dynamic)
    {
    case DEFAULT_REDUCE_WARP_SIZE0:
        mul_mat_vec_q_switch_rows_per_blk<type, DEFAULT_REDUCE_WARP_SIZE0>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
        break;
    case DEFAULT_REDUCE_WARP_SIZE1:
        mul_mat_vec_q_switch_rows_per_blk<type, DEFAULT_REDUCE_WARP_SIZE1>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
        break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}

static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream, ggml_tensor * nextdst=nullptr) {
    switch (type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_Q4_0>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_Q4_1>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_Q5_0>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_Q5_1>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_Q8_0>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_Q2_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_Q3_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_Q4_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_Q5_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_Q6_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_IQ2_S>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_IQ1_S>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_IQ1_M>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_warpsize<GGML_TYPE_IQ3_S>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream, nextdst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, ggml_tensor * nextdst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 == 1); // Implementation is only correct for batch size 1.

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        GGML_ASSERT(ggml_is_contiguous(src0));
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,                 stream, nextdst);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream, ggml_tensor * nextdst) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, stream, nextdst);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}
