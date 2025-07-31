#include "ggml.h"
#include "common.cuh"
#include "mmv.cuh"

#define mywarpsize 32

template <typename T, typename type_acc, int block_size, int rows_per_blk, op_another opins>
static __global__ void mul_mat_vec(
        const T * __restrict__ x, const float * __restrict__ y, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int64_t ncols2, const int64_t nchannels_y, const int64_t stride_row,
        const int64_t channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int64_t sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst, float *nextsrc0, float *nextsrc1) {
    const int64_t row         = blockIdx.x;
    const int64_t channel_dst = blockIdx.y;
    const int64_t channel_x   = ids ? ids[channel_dst]          : channel_dst / channel_ratio;
    const int64_t channel_y   = ids ? channel_dst % nchannels_y : channel_dst;
    const int64_t sample_dst  = blockIdx.z;
    const int64_t sample_x    = sample_dst / sample_ratio;
    const int64_t sample_y    = sample_dst;
    const int     tid         = threadIdx.x;
    constexpr int warp_size   = mywarpsize;

    //x   += sample_x  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    y   += sample_y  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;

    const float2 * y2 = (const float2 *) y;

    __shared__ float buf_iw[rows_per_blk][mywarpsize];

    if (block_size > warp_size) {
#pragma unroll
        for (size_t i = 0; i < rows_per_blk; i++){
            if (tid < mywarpsize) {
                buf_iw[i][tid] = 0.0f;
            }
        }
        __syncthreads();
    }

    float sumf[rows_per_blk];
#pragma unroll
    for (size_t i = 0; i < rows_per_blk; i++){
        sumf[i] = 0.f;
    }

    if constexpr (std::is_same<T, float>::value) {
        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpy = y2[col2];
#pragma unroll
            for (size_t i = 0; i < rows_per_blk; i++){
                const T *newx = x + sample_x  *stride_sample_x   + channel_x  *stride_channel_x   + (row*rows_per_blk+i)*stride_row;
                const float2 * x2 = (const float2 *) x;
                const float2 tmpx = x2[col2];
                sumf[i] += tmpx.x*tmpy.x;
                sumf[i] += tmpx.y*tmpy.y;
            }
        }
    } else if constexpr (std::is_same<T, half>::value) {
        if (std::is_same<type_acc, float>::value) {
            for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpy = y2[col2];
#pragma unroll
                for (size_t i = 0; i < rows_per_blk; i++){
                    const T *newx = x + sample_x  *stride_sample_x   + channel_x  *stride_channel_x   + (row*rows_per_blk+i)*stride_row;
                    const half2 * x2 = (const half2 *) newx;
                    const float2 tmpx = __half22float2(x2[col2]);
                    sumf[i] += tmpx.x * tmpy.x;
                    sumf[i] += tmpx.y * tmpy.y;
                }
            }
        } else {
            half2 sumh2[rows_per_blk];
#pragma unroll
            for (size_t i = 0; i < rows_per_blk; i++){
                sumh2[i] = make_half2(0.0f, 0.0f);
            }

            for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmp = y2[col2];
#pragma unroll
                for (size_t i = 0; i < rows_per_blk; i++){
                    const T *newx = x + sample_x  *stride_sample_x   + channel_x  *stride_channel_x   + (row*rows_per_blk+i)*stride_row;
                    const half2 * x2 = (const half2 *) newx;
                    sumh2[i] += x2[col2] * make_half2(tmp.x, tmp.y);
                }
            }

#pragma unroll
            for (size_t i = 0; i < rows_per_blk; i++){
                sumf[i] = __low2float(sumh2[i]) + __high2float(sumh2[i]);
            }
        }
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpy = y2[col2];
#pragma unroll
            for (size_t i = 0; i < rows_per_blk; i++){
                const T *newx = x + sample_x  *stride_sample_x   + channel_x  *stride_channel_x   + (row*rows_per_blk+i)*stride_row;
                const int * x2 = (const int *) x;
                const int tmpx = x2[col2];
                sumf[i] += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]) * tmpy.x;
                sumf[i] += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]) * tmpy.y;
            }
        }
    } else {
        static_assert(std::is_same<T, void>::value, "unsupported type");
    }

#pragma unroll
    for (size_t i = 0; i < rows_per_blk; i++){
        sumf[i] = warp_reduce_sum<mywarpsize>(sumf[i]);
    }

    if (block_size > warp_size) {
#pragma unroll
        for (size_t i = 0; i < rows_per_blk; i++){
            buf_iw[i][tid/mywarpsize] = sumf[i];
            __syncthreads();
            if (tid >= mywarpsize) {
                continue;
            }
            sumf[i] = buf_iw[i][tid];
            sumf[i] = warp_reduce_sum<mywarpsize>(sumf[i]);
        }
    }

    if (tid != 0) {
        return;
    }

#pragma unroll
    for (size_t i = 0; i < rows_per_blk; i++) {
        opins(dst, row*rows_per_blk+i, sumf[i], nextsrc0, nextsrc1);
    }
}

template <typename T, typename type_acc, int block_size, int rows_per_blk>
static void launch_mul_mat_vec_cuda_dstop(const dim3 &block_nums, const dim3 &block_dims, cudaStream_t stream, ggml_tensor * nextdst,
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols2, const int64_t nchannels_y, const int64_t stride_row,
        const int64_t channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int64_t sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst) {
        
        float * nextsrc0 = nullptr;
        float * nextsrc1 = nullptr;
        if (nextdst != nullptr) {
            nextsrc0 = (float *) nextdst->src[0]->data;
            nextsrc1 = (float *) nextdst->src[1]->data;
        }
        
        const int smem = 0;

        if (nextdst == nullptr || (nextdst && nextdst->op == GGML_OP_ADD)) {
            mul_mat_vec<T, type_acc, block_size, rows_per_blk, op_add_another><<<block_nums, block_dims, smem, stream>>>(
                x, y, ids, dst, ncols2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextsrc0, nextsrc1);
        } else if (nextdst && nextdst->op == GGML_OP_MUL) {
            mul_mat_vec<T, type_acc, block_size, rows_per_blk, op_mul_another><<<block_nums, block_dims, smem, stream>>>(
                x, y, ids, dst, ncols2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nextsrc0, nextsrc1);
        } else {
            GGML_ABORT("fatal error");
        }
}

template <typename T, typename type_acc>
static void launch_mul_mat_vec_cuda(
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        cudaStream_t stream, ggml_tensor * nextdst) {
    GGML_ASSERT(ncols      % 2 == 0);
    GGML_ASSERT(stride_row % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const int64_t channel_ratio = nchannels_dst / nchannels_x;
    const int64_t sample_ratio  = nsamples_dst  / nsamples_x;
    int device;
    int warp_size;

    CUDA_CHECK(cudaGetDevice(&device));
    warp_size = mywarpsize;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const static int rows_per_blk1 = 1;
    dim3 block_nums(nrows/rows_per_blk1, nchannels_dst, nsamples_dst);
    const static int rows_per_blk2 = 2;
    const static int rows_per_blk4 = 4;
    if(block_size_best >= 192){
        block_nums.x = nrows / rows_per_blk4;
    }
    else if(block_size_best >= 128){
        block_nums.x = nrows / rows_per_blk2;
    }

    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
        case   32: {
            launch_mul_mat_vec_cuda_dstop<T, type_acc,  32, rows_per_blk1>(block_nums, block_dims, stream, nextdst,
                x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case   64: {
            launch_mul_mat_vec_cuda_dstop<T, type_acc,  64, rows_per_blk1>(block_nums, block_dims, stream, nextdst,
                x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case   96: {
            launch_mul_mat_vec_cuda_dstop<T, type_acc,  96, rows_per_blk1>(block_nums, block_dims, stream, nextdst,
                x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  128: {
            launch_mul_mat_vec_cuda_dstop<T, type_acc, 128, rows_per_blk2>(block_nums, block_dims, stream, nextdst,
                x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  160: {
            launch_mul_mat_vec_cuda_dstop<T, type_acc, 160, rows_per_blk2>(block_nums, block_dims, stream, nextdst,
                x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  192: {
            launch_mul_mat_vec_cuda_dstop<T, type_acc, 192, rows_per_blk4>(block_nums, block_dims, stream, nextdst,
                x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  224: {
            launch_mul_mat_vec_cuda_dstop<T, type_acc, 224, rows_per_blk4>(block_nums, block_dims, stream, nextdst,
                x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  256: {
            launch_mul_mat_vec_cuda_dstop<T, type_acc, 256, rows_per_blk4>(block_nums, block_dims, stream, nextdst,
                x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

template<typename T>
static void mul_mat_vec_cuda(
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        enum ggml_prec prec, cudaStream_t stream, ggml_tensor * nextdst) {
    if constexpr(std::is_same<T, half>::value) {
        if (prec == GGML_PREC_DEFAULT) {
            launch_mul_mat_vec_cuda<T, half>
                (x, y, ids, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, nextdst);
            return;
        }
    }
    launch_mul_mat_vec_cuda<T, float>
        (x, y, ids, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
         stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, nextdst);
}

void ggml_cuda_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, ggml_tensor * nextdst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(!ids || ne12 == 1); // Implementation is only correct for  batch size 1.
    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    GGML_ASSERT(ncols_dst == 1);

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            mul_mat_vec_cuda(src0_d, src1_d, ids_d, dst_d, ne00, ne01, s01,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 prec, ctx.stream(), nextdst);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            mul_mat_vec_cuda(src0_d, src1_d, ids_d, dst_d, ne00, ne01, s01,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 prec, ctx.stream(), nextdst);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            mul_mat_vec_cuda(src0_d, src1_d, ids_d, dst_d, ne00, ne01, s01,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 prec, ctx.stream(), nextdst);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_op_mul_mat_vec(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream, ggml_tensor * nextdst) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    GGML_ASSERT(src1_ncols == 1);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;


    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t nchannels_dst      = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_dst       = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_vec_cuda(src0_d, src1_ddf_i, nullptr, dst_dd_i, ne00, row_diff, stride_row,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, prec, stream, nextdst);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_vec_cuda(src0_d, src1_ddf_i, nullptr, dst_dd_i, ne00, row_diff, stride_row,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, prec, stream, nextdst);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_vec_cuda(src0_d, src1_ddf_i, nullptr, dst_dd_i, ne00, row_diff, stride_row,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, prec, stream, nextdst);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED(ctx);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}
