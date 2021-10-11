#include "helper.h"
#include <cublasLt.h>
#include "reduce.cuh"

namespace {
template<typename scalar_t>
__global__ void round_i8_kernel(
    int32_t batch, int32_t n, int32_t m,
    const __restrict__ scalar_t *mat,       // b, n, m
    const __restrict__ scalar_t *scale,     // b, n
    __restrict__ int8_t *out
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;   // mat[batch][n][m], scale[batch][n]
    scalar_t local_scale = scale[blockIdx.x * n + blockIdx.y];

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x){
        out[base_idx + i] = (int8_t)nearbyintf(mat[base_idx + i] / local_scale); 
    }
}

template<typename scalar_t, bool broad_cast_x, bool broad_cast_y>
__global__ void scale_i32_kernel(
    int32_t batch, int32_t n, int32_t m,
    const __restrict__ int32_t *mat,        // b, n, m
    const __restrict__ scalar_t *scale_x,   // b, n
    const __restrict__ scalar_t *scale_y,   // b, m
    __restrict__ scalar_t *out
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;
    for (int32_t i = threadIdx.x; i < m; i += blockDim.x){
        if (broad_cast_x && broad_cast_y) {
            out[base_idx + i] = (float)mat[base_idx + i] * scale_x[blockIdx.y] * scale_y[i];
        }
        else if(broad_cast_x && !broad_cast_y) {
            out[base_idx + i] = (float)mat[base_idx + i] * scale_x[blockIdx.y] * scale_y[blockIdx.x * m + i];
        }
        else if (!broad_cast_x && broad_cast_y) {
            out[base_idx + i] = (float)mat[base_idx + i] * scale_x[blockIdx.x * n + blockIdx.y] * scale_y[i];
        }
        else {
            out[base_idx + i] = (float)mat[base_idx + i] * scale_x[blockIdx.x * n + blockIdx.y] * scale_y[blockIdx.x * m + i];
        }
    }
}
template<typename scalar_t>
__global__ void calc_scale_kernel(
    int32_t batch, int32_t n, int32_t m,
    const __restrict__ scalar_t *mat,        // b, n, m
    __restrict__ scalar_t *out  // b, n
) {
    float local_max = 0;

    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;
    for (int32_t i = 0; i < m; i += blockDim.x){
        int32_t offset = threadIdx.x + i;
        float v = 0;
        if (offset < m) {
            v = fabsf(mat[base_idx + offset]);
        }
        local_max = fmaxf(blockReduceMax(v), local_max);
    }

    if (threadIdx.x == 0) {
        out[ blockIdx.x * n + blockIdx.y ] = local_max / 127.0f;
    }
}

}

struct GemmMatrixContext {
    cublasLtMatrixLayout_t layout, layout_transform;
    cublasLtMatrixTransformDesc_t transform;
    void *buffer;
};

struct GemmContext {
    cublasLtHandle_t handle;
    cublasLtMatmulAlgo_t algo;
    cublasLtMatmulDesc_t matmul_desc;
    GemmMatrixContext A, B, C;
    void *workspace;
    size_t workspace_size;
};

void bmm_i8_kernel(
    GemmContext ctx,
    int32_t batch, int n, int m, int k,
    const int8_t *A,
    int64_t stride_a,
    const int8_t *B,
    int64_t stride_b,
    int32_t *C,
    int64_t stride_c,
    cudaStream_t stream
) {
    cublasLtMatrixLayout_t layout_A, layout_B, layout_C;
    const int8_t *buffer_A, *buffer_B;
    int32_t *buffer_C;
    int8_t alpha = 1, beta = 0;
    int32_t batch_count_1 = 1;

    if (ctx.A.transform != NULL) {
        if (stride_a == 0) {
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
                ctx.A.layout,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count_1,
                sizeof(batch_count_1)
            ));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
                ctx.A.layout_transform,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count_1,
                sizeof(batch_count_1)
            ));
        } else {
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
                ctx.A.layout,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch,
                sizeof(batch)
            ));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
                ctx.A.layout_transform,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch,
                sizeof(batch)
            ));
        }
        checkCublasStatus(cublasLtMatrixTransform(
            ctx.handle,
            ctx.A.transform,
            &alpha,
            A,
            ctx.A.layout,
            &beta,
            NULL,
            NULL,
            ctx.A.buffer,
            ctx.A.layout_transform,
            stream
        ));
        layout_A = ctx.A.layout_transform;
        buffer_A = (int8_t *)ctx.A.buffer;
    } else {
        layout_A = ctx.A.layout;
        buffer_A = A;
    }
    
    if (ctx.B.transform != NULL) {
        if (stride_b == 0) {
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
                ctx.B.layout,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count_1,
                sizeof(batch_count_1)
            ));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
                ctx.B.layout_transform,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count_1,
                sizeof(batch_count_1)
            ));
        } else {
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
                ctx.B.layout,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch,
                sizeof(batch)
            ));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
                ctx.B.layout_transform,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch,
                sizeof(batch)
            ));
        }
        checkCublasStatus(cublasLtMatrixTransform(
            ctx.handle,
            ctx.B.transform,
            &alpha,
            B,
            ctx.B.layout,
            &beta,
            NULL,
            NULL,
            ctx.B.buffer,
            ctx.B.layout_transform,
            stream
        ));
        layout_B = ctx.B.layout_transform;
        buffer_B = (int8_t *)ctx.B.buffer;
    } else {
        layout_B = ctx.B.layout;
        buffer_B = B;
    }

    if (ctx.C.transform != NULL) {
        layout_C = ctx.C.layout_transform;
        buffer_C = (int32_t *)ctx.C.buffer;
    } else {
        layout_C = ctx.C.layout;
        buffer_C = C;
    }

    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        layout_A,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch,
        sizeof(batch)
    ));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        layout_B,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch,
        sizeof(batch)
    ));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        layout_C,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch,
        sizeof(batch)
    ));


    int32_t i1 = 1, i0 = 0;
    
    checkCublasStatus(
        cublasLtMatmul(
            ctx.handle,
            ctx.matmul_desc,
            &i1,
            buffer_A,
            layout_A,
            buffer_B,
            layout_B,
            &i0,
            buffer_C,
            layout_C,
            buffer_C,
            layout_C,
            &ctx.algo,
            ctx.workspace,
            ctx.workspace_size,
            stream
        )
    );

    if (ctx.C.transform != NULL) {
        checkCublasStatus(cublasLtMatrixTransform(
            ctx.handle,
            ctx.C.transform,
            &i1,
            buffer_C,
            layout_C,
            &i0,
            NULL,
            NULL,
            C,
            ctx.C.layout,
            stream
        ));
    }
}

void bmm_f16_kernel(
    GemmContext ctx,
    int32_t batch, int n, int m, int k,
    const __half *A,
    int64_t stride_a,
    const __half *B,
    int64_t stride_b,
    __half *C,
    int64_t stride_c,
    cudaStream_t stream
) {

    __half alpha = 1, beta = 0;

    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        ctx.A.layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch,
        sizeof(batch)
    ));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        ctx.B.layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch,
        sizeof(batch)
    ));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        ctx.C.layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch,
        sizeof(batch)
    ));
    checkCublasStatus(cublasLtMatmul(
        ctx.handle,
        ctx.matmul_desc,
        &alpha,
        A,
        ctx.A.layout,
        B,
        ctx.B.layout,
        &beta,
        C,
        ctx.C.layout,
        C,
        ctx.C.layout,
        &ctx.algo,
        ctx.workspace,
        ctx.workspace_size,
        stream
    ));
}

template<typename scalar_t>
void round_i8_launcher(
    int batch, int n, int m,
    const scalar_t *mat,       // b, n, m
    const scalar_t *scale,     // b, n
    int8_t *out,
    cudaStream_t stream
) {
    dim3 blocks(batch, n);
    dim3 threads( ::min(m, 1024) );
    assert(m % 32 == 0);
    ::round_i8_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        batch, n, m,
        mat,
        scale,
        out
    );
}

template<typename scalar_t>
void scale_i32_launcher(
    int batch, int n, int m,
    const int32_t *mat,        // b, n, m
    const scalar_t *scale_x,   // b, n
    const scalar_t *scale_y,   // b, m
    scalar_t *out,
    bool broadcast_x,
    bool broadcast_y,
    cudaStream_t stream
) {
    dim3 blocks(batch, n);
    dim3 threads( ::min(m, 1024) );
    assert(m % 32 == 0);

    if (broadcast_x && broadcast_y) {
        ::scale_i32_kernel<scalar_t, true, true><<<blocks, threads, 0, stream>>>(
            batch, n, m,
            mat,
            scale_x,
            scale_y,
            out
        );
    }
    else if (broadcast_x && !broadcast_y) {
        ::scale_i32_kernel<scalar_t, true, false><<<blocks, threads, 0, stream>>>(
            batch, n, m,
            mat,
            scale_x,
            scale_y,
            out
        );
    }
    else if (!broadcast_x && broadcast_y) {
        ::scale_i32_kernel<scalar_t, false, true><<<blocks, threads, 0, stream>>>(
            batch, n, m,
            mat,
            scale_x,
            scale_y,
            out
        );
    }
    else {
        ::scale_i32_kernel<scalar_t, false, false><<<blocks, threads, 0, stream>>>(
            batch, n, m,
            mat,
            scale_x,
            scale_y,
            out
        );
    }
}

template<typename scalar_t>
void calc_scale_launcher(
    int batch, int n, int m,
    const scalar_t *mat,        // b, n, m
    scalar_t *out,             // b, n
    cudaStream_t stream
) {
    dim3 blocks(batch, n);
    dim3 threads( ::min(m, 1024) );
    assert(m % 32 == 0);
    ::calc_scale_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        batch, n, m,
        mat,
        out
    );
}