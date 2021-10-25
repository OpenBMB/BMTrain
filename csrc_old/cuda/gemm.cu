#include "common/helper.h"
#include "common/reduce.cuh"
#include "ops/gemm.h"

namespace {

__device__ inline float read_float(const int32_t *ptr) {
    return (float)__ldg(ptr);
}

__device__ inline float read_float(const float *ptr) {
    return __ldg(ptr);
}

__device__ inline float read_float(const half *ptr) {
    return __half2float( __ldg(ptr) );
}

template<typename scalar_t>
__global__ void round_i8_kernel(
    int32_t batch, int32_t n, int32_t m,
    const scalar_t *mat,       // b, n, m
    const scalar_t *scale,     // b, n
    int8_t *out
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;   // mat[batch][n][m], scale[batch][n]
    float local_scale = read_float(scale + blockIdx.x * n + blockIdx.y);

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x){
        out[base_idx + i] = (int8_t)nearbyintf(read_float(mat + base_idx + i) / local_scale); 
    }
}

template<typename scalar_t>
__global__ void round_i8_transpose_kernel(
    int32_t batch, int32_t n, int32_t m,
    const scalar_t *mat,       // b, n, m
    const scalar_t *scale,     // b, m
    int8_t *out
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;   // mat[batch][n][m], scale[batch][n]

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x){
        out[base_idx + i] = (int8_t)nearbyintf(read_float(mat + base_idx + i) / read_float(scale + blockIdx.x * m + i)); 
    }
}

template<typename scalar_t, bool broad_cast_x, bool broad_cast_y>
__global__ void scale_i32_kernel(
    int32_t batch, int32_t n, int32_t m,
    const int32_t *mat,        // b, n, m
    const scalar_t *scale_x,   // b, n
    const scalar_t *scale_y,   // b, m
    scalar_t *out
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;
    float scale_x_value = 0;
    if (broad_cast_x) {
        scale_x_value = read_float(scale_x + blockIdx.y);
    } else {
        scale_x_value = read_float(scale_x + blockIdx.x * n + blockIdx.y);
    }

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x){
        if (broad_cast_y) {
            out[base_idx + i] = read_float(mat + base_idx + i) * scale_x_value * read_float(scale_y + i);
        }
        else {
            out[base_idx + i] = read_float(mat + base_idx + i) * scale_x_value * read_float(scale_y + blockIdx.x * m + i);
        }
    }
}

template<bool broad_cast_x, bool broad_cast_y>
__global__ void scale_i32_kernel(
    int32_t batch, int32_t n, int32_t m,
    const int32_t *mat,        // b, n, m
    const half *scale_x,   // b, n
    const half *scale_y,   // b, m
    half *out
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;
    float scale_x_value = 0;
    if (broad_cast_x) {
        scale_x_value = read_float(scale_x + blockIdx.y);
    } else {
        scale_x_value = read_float(scale_x + blockIdx.x * n + blockIdx.y);
    }

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x){
        if (broad_cast_y) {
            out[base_idx + i] = __float2half(read_float(mat + base_idx + i) * scale_x_value * read_float(scale_y + i));
        }
        else {
            out[base_idx + i] = __float2half(read_float(mat + base_idx + i) * scale_x_value * read_float(scale_y + blockIdx.x * m + i));
        }
    }
}

template<typename scalar_t>
__global__ void calc_scale_kernel(
    int32_t batch, int32_t n, int32_t m,
    const scalar_t *mat,        // b, n, m
    scalar_t *out  // b, n
) {
    float local_max = 0;

    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;
    for (int32_t i = 0; i < m; i += blockDim.x){
        int32_t offset = threadIdx.x + i;
        float v = 0;
        if (offset < m) {
            v = fabsf(read_float(mat + base_idx + offset));
        }
        local_max = fmaxf(blockReduceMax(v), local_max);
    }

    if (threadIdx.x == 0) {
        out[ blockIdx.x * n + blockIdx.y ] = local_max / 127.0;
    }
}

__global__ void calc_scale_kernel(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,        // b, n, m
    half *out  // b, n
) {
    float local_max = 0;

    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;
    for (int32_t i = 0; i < m; i += blockDim.x){
        int32_t offset = threadIdx.x + i;
        float v = 0;
        if (offset < m) {
            v = fabsf(read_float(mat + base_idx + offset));
        }
        local_max = fmaxf(blockReduceMax(v), local_max);
    }

    if (threadIdx.x == 0) {
        out[ blockIdx.x * n + blockIdx.y ] = __float2half(local_max / 127.0);
    }
}

template<typename scalar_t>
__global__ void calc_scale_transpose_kernel(
    int32_t batch, int32_t n, int32_t m,
    const scalar_t *in,        // b, n, m
    scalar_t *out  // b, m
) {
    __shared__ float mat[WARP_SZ][WARP_SZ + 1]; // shared bank conflict

    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t base_idx = (blockIdx.x * n + threadIdx.y) * m + col_idx;
    
    float local_max = 0.0;
    for (int32_t i = 0; i < n; i += WARP_SZ) {
        // put & transpose
        if (i + threadIdx.y < n && col_idx < m) {
            mat[threadIdx.x][threadIdx.y] = fabsf(read_float(in + base_idx + i * m));
        } else {
            mat[threadIdx.x][threadIdx.y] = 0;
        }
        __syncthreads();
        float v = mat[threadIdx.y][threadIdx.x];
        local_max = fmaxf(warpReduceMax(v), local_max);
    }
    if (threadIdx.x == 0) {
        mat[threadIdx.y][WARP_SZ] = local_max / 127.0;
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        out[blockIdx.x * m + blockIdx.y * WARP_SZ + threadIdx.x] = mat[threadIdx.x][WARP_SZ];
    }
}

__global__ void calc_scale_transpose_kernel(
    int32_t batch, int32_t n, int32_t m,
    const half *in,        // b, n, m
    half *out  // b, m
) {
    __shared__ float mat[WARP_SZ][WARP_SZ + 1]; // shared bank conflict

    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t base_idx = (blockIdx.x * n + threadIdx.y) * m + col_idx;
    
    float local_max = 0.0;
    for (int32_t i = 0; i < n; i += WARP_SZ) {
        // put & transpose
        if (i + threadIdx.y < n && col_idx < m) {
            mat[threadIdx.x][threadIdx.y] = fabsf(read_float(in + base_idx + i * m));
        } else {
            mat[threadIdx.x][threadIdx.y] = 0;
        }
        __syncthreads();
        float v = mat[threadIdx.y][threadIdx.x];
        local_max = fmaxf(warpReduceMax(v), local_max);
    }
    if (threadIdx.x == 0) {
        mat[threadIdx.y][WARP_SZ] = local_max / 127.0;
    }
    __syncthreads();
    if (threadIdx.y == 0 && col_idx < m) {
        out[blockIdx.x * m + col_idx] = __float2half(mat[threadIdx.x][WARP_SZ]);
    }
}

}


void bmm_i8_kernel(
    GemmContext ctx,
    int32_t batch_size,
    const int8_t *A,
    bool cast_A,
    const int8_t *B,
    bool cast_B,
    int32_t *C,
    cudaStream_t stream
) {
    cublasLtMatrixLayout_t layout_A, layout_B, layout_C;
    const int8_t *buffer_A, *buffer_B;
    int32_t *buffer_C;
    int8_t alpha = 1, beta = 0;
    int32_t batch_count_1 = 1;

    if (ctx.A.transform != NULL) {
        if (cast_A) {
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
                &batch_size,
                sizeof(batch_size)
            ));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
                ctx.A.layout_transform,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_size,
                sizeof(batch_size)
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
        if (cast_B == 0) {
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
                &batch_size,
                sizeof(batch_size)
            ));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
                ctx.B.layout_transform,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_size,
                sizeof(batch_size)
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
        &batch_size,
        sizeof(batch_size)
    ));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        layout_B,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)
    ));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        layout_C,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)
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
            NULL, // &ctx.algo,
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

template<typename sclar_t>
void bmm_fp_kernel(
    GemmContext ctx,
    int32_t batch_size,
    const sclar_t *A,
    const sclar_t *B,
    sclar_t *C,
    cudaStream_t stream
) {

    sclar_t alpha = 1.0, beta = 0.0;

    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        ctx.A.layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)
    ));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        ctx.B.layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)
    ));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        ctx.C.layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)
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
        NULL,   // &ctx.algo
        ctx.workspace,
        ctx.workspace_size,
        stream
    ));
}

void bmm_fp_kernel(
    GemmContext ctx,
    int32_t batch_size,
    const half *A,
    const half *B,
    half *C,
    cudaStream_t stream
) {

    half alpha = __float2half(1.0), beta = __float2half(0.0);

    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        ctx.A.layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)
    ));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        ctx.B.layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)
    ));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        ctx.C.layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)
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
        NULL,   // &ctx.algo
        ctx.workspace,
        ctx.workspace_size,
        stream
    ));
}

void bmm_f16_kernel(
    GemmContext ctx,
    int32_t batch_size,
    const half *A,
    const half *B,
    half *C,
    cudaStream_t stream
) {
    bmm_fp_kernel(ctx, batch_size, A, B, C, stream);
}

void bmm_f32_kernel(
    GemmContext ctx,
    int32_t batch_size,
    const float *A,
    const float *B,
    float *C,
    cudaStream_t stream
) {
    bmm_fp_kernel<float>(ctx, batch_size, A, B, C, stream);
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
    dim3 threads( ::min(round_up(m, WARP_SZ), 1024) );
    ::round_i8_kernel<<<blocks, threads, 0, stream>>>(
        batch, n, m,
        mat,
        scale,
        out
    );
}

void round_scale_i8(int batch, int n, int m, const float *mat, const float *scale, int8_t *out, cudaStream_t stream) { round_i8_launcher(batch, n, m, mat, scale, out, stream); }
void round_scale_i8(int batch, int n, int m, const half *mat, const half *scale, int8_t *out, cudaStream_t stream) { round_i8_launcher(batch, n, m, mat, scale, out, stream); }

template<typename scalar_t>
void round_i8_transpose_launcher(
    int batch, int n, int m,
    const scalar_t *mat,       // b, n, m
    const scalar_t *scale,     // b, m
    int8_t *out,
    cudaStream_t stream
) {
    dim3 blocks(batch, n);
    dim3 threads( ::min(round_up(m, WARP_SZ), 1024) );
    ::round_i8_transpose_kernel<<<blocks, threads, 0, stream>>>(
        batch, n, m,
        mat,
        scale,
        out
    );
}

void round_scale_i8_transpose(int batch, int n, int m, const float *mat, const float *scale, int8_t *out, cudaStream_t stream) { round_i8_transpose_launcher(batch, n, m, mat, scale, out, stream); }
void round_scale_i8_transpose(int batch, int n, int m, const half *mat, const half *scale, int8_t *out, cudaStream_t stream) { round_i8_transpose_launcher(batch, n, m, mat, scale, out, stream); }

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
    dim3 threads( ::min(round_up(m, WARP_SZ), 1024) );

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

void scale_i32_launcher(
    int batch, int n, int m,
    const int32_t *mat,        // b, n, m
    const half *scale_x,   // b, n
    const half *scale_y,   // b, m
    half *out,
    bool broadcast_x,
    bool broadcast_y,
    cudaStream_t stream
) {
    dim3 blocks(batch, n);
    dim3 threads( ::min(round_up(m, WARP_SZ), 1024) );

    if (broadcast_x && broadcast_y) {
        ::scale_i32_kernel<true, true><<<blocks, threads, 0, stream>>>(
            batch, n, m,
            mat,
            scale_x,
            scale_y,
            out
        );
    }
    else if (broadcast_x && !broadcast_y) {
        ::scale_i32_kernel<true, false><<<blocks, threads, 0, stream>>>(
            batch, n, m,
            mat,
            scale_x,
            scale_y,
            out
        );
    }
    else if (!broadcast_x && broadcast_y) {
        ::scale_i32_kernel<false, true><<<blocks, threads, 0, stream>>>(
            batch, n, m,
            mat,
            scale_x,
            scale_y,
            out
        );
    }
    else {
        ::scale_i32_kernel<false, false><<<blocks, threads, 0, stream>>>(
            batch, n, m,
            mat,
            scale_x,
            scale_y,
            out
        );
    }
}

void scale_i32(int batch, int n, int m, const int32_t *mat, const half *scale_x, const half *scale_y, half *out, bool broadcast_x, bool broadcast_y, cudaStream_t stream) {
    scale_i32_launcher(batch, n, m, mat, scale_x, scale_y, out, broadcast_x, broadcast_y, stream);
}
void scale_i32(int batch, int n, int m, const int32_t *mat, const float *scale_x, const float *scale_y, float *out, bool broadcast_x, bool broadcast_y, cudaStream_t stream) {
    scale_i32_launcher<float>(batch, n, m, mat, scale_x, scale_y, out, broadcast_x, broadcast_y, stream);
}

template<typename scalar_t>
void calc_scale_launcher(
    int batch, int n, int m,
    const scalar_t *mat,        // b, n, m
    scalar_t *out,             // b, n
    cudaStream_t stream
) {
    dim3 blocks(batch, n);
    dim3 threads( ::min(round_up(m, WARP_SZ), 1024) );
    ::calc_scale_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        batch, n, m,
        mat,
        out
    );
}

void calc_scale_launcher(
    int batch, int n, int m,
    const half *mat,        // b, n, m
    half *out,             // b, n
    cudaStream_t stream
) {
    dim3 blocks(batch, n);
    dim3 threads( ::min(round_up(m, WARP_SZ), 1024) );
    ::calc_scale_kernel<<<blocks, threads, 0, stream>>>(
        batch, n, m,
        mat,
        out
    );
}

void calc_scale(int batch, int n, int m, const half *mat, half *out, cudaStream_t stream) {
    calc_scale_launcher(batch, n, m, mat, out, stream);
}

void calc_scale(int batch, int n, int m, const float *mat, float *out, cudaStream_t stream) {
    calc_scale_launcher<float>(batch, n, m, mat, out, stream);
}


template<typename scalar_t>
void calc_scale_transpose_launcher(
    int batch, int n, int m,
    const scalar_t *mat,        // b, n, m
    scalar_t *out,             // b, m
    cudaStream_t stream
) {
    dim3 blocks(batch,  (m + WARP_SZ - 1) / WARP_SZ );
    dim3 threads(WARP_SZ, WARP_SZ);

    ::calc_scale_transpose_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        batch, n, m,
        mat,
        out
    );
}

void calc_scale_transpose_launcher(
    int batch, int n, int m,
    const half *mat,        // b, n, m
    half *out,             // b, m
    cudaStream_t stream
) {
    dim3 blocks(batch,  (m + WARP_SZ - 1) / WARP_SZ );
    dim3 threads(WARP_SZ, WARP_SZ);

    ::calc_scale_transpose_kernel<<<blocks, threads, 0, stream>>>(
        batch, n, m,
        mat,
        out
    );
}

void calc_scale_transpose(int batch, int n, int m, const half *mat, half *out, cudaStream_t stream) {
    calc_scale_transpose_launcher(batch, n, m, mat, out, stream);
}

void calc_scale_transpose(int batch, int n, int m, const float *mat, float *out, cudaStream_t stream) {
    calc_scale_transpose_launcher<float>(batch, n, m, mat, out, stream);
}