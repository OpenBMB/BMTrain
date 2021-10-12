#pragma once
#include <cublasLt.h>

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

void bmm_i8_kernel(GemmContext ctx, int32_t batch_size, const int8_t *A, bool cast_A, const int8_t *B, bool cast_B, int32_t *C, cudaStream_t stream) ;
void bmm_f16_kernel(GemmContext ctx, int32_t batch_size, const half *A, const half *B, half *C, cudaStream_t stream);
void bmm_f32_kernel(GemmContext ctx, int32_t batch_size, const float *A, const float *B, float *C, cudaStream_t stream);


void round_scale_i8(int batch, int n, int m, const float *mat, const float *scale, int8_t *out, cudaStream_t stream);
void round_scale_i8(int batch, int n, int m, const half *mat, const half *scale, int8_t *out, cudaStream_t stream);

void scale_i32_f16(int batch, int n, int m, const int32_t *mat, const half *scale_x, const half *scale_y, half *out, bool broadcast_x, bool broadcast_y, cudaStream_t stream);
void scale_i32_f32(int batch, int n, int m, const int32_t *mat, const float *scale_x, const float *scale_y, float *out, bool broadcast_x, bool broadcast_y, cudaStream_t stream);

void calc_scale_f16(int batch, int n, int m, const half *mat, half *out, cudaStream_t stream);
void calc_scale_f32(int batch, int n, int m, const float *mat, float *out, cudaStream_t stream);


void release_gemm_context(GemmContext ctx);
GemmContext create_gemm_context_i8(
    int m, int n, int k,
    bool aT, bool bT,
    int32_t lda, int32_t ldb, int32_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c
);
GemmContext create_gemm_context_fp(
    cudaDataType data_type,
    int m, int n, int k,
    bool aT, bool bT,
    int32_t lda, int32_t ldb, int32_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c
);