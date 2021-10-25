#include "ops/gemm.h"
#include "common/helper.h"
#include <cassert>
#include <map>

std::map<int, cublasLtHandle_t> handles;

cublasLtHandle_t get_handle() {
    int device = 0;
    cublasLtHandle_t handle;

    checkCudaStatus(cudaGetDevice(&device));
    if ( handles.find(device) == handles.end() ) {
        checkCublasStatus( cublasLtCreate(&handle) );
        handles[device] = handle;
    } else {
        handle = handles[device];
    }
    return handle;
}


GemmContext create_gemm_context_i8(
    int m, int n, int k,
    bool aT, bool bT,
    int32_t lda, int32_t ldb, int32_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c
) {
    assert(m % 4 == 0);
    assert(n % 4 == 0);
    assert(k % 4 == 0);
    assert(lda % 4 == 0);
    assert(ldb % 4 == 0);
    assert(ldc % 4 == 0);
    assert(stride_a % 4 == 0);
    assert(stride_b % 4 == 0);
    assert(stride_c % 4 == 0);
    
    GemmContext ctx;
    ctx.handle = get_handle();
#if (CUDART_VERSION >= 11000)
    checkCublasStatus(cublasLtMatmulDescCreate(&ctx.matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
#else
    checkCublasStatus(cublasLtMatmulDescCreate(&ctx.matmul_desc, CUDA_R_32I));
#endif
    if (aT) {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&ctx.A.layout, CUDA_R_8I, k, m, lda));
    } else {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&ctx.A.layout, CUDA_R_8I, m, k, lda));
    }
    if (bT) {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&ctx.B.layout, CUDA_R_8I, n, k, ldb));
    } else {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&ctx.B.layout, CUDA_R_8I, k, n, ldb));
    }
    checkCublasStatus(cublasLtMatrixLayoutCreate(&ctx.C.layout, CUDA_R_32I, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(ctx.A.layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(ctx.B.layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(ctx.C.layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));

    ctx.A.layout_transform = NULL; ctx.A.transform = NULL; ctx.A.buffer = NULL;
    ctx.B.layout_transform = NULL; ctx.B.transform = NULL; ctx.B.buffer = NULL;
    ctx.C.layout_transform = NULL; ctx.C.transform = NULL; ctx.C.buffer = NULL;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    if (aT) {
        checkCublasStatus(cublasLtMatmulDescSetAttribute(
            ctx.matmul_desc,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &opTranspose,
            sizeof(opTranspose)
        ));
    }
    if (bT) {
        checkCublasStatus(cublasLtMatmulDescSetAttribute(
            ctx.matmul_desc,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &opTranspose,
            sizeof(opTranspose)
        ));
    }
    ctx.workspace = NULL; ctx.workspace_size = 0;
    return ctx;
}


GemmContext create_gemm_context_fp(
    cudaDataType data_type,
    int m, int n, int k,
    bool aT, bool bT,
    int32_t lda, int32_t ldb, int32_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c
) {
    assert(m % 4 == 0);
    assert(n % 4 == 0);
    assert(k % 4 == 0);
    GemmContext ctx;
    ctx.handle = get_handle();
#if (CUDART_VERSION >= 11000)
    if (data_type == CUDA_R_16F) {
        checkCublasStatus(cublasLtMatmulDescCreate(&ctx.matmul_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    }
    else if (data_type == CUDA_R_32F) {
        checkCublasStatus(cublasLtMatmulDescCreate(&ctx.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    }
    else {
        throw std::logic_error("Unsupported data type");
    }
#else
    if (data_type == CUDA_R_16F) {
        checkCublasStatus(cublasLtMatmulDescCreate(&ctx.matmul_desc, CUDA_R_16F));
    }
    else if (data_type == CUDA_R_32F) {
        checkCublasStatus(cublasLtMatmulDescCreate(&ctx.matmul_desc, CUDA_R_32F));
    }
    else {
        throw std::logic_error("Unsupported data type");
    }
#endif
    if (aT) {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&ctx.A.layout, data_type, k, m, lda));
    } else {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&ctx.A.layout, data_type, m, k, lda));
    }
    if (bT) {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&ctx.B.layout, data_type, n, k, ldb));
    } else {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&ctx.B.layout, data_type, k, n, ldb));
    }
    checkCublasStatus(cublasLtMatrixLayoutCreate(&ctx.C.layout, data_type, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(ctx.A.layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(ctx.B.layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(ctx.C.layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));

    ctx.A.layout_transform = NULL; ctx.A.transform = NULL; ctx.A.buffer = NULL;
    ctx.B.layout_transform = NULL; ctx.B.transform = NULL; ctx.B.buffer = NULL;
    ctx.C.layout_transform = NULL; ctx.C.transform = NULL; ctx.C.buffer = NULL;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    if (aT) {
        checkCublasStatus(cublasLtMatmulDescSetAttribute(
            ctx.matmul_desc,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &opTranspose,
            sizeof(opTranspose)
        ));
    }
    if (bT) {
        checkCublasStatus(cublasLtMatmulDescSetAttribute(
            ctx.matmul_desc,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &opTranspose,
            sizeof(opTranspose)
        ));
    }
    ctx.workspace = NULL; ctx.workspace_size = 0;
    return ctx;
}

void release_matrix(GemmMatrixContext mat) {
    if (mat.buffer) checkCudaStatus(cudaFree(mat.buffer));
    if (mat.transform) checkCublasStatus(cublasLtMatrixTransformDescDestroy(mat.transform));
    if (mat.layout) checkCublasStatus(cublasLtMatrixLayoutDestroy(mat.layout));
    if (mat.layout_transform) checkCublasStatus(cublasLtMatrixLayoutDestroy(mat.layout_transform));
}

void release_gemm_context(GemmContext ctx) {
    release_matrix(ctx.A);
    release_matrix(ctx.B);
    release_matrix(ctx.C);
    if (ctx.matmul_desc) checkCublasStatus(cublasLtMatmulDescDestroy(ctx.matmul_desc));
}