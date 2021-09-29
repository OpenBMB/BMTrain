#include "helper.h"
#include <torch/extension.h>

namespace {

template<typename scalar_t, bool broad_cast1, bool broad_cast2>
__global__ void scale_2dim_kernel(
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> input_tensor,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> scale_1,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> scale_2,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> out
    ) {
    
    const int n = blockIdx.z;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out.size(1) && y < out.size(2)) {
        if (!broad_cast1 && !broad_cast2) {
            out[n][x][y] = (scalar_t)((float)(input_tensor[n][x][y]) * (float)(scale_1[n][x]) * (float)(scale_2[n][y]));
        }
        else if(broad_cast1 && !broad_cast2) {
            out[n][x][y] = (scalar_t)((float)(input_tensor[n][x][y]) * (float)(scale_1[0][x]) * (float)(scale_2[n][y]));
        }
        else if(!broad_cast1 && broad_cast2) {
            out[n][x][y] = (scalar_t)((float)(input_tensor[n][x][y]) * (float)(scale_1[n][x]) * (float)(scale_2[0][y]));
        }
        else if (broad_cast1 && broad_cast2) {
            out[n][x][y] = (scalar_t)((float)(input_tensor[n][x][y]) * (float)(scale_1[0][x]) * (float)(scale_2[0][y]));
        }
    }
}

template<typename T>
__global__ void round_to_int8_kernel(
    const torch::PackedTensorAccessor32<T, 1, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<T, 1, torch::RestrictPtrTraits> scale,
    torch::PackedTensorAccessor32<int8_t, 1, torch::RestrictPtrTraits> out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out.size(0)) {
        out[idx] = (int8_t)nearbyintf(x[idx] / scale[idx]);
    }
}

}

void* create_cublaslt_handle() {
    cublasLtHandle_t handle;
    checkCublasStatus(cublasLtCreate(&handle));
    return (void*)handle;
}


void scale_2dim(
    torch::Tensor x,
    torch::Tensor scale_1,
    torch::Tensor scale_2,
    torch::Tensor out
) {
    int32_t thread_size = 32;
    const dim3 threads(thread_size, thread_size);
    const dim3 blocks((x.size(1) + thread_size - 1) / thread_size, (x.size(2) + thread_size - 1) / thread_size, x.size(0));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(out.type(), "i8linear_scale_cuda", ([&] {
        if (scale_1.size(0) == 1 && scale_2.size(0) == 1) {
            scale_2dim_kernel<scalar_t, true, true><<<blocks,threads>>>(
                x.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                scale_1.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                scale_2.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
            );
        }
        else if (scale_1.size(0) != 1 && scale_2.size(0) == 1) {
            scale_2dim_kernel<scalar_t, false, true><<<blocks,threads>>>(
                x.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                scale_1.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                scale_2.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
            );
        }
        else if (scale_1.size(0) == 1 && scale_2.size(0) != 1) {
            scale_2dim_kernel<scalar_t, true, false><<<blocks,threads>>>(
                x.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                scale_1.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                scale_2.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
            );
        }
        else if (scale_1.size(0) != 1 && scale_2.size(0) != 1) {
            scale_2dim_kernel<scalar_t, false, false><<<blocks,threads>>>(
                x.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                scale_1.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                scale_2.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
            );
        }
    }));
    
}

void LtIgemm(cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const int8_t *A,
                   int64_t stride_a,
                   const int8_t *B,
                   int64_t stride_b,
                   int32_t *C,
                   int64_t stride_c,
                   int32_t num_batches) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    int32_t alpha = 1, beta = 0;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // ---------------------------------------------------------------------------------------------
    // init MatMul Desc

#if (CUDART_VERSION >= 11000)
    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
#else
    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUDA_R_32I));
#endif

    
    // transA
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
    
    // ---------------------------------------------------------------------------------------------
    // create descriptors for original matrices
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, k, m, k));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, k));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, m));

    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_batches, sizeof(num_batches)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_batches, sizeof(num_batches)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_batches, sizeof(num_batches)));

    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));

    // ---------------------------------------------------------------------------------------------
    // transforms and computation
    // no need to transform C matrix as beta is assumed to be 0
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     matmulDesc,
                                     &alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     &beta,
                                     C,
                                     Cdesc,
                                     C,
                                     Cdesc,
                                     NULL,
                                     NULL,
                                     0,
                                     0));

    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
}


void round_to_int8(torch::Tensor x, torch::Tensor scale, torch::Tensor out) {
    int32_t thread_size = 1024;
    const dim3 threads(thread_size);
    const dim3 blocks((x.size(0) + thread_size - 1) / thread_size);

    AT_DISPATCH_INTEGRAL_TYPES(out.type(), "calc_bias_bucket_cuda", ([&] {
        round_to_int8_kernel<scalar_t><<<blocks,threads>>>(
            x.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            out.packed_accessor32<int8_t, 1, torch::RestrictPtrTraits>()
        );
    }));
}