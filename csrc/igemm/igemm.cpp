#include <torch/extension.h>
#include "healper.h"
#include <vector>
#include <cstdio>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INT8(x) AT_ASSERTM((x.dtype() == torch::kInt8), #x " must be int8")
#define CHECK_INT32(x) AT_ASSERTM((x.dtype() == torch::kInt32), #x " must be int32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void LtIgemm(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, int64_t stride_a, const int8_t *B, int64_t stride_b, int32_t *C, int64_t stride_c, int32_t num_batches);
void* create_cublaslt_handle();
void scale_2dim(torch::Tensor x, torch::Tensor scale_1, torch::Tensor scale_2, torch::Tensor out);

torch::Tensor i8linear_scale(
        torch::Tensor x,
        torch::Tensor scale_1,
        torch::Tensor scale_2
    ) {
    
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT32(x);
    CHECK_CUDA(scale_1); CHECK_CONTIGUOUS(scale_1);
    CHECK_CUDA(scale_2); CHECK_CONTIGUOUS(scale_2);
    AT_ASSERTM(scale_1.dtype() == scale_2.dtype(), "scale_1 and scale_2 must have the same dtype");
    AT_ASSERTM(scale_1.dim() == 2, "scale_1 must be a 2-dim tensor");
    AT_ASSERTM(scale_2.dim() == 2, "scale_2 must be a 2-dim tensor");
    AT_ASSERTM(x.dim() == 3, "x must be a 3-dim tensor");

    torch::Tensor ret = torch::zeros(x.sizes(), torch::TensorOptions().dtype( scale_1.dtype() ).device( x.device() ) );
    scale_2dim(x, scale_1, scale_2, ret);
    return ret;
}

torch::Tensor i8linear_forward(
    void* handle,
    torch::Tensor x,
    torch::Tensor A) {
    CHECK_INPUT(x);
    CHECK_INPUT(A);
    AT_ASSERTM( (x.dim() == 3), "Matmul shape error (x.dim() != 3)" );
    AT_ASSERTM( (x.size(-1) == A.size(-1)), "Matmul shape error (x.size(-1) != A.size(-1))" );
    AT_ASSERTM( (A.dim() == 3), "Matmul shape error (A.dim() != 3)" );

    auto curr_device_idx = at::cuda::current_device();
    AT_ASSERTM( (x.device().index() == curr_device_idx), "Matmul device error (x not on current device)" );
    AT_ASSERTM( (A.device().index() == curr_device_idx), "Matmul device error (A not on current device)" );

    // calc batch_num
    int64_t num_batch_x = x.size(0);
    int64_t num_batch_A = A.size(0);
    int64_t num_batch = 0;
    if (num_batch_A == num_batch_x) {
        num_batch = num_batch_A;
    } else {
        if (num_batch_A == 1) num_batch = num_batch_x;
        else if (num_batch_x == 1) num_batch = num_batch_A;
        else {
            AT_ASSERTM( (num_batch_A == num_batch_x), "x.size(0) != A.size(0)" );
        }
    }

    // create return tensor
    auto size_x = x.sizes().vec();
    size_x[size_x.size() - 1] =  A.size(-2);
    size_x[0] = num_batch;
    torch::Tensor ret = torch::zeros(size_x, torch::TensorOptions().dtype(torch::kInt32).device( x.device() ) );
    
    int64_t stride_x = x.stride(0);
    int64_t stride_ret = ret.stride(0);
    int64_t stride_A = A.stride(0);

    if (num_batch_x == 1) stride_x = 0;
    if (num_batch_A == 1) stride_A = 0;
    if (num_batch == 0) stride_ret = 0;

    LtIgemm(
        (cublasLtHandle_t)handle,
        (int)A.size(-2), // m
        (int)x.size(-2), // n
        (int)x.size(-1), // k
        A.data_ptr<int8_t>(),
        stride_A,
        x.data_ptr<int8_t>(),
        stride_x,
        ret.data_ptr<int32_t>(),
        stride_ret,
        num_batch
    );
    return ret;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &i8linear_forward, "IGEMM forward (CUDA)");
    m.def("create_handle", &create_cublaslt_handle, "create cublasLt handle");
    m.def("scale_2d", &i8linear_scale, "scale in two dimensions");
}
