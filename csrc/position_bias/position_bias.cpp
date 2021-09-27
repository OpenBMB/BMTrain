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

torch::Tensor position_bias(
    int num_buckets,
    int num_heads,
    int max_distance
    bool bidirectional,
    torch::Tensor relative_positions) {
    
    CHECK_CUDA(relative_positions);
    CHECK_INT32(relative_positions);
    AT_ASSERTM( (relative_positions.dim() == 2), "relative_positions shape error (relative_positions.dim() != 2)" );

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
