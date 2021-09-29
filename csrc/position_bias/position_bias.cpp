#include <torch/extension.h>
#include "helper.h"
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
#define CHECK_LONG(x) AT_ASSERTM((x.dtype() == torch::kLong), #x " must be long")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void calc_bias_bucket(float log_max, float log_scale, int num_buckets, int max_buckets, torch::Tensor x, torch::Tensor out);
torch::Tensor position_bias(
    int num_buckets,
    int max_buckets,
    float log_maxexact,
    float log_scale,
    torch::Tensor relative_positions) {

    CHECK_CUDA(relative_positions);
    CHECK_LONG(relative_positions);
    AT_ASSERTM( (relative_positions.dim() == 2), "relative_positions shape error (relative_positions.dim() != 2)" );
    torch::Tensor viewd_position = relative_positions.view(-1);
    torch::Tensor out = torch::empty_like(relative_positions, torch::TensorOptions().dtype( torch::kLong ).device( relative_positions.device() ));
    torch::Tensor viewd_out = out.view(-1);
    calc_bias_bucket(log_maxexact, log_scale, num_buckets, max_buckets, viewd_position, viewd_out);
    return out;
}

