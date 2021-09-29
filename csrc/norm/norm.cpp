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


void normalize_cuda(const torch::Tensor in,  torch::Tensor out,  float eps, bool rd_mean);

torch::Tensor normalize_forward(torch::Tensor input, float eps, bool rd_mean) {
    CHECK_CUDA(input);
    AT_ASSERTM(input.dtype() == torch::kHalf, "Input must be a Half Tensor");
    AT_ASSERTM(input.ndimension() == 2, "Input must be a 2-DIM Tensor");
    torch::Tensor output = torch::empty_like(input, torch::TensorOptions().dtype( input.dtype() ).device( input.device() ));
    normalize_cuda(input, output, eps, rd_mean);
    return output;
}