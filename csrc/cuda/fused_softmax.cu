#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "reduce.cuh"

namespace {
// blocks <m>,      threads<1024>
__global__ void fused_softmax_forward(
    int64_t n,
    const half *input,      // (m, n)
    half *softmax           // (m, n)
) {
    int64_t base_idx = blockIdx.x * n;

    float local_max = -INFINITY;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(__half2float(input[base_idx + i]), local_max);
    }

    local_max = fmaxf(block_allreduce_max(local_max), -1e6);
    
    float local_sum = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += expf(__half2float(input[base_idx + i]) - local_max);
    }
    local_sum = block_allreduce_sum(local_sum) + 1e-10; // avoid nan
    
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        softmax[base_idx + i] = __float2half( expf(__half2float(input[base_idx + i]) - local_max) / local_sum );
    }
}

// blocks <m>,      threads<1024>
__global__ void fused_softmax_backward(
    int64_t n,
    const half *grad_output,    // (m, n)
    const half *softmax,        // (m, n)
    half *grad_input            // (m, n)
) {
    int64_t base_idx = blockIdx.x * n;

    float local_sum = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += __half2float(grad_output[base_idx + i]) * __half2float(softmax[base_idx + i]);
    }
    local_sum = block_allreduce_sum(local_sum);
    
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        grad_input[base_idx + i] = __float2half( __half2float(softmax[base_idx + i]) * ( __half2float(grad_output[base_idx + i]) - local_sum ) );
    }
}

// blocks <m>,      threads<1024>
__global__ void fused_softmax_forward_inplace(
    int64_t n,
    half *x                     // (m, n)
) {
    int64_t base_idx = blockIdx.x * n;

    float local_max = -INFINITY;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(__half2float(x[base_idx + i]), local_max);
    }
    local_max = fmaxf(block_allreduce_max(local_max), -1e6);
    
    float local_sum = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += expf(__half2float(x[base_idx + i]) - local_max);
    }
    local_sum = block_allreduce_sum(local_sum) + 1e-10; // avoid nan
    
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        x[base_idx + i] = __float2half( expf(__half2float(x[base_idx + i]) - local_max) / local_sum );
    }
}

// blocks <m>,      threads<1024>
__global__ void fused_softmax_backward_inplace(
    int64_t n,
    const half *grad_output,    // (m, n)
    half* x                     // (m, n)
) {
    int64_t base_idx = blockIdx.x * n;

    float local_sum = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += __half2float(grad_output[base_idx + i]) * __half2float(x[base_idx + i]);
    }
    local_sum = block_allreduce_sum(local_sum);
    
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        x[base_idx + i] = __float2half( __half2float(x[base_idx + i]) * ( __half2float(grad_output[base_idx + i]) - local_sum ) );
    }
}

}

void fused_softmax_forward_launcher(
    int32_t m, int32_t n,
    const torch::Tensor &input,
    torch::Tensor &softmax
) {
    auto input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    auto softmax_ptr = reinterpret_cast<half*>(softmax.data_ptr<at::Half>());
    int32_t threads = 1024;
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_softmax_forward<<<m, threads, 0, stream.stream()>>>(n, input_ptr, softmax_ptr);
}

void fused_softmax_backward_launcher(
    int32_t m, int32_t n,
    const torch::Tensor &grad_output,
    const torch::Tensor &softmax,
    torch::Tensor &grad_input
) {
    auto output_ptr = reinterpret_cast<half*>(grad_output.data_ptr<at::Half>());
    auto softmax_ptr = reinterpret_cast<half*>(softmax.data_ptr<at::Half>());
    auto input_ptr = reinterpret_cast<half*>(grad_input.data_ptr<at::Half>());
    int32_t threads = 1024;
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_softmax_backward<<<m, threads, 0, stream.stream()>>>(n, output_ptr, softmax_ptr, input_ptr);
}

void fused_softmax_forward_inplace_launcher(
    int32_t m, int32_t n,
    torch::Tensor &x
) {
    auto x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());
    int32_t threads = 1024;
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_softmax_forward_inplace<<<m, threads, 0, stream.stream()>>>(n, x_ptr);
}

void fused_softmax_backward_inplace_launcher(
    int32_t m, int32_t n,
    const torch::Tensor &grad_output,
    torch::Tensor &x
) {
    auto output_ptr = reinterpret_cast<half*>(grad_output.data_ptr<at::Half>());
    auto x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());
    int32_t threads = 1024;
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_softmax_backward_inplace<<<m, threads, 0, stream.stream()>>>(n, output_ptr, x_ptr);
}