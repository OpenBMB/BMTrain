#include "reduce.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include "bfloat16.cuh"

namespace {
// blocks <m>,      threads<1024>
__global__ void cross_entropy_forward_fp16(
    int64_t n,
    const half *input,      // (m, n)
    const int32_t *target,  // (m)
    half *softmax,          // (m, n)
    float *output,          // (m)
    int32_t ignore_index
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

    if (threadIdx.x == 0) {
        if (target[blockIdx.x] != ignore_index) {
            output[blockIdx.x] = -__half2float(input[base_idx + target[blockIdx.x]]) + local_max + logf(local_sum);
        } else {
            output[blockIdx.x] = 0;
        }
    }
}

// blocks <m>,      threads<1024>
__global__ void cross_entropy_backward_inplace_fp16(
    int64_t n,
    const float *grad_output,   // (m)
    const int32_t *target,      // (m)
    half *x,                    // (m, n)
    int32_t ignore_index
) {
    int64_t base_idx = blockIdx.x * n;

    int32_t t = target[blockIdx.x];
    if (t == ignore_index) {
        half v = __float2half(0.);
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            x[base_idx + i] = v;
        }
    }
    else {
        half v = __float2half(grad_output[blockIdx.x]);
        __syncthreads();
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            x[base_idx + i] = i==t ? __hsub(__hmul(x[base_idx + i], v), v) : __hmul(x[base_idx + i], v);
        }
    }
}

// blocks <m>,      threads<1024>
__global__ void cross_entropy_forward_bf16(
    int64_t n,
    std::uintptr_t input_ptr,      // (m, n)
    const int32_t *target,  // (m)
    std::uintptr_t softmax_ptr,          // (m, n)
    float *output,          // (m)
    int32_t ignore_index
) {
#ifdef BF16_SUPPORT
    __nv_bfloat16* input = reinterpret_cast<__nv_bfloat16*>(input_ptr);
    __nv_bfloat16* softmax = reinterpret_cast<__nv_bfloat16*>(softmax_ptr);
    int64_t base_idx = blockIdx.x * n;

    float local_max = -INFINITY;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(__bfloat162float(input[base_idx + i]), local_max);
    }

    local_max = fmaxf(block_allreduce_max(local_max), -1e6);
    
    float local_sum = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += expf(__bfloat162float(input[base_idx + i]) - local_max);
    }
    local_sum = block_allreduce_sum(local_sum) + 1e-10; // avoid nan
    
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        softmax[base_idx + i] = __float2bfloat16( expf(__bfloat162float(input[base_idx + i]) - local_max) / local_sum );
    }

    if (threadIdx.x == 0) {
        if (target[blockIdx.x] != ignore_index) {
            output[blockIdx.x] = -__bfloat162float(input[base_idx + target[blockIdx.x]]) + local_max + logf(local_sum);
        } else {
            output[blockIdx.x] = 0;
        }
    }
#endif
}

// blocks <m>,      threads<1024>
__global__ void cross_entropy_backward_inplace_bf16(
    int64_t n,
    const float *grad_output,   // (m)
    const int32_t *target,      // (m)
    std::uintptr_t x_ptr,                    // (m, n)
    int32_t ignore_index
) {
#ifdef BF16_SUPPORT
    __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(x_ptr);
    int64_t base_idx = blockIdx.x * n;

    int32_t t = target[blockIdx.x];
    if (t == ignore_index) {
        __nv_bfloat16 v = __float2bfloat16(0.);
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            x[base_idx + i] = v;
        }
    }
    else {
        #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 v = __float2bfloat16(grad_output[blockIdx.x]);
        __syncthreads();
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            x[base_idx + i] = i==t ? __hsub(__hmul(x[base_idx + i], v), v) : __hmul(x[base_idx + i], v);
        }
        #else
        float v = grad_output[blockIdx.x];
        __syncthreads();
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            x[base_idx + i] = __float2bfloat16(i==t ? (__bfloat162float(x[base_idx + i])*v)-v : __bfloat162float(x[base_idx + i])*v);
        }
        #endif
    }
#endif
}

}

void cross_entropy_forward_fp16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t input,
    std::uintptr_t target,
    std::uintptr_t softmax,
    std::uintptr_t output,
    int32_t ignore_index,
    std::uintptr_t stream
) {
    auto input_ptr = reinterpret_cast<half*>(input);
    auto target_ptr = reinterpret_cast<int32_t*>(target);
    auto softmax_ptr = reinterpret_cast<half*>(softmax);
    auto output_ptr = reinterpret_cast<float*>(output);
    int32_t threads = 1024;
    cross_entropy_forward_fp16<<<m, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(n, input_ptr, target_ptr, softmax_ptr, output_ptr, ignore_index);
}

void cross_entropy_backward_inplace_fp16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t grad_output,
    std::uintptr_t target,
    std::uintptr_t x,
    int32_t ignore_index,
    std::uintptr_t stream
) {
    auto output_ptr = reinterpret_cast<float*>(grad_output);
    auto target_ptr = reinterpret_cast<int32_t*>(target);
    auto x_ptr = reinterpret_cast<half*>(x);
    int32_t threads = 1024;
    cross_entropy_backward_inplace_fp16<<<m, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(n, output_ptr, target_ptr, x_ptr, ignore_index);
}

void cross_entropy_forward_bf16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t input,
    std::uintptr_t target,
    std::uintptr_t softmax,
    std::uintptr_t output,
    int32_t ignore_index,
    std::uintptr_t stream
) {
    auto target_ptr = reinterpret_cast<int32_t*>(target);
    auto output_ptr = reinterpret_cast<float*>(output);
    int32_t threads = 1024;
    cross_entropy_forward_bf16<<<m, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(n, input, target_ptr, softmax, output_ptr, ignore_index);
}

void cross_entropy_backward_inplace_bf16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t grad_output,
    std::uintptr_t target,
    std::uintptr_t x,
    int32_t ignore_index,
    std::uintptr_t stream
) {
    auto output_ptr = reinterpret_cast<float*>(grad_output);
    auto target_ptr = reinterpret_cast<int32_t*>(target);
    int32_t threads = 1024;
    cross_entropy_backward_inplace_bf16<<<m, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(n, output_ptr, target_ptr, x, ignore_index);
}