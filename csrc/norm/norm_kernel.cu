#include "helper.h"
#include "reduce.cuh"
#include <torch/extension.h>
#include <cuda_fp16.h>

namespace {

template<bool rd_mean>
__global__ void normalize_kernel(
    const torch::PackedTensorAccessor32<at::Half, 2, torch::RestrictPtrTraits> in, 
    torch::PackedTensorAccessor32<at::Half, 2, torch::RestrictPtrTraits> out,
    float eps) 
{
    int max_size = in.size(1);
    float total_v = 0;
    float total_v_sq = 0;

    __shared__ float mean_v;
    __shared__ float mean_v_sq;

    for (int i = 0; i < max_size; i += blockDim.x) {
        float v = 0;
        if (i + threadIdx.x < max_size) {
            v = in[blockIdx.x][i + threadIdx.x];
        }
        if (rd_mean) total_v += blockReduceSum(v);
        total_v_sq += blockReduceSum(v * v);
    }

    if (threadIdx.x == 0) {
        mean_v = total_v / max_size;
        mean_v_sq = total_v_sq / max_size;
    }
    __syncthreads();

    float s_var = mean_v_sq;
    if (rd_mean) s_var -= mean_v * mean_v;

    for (int i = 0; i < max_size; i += blockDim.x) {
        if (i + threadIdx.x < max_size) {
            float v = in[blockIdx.x][i + threadIdx.x];
            
            if (rd_mean) out[blockIdx.x][i + threadIdx.x] = __float2half((v - mean_v) * rsqrtf(s_var + eps));
            else out[blockIdx.x][i + threadIdx.x] = __float2half(v * rsqrtf(s_var + eps));
        }
    }
}

}

void normalize_cuda(
    const torch::Tensor in, 
    torch::Tensor out, 
    float eps,
    bool rd_mean
) {
    dim3 blocks(in.size(0));
    dim3 threads(::min( round_up(in.size(1), 32) , 1024));
    if (rd_mean) {
        normalize_kernel<true><<<blocks, threads>>>(
            in.packed_accessor32<at::Half, 2, torch::RestrictPtrTraits>(),
            out.packed_accessor32<at::Half, 2, torch::RestrictPtrTraits>(),
            eps
        );
    } else {
        normalize_kernel<false><<<blocks, threads>>>(
            in.packed_accessor32<at::Half, 2, torch::RestrictPtrTraits>(),
            out.packed_accessor32<at::Half, 2, torch::RestrictPtrTraits>(),
            eps
        );
    }
    
}