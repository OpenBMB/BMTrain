#include "helper.h"
#include "reduce.cuh"
#include <torch/extension.h>
#include <cuda_fp16.h>

namespace {

__global__ void softmax_kernel(
    const torch::PackedTensorAccessor32<at::Half, 2, torch::RestrictPtrTraits> in, 
    torch::PackedTensorAccessor32<at::Half, 2, torch::RestrictPtrTraits> out) 
{
    int max_size = in.size(1);
    float local_max_v = -INFINITY;

    __shared__ float global_max_v;

    for (int i = 0; i < max_size; i += blockDim.x) {
        float v = (i + threadIdx.x < max_size) ? float(in[blockIdx.x][i + threadIdx.x]) : -INFINITY;
        v = blockReduceMax(v);
        local_max_v = fmaxf(local_max_v, v);
    }

    if (threadIdx.x == 0) {
        global_max_v = local_max_v;
    }
    __syncthreads();

    float local_sum_exp = 0;
    __shared__ float global_sum_exp;

    for (int i = 0; i < max_size; i += blockDim.x) {
        float v = 0;
        if (i + threadIdx.x < max_size) {
            v = in[blockIdx.x][i + threadIdx.x];
            v = expf(v - global_max_v);
            out[blockIdx.x][i + threadIdx.x] = v;
        }
        local_sum_exp += blockReduceSum(v);
    }

    if (threadIdx.x == 0) {
        global_sum_exp = local_sum_exp;
    }
    __syncthreads();

    for (int i = 0; i < max_size; i += blockDim.x) {
        if (i + threadIdx.x < max_size) {
            out[blockIdx.x][i + threadIdx.x] /= global_sum_exp;
        }
    }
}

}

void softmax_cuda(
    const torch::Tensor in, 
    torch::Tensor out
) {
    dim3 blocks(in.size(0));
    dim3 threads(::min( round_up(in.size(1), 32) , 1024));
    softmax_kernel<<<blocks, threads>>>(
        in.packed_accessor32<at::Half, 2, torch::RestrictPtrTraits>(), 
        out.packed_accessor32<at::Half, 2, torch::RestrictPtrTraits>()
    );
}