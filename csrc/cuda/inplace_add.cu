#include "common/reduce.cuh"
#include <cuda_fp16.h>

// block <batch_size>,  thread<1024>
__global__ void cu_inplace_add(
    int32_t batch, int32_t n,
    half *x,        // (batch, n)
    const half *y   // (batch, n)
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            x[base_idx + i] = __hadd(x[base_idx + i], y[base_idx + i]);;
        }
    }
}