#include "common/reduce.cuh"
#include <cuda_fp16.h>

__global__ void cu_inplace_mask(
    int32_t batch, int32_t n,
    half *x,                // (batch, n)
    const int8_t *mask      // (batch, n)
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            x[base_idx + i] = mask[base_idx + i] ? x[base_idx + i] : __float2half(0);
        }
    }
}