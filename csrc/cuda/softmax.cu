#include <cuda_fp16.h>
#include "common/reduce.cuh"

// grid <batch, m / 32>,    thread <32, 32>
__global__  void cu_softmax_forward(
    int32_t batch, int32_t n, int32_t m,
    const half *in,    // batch, n, m
    half *out          // batch, n, m
) {
    float local_max = -INFINITY;

    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            local_max = fmaxf((float)__ldg(in + base_mat_idx + i * m), local_max);
        }
    }

    local_max = transposeReduceMax(local_max);
    
    float local_sum = 0;
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            local_sum += expf((float)__ldg(in + base_mat_idx + i * m) - local_max);
        }
    }

    local_sum = transposeReduceSum(local_sum);
    
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            out[base_mat_idx + i * m] = __float2half( expf((float)__ldg(in + base_mat_idx + i * m) - local_max) / local_sum );
        }
    }
}

// grid <batch, m / 32>,    thread <32, 32>
__global__  void cu_softmax_backward(
    int32_t batch, int32_t n, int32_t m,
    const half *out,       // batch, n, m 
    const half *grad_in,   // batch, n, m
    half *grad_out         // batch, n, m
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_sum = 0;
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            local_sum += (float)__ldg(out + base_mat_idx + i * m) * (float)__ldg(grad_in + base_mat_idx + i * m);
        }
    }
    local_sum = transposeReduceSum(local_sum);
    
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            grad_out[base_mat_idx + i * m] = __float2half((float)__ldg(out + base_mat_idx + i * m) * ((float)__ldg(grad_in + base_mat_idx + i * m) - local_sum ) );
        }
    }
}