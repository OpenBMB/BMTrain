#include "common/reduce.cuh"
#include <cuda_fp16.h>

// block <batch_size>   thread<>
__global__ void cu_gelu_forward(
    int32_t batch, int32_t n,
    const half *mat,    // (batch, n)
    half *out           // (batch, n)
) {
    
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            float x = __half2float(mat[base_idx + i]);
            x = 0.5 * x * (1.0 + tanhf(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)));
            out[base_idx + i] = __float2half(x);
        }
    }
}

__global__ void cu_gelu_backward(
    int32_t batch, int32_t n,
    const half *grad_out,   // (batch, n)
    const half *mat,        // (batch, n)
    half *grad              // (batch, n)
) {
    
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            float v = __half2float(grad_out[base_idx + i]);
            float x = __half2float(mat[base_idx + i]);
            float gelu_grad;

            if (-5 < x && x < 5) {
                float x3 = x * x * x;
                float sech2 = 1.0 / coshf(0.797885 * x + 0.0356774 * x3);
                sech2 = sech2 * sech2;

                gelu_grad = 0.5 + (0.398942 * x + 0.0535161 * x3) * sech2 + 0.5 * tanhf(0.797885 * x + 0.0356774 * x3);
            }
            else {
                gelu_grad = x < 0 ? 0 : 1;
            }

            grad[base_idx + i] = __float2half(gelu_grad * v);
        }
    }
}
