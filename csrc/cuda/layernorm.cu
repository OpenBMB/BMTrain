#include "helper.h"
#include "reduce.cuh"

namespace {

template<typename scalar_t, bool rd_mean>
__global__ void layernorm_forward_kernel(
    int32_t batch, int32_t n,
    const __restrict__ scalar_t *mat, 
    __restrict__ scalar_t *out,
    float eps
) {
    float total_v = 0;
    float total_v_sq = 0;
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    __shared__ float mean_v;
    __shared__ float mean_v_sq;

    for (int i = 0; i < n; i += blockDim.x) {
        float v = 0;
        if (i + threadIdx.x < n) {
            v =  mat[base_idx + i];
        }
        if (rd_mean) total_v += blockReduceSum(v);
        total_v_sq += blockReduceSum(v * v);
    }

    if (threadIdx.x == 0) {
        mean_v = total_v / n;
        mean_v_sq = total_v_sq / n;
    }
    __syncthreads();

    float s_var = mean_v_sq;
    if (rd_mean) s_var -= mean_v * mean_v;

    for (int i = 0; i < max_size; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            float v = mat[base_idx + i];
            if (rd_mean) out[base_idx + i] = ((v - mean_v) * rsqrtf(s_var + eps));
            else out[base_idx + i] = (v * rsqrtf(s_var + eps));
            
        }
    }
}


template<typename scalar_t, bool rd_mean>
__global__ void layernorm_backward_kernel(
    int32_t batch, int32_t n,
    const __restrict__ scalar_t *x,         // b, n
    const __restrict__ scalar_t *grad_in,   // b, n
    __restrict__ scalar_t *grad_out,        // b, n
    float eps
) {
    float total_v = 0;
    float total_v_sq = 0;
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    __shared__ float mean_v;
    __shared__ float mean_v_sq;

    for (int i = 0; i < n; i += blockDim.x) {
        float v = 0;
        if (i + threadIdx.x < n) {
            v =  mat[base_idx + i];
        }
        if (rd_mean) total_v += blockReduceSum(v);
        total_v_sq += blockReduceSum(v * v);
    }

    if (threadIdx.x == 0) {
        mean_v = total_v / n;
        mean_v_sq = total_v_sq / n;
    }
    __syncthreads();

    float s_var = mean_v_sq;


    float local_grad_var = 0;
    float local_grad_mean = 0;
    float rsqrt_var = 0;
    __shared__ float global_grad_var = 0;
    __shared__ float global_grad_mean = 0;

    if (rd_mean) {
        s_var -= mean_v * mean_v;
        rsqrt_var = rsqrtf(s_var + eps);
        for (int i = 0; i < n; i += blockDim.x) {
            if (i + threadIdx.x < n) {
                local_grad_var += blockReduceSum(
                    grad_in[base_idx + i] * -0.5 * (x[base_idx + i] - mean_v) * rsqrt_var / s_var
                );
                local_grad_mean += blockReduceSum(
                    -grad_in[base_idx + i] * rsqrt_var
                );
            } else {
                local_grad_var += blockReduceSum(0f);
                local_grad_mean += blockReduceSum(0f);
            }
        }
        local_grad_mean -= 2 * local_grad_var * mean_v;
        if (threadIdx.x == 0) {
            global_grad_mean = local_grad_mean;
            global_grad_var = local_grad_var;
        }
    }
    else {
        rsqrt_var = rsqrtf(s_var + eps);
        for (int i = 0; i < n; i += blockDim.x) {
            if (i + threadIdx.x < n) {
                local_grad_var += blockReduceSum(
                    grad_in[base_idx + i] * -0.5 * x[base_idx + i] * rsqrt_var / s_var
                );
            } else {
                local_grad_var += blockReduceSum(0f);
            }
        }
        if (threadIdx.x == 0) {
            global_grad_var = local_grad_var;
        }
    }
    __syncthreads();

    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            grad_out[base_idx + i] = grad_in[base_idx + i] * rsqrt_var + ((global_grad_mean + global_grad_var * x[base_idx + i] * 2) / n);
        }
    }
}

}

template<typename scalar_t>
void layernorm_forward_launcher(
    int32_t batch, int32_t n,
    const __restrict__ scalar_t *mat, 
    __restrict__ scalar_t *out,
    bool rd_mean,
    float eps,
    cudaStream_t stream
) {
    dim3 blocks(batch);
    dim3 threads(::min(1024, n));

    if (rd_mean) {
        ::layernorm_forward_kernel<scalar_t, true><<<blocks, threads, 0, stream>>>(
            batch, n, mat, out, eps
        );
    } else {
        ::layernorm_forward_kernel<scalar_t, false><<<blocks, threads, 0, stream>>>(
            batch, n, mat, out, eps
        );
    }
}

template<typename scalar_t>
void layernorm_backward_launcher(
    int32_t batch, int32_t n,
    const __restrict__ scalar_t *x,         // b, n
    const __restrict__ scalar_t *grad_in,   // b, n
    __restrict__ scalar_t *grad_out,        // b, n
    bool rd_mean,
    float eps,
    cudaStream_t stream
) {
    dim3 blocks(batch);
    dim3 threads(::min(1024, n));
    if (rd_mean) {
        ::layernorm_backward_kernel<scalar_t, true><<<blocks, threads, 0, stream>>>(
            batch, n, x, grad_in, grad_out, eps
        );
    } else {
        ::layernorm_backward_kernel<scalar_t, false><<<blocks, threads, 0, stream>>>(
            batch, n, x, grad_in, grad_out, eps
        );
    }

}