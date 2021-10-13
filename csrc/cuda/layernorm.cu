#include "common/helper.h"
#include "common/reduce.cuh"
#include "ops/layernorm.h"

namespace {


__device__ inline float read_float(const float *ptr) {
    return __ldg(ptr);
}

__device__ inline float read_float(const half *ptr) {
    return __half2float( __ldg(ptr) );
}

template<typename scalar_t, bool rd_mean>
__global__ void layernorm_forward_kernel(
    int32_t batch, int32_t n,
    const scalar_t *mat, 
    scalar_t *out,
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
            v = read_float(mat + base_idx + i);
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

    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            float v = read_float(mat + base_idx + i);
            if (rd_mean) out[base_idx + i] = ((v - mean_v) * rsqrtf(s_var + eps));
            else out[base_idx + i] = (v * rsqrtf(s_var + eps));
        }
    }
}

template<bool rd_mean>
__global__ void layernorm_forward_kernel(
    int32_t batch, int32_t n,
    const half *mat, 
    half *out,
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
            v = read_float(mat + base_idx + i);
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

    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            float v = read_float(mat + base_idx + i);

            if (rd_mean) out[base_idx + i] = __float2half((v - mean_v) * rsqrtf(s_var + eps));
            else out[base_idx + i] = __float2half(v * rsqrtf(s_var + eps));
        }
    }
}

template<typename scalar_t, bool rd_mean>
__global__ void layernorm_backward_kernel(
    int32_t batch, int32_t n,
    const scalar_t *x,         // b, n
    const scalar_t *grad_in,   // b, n
    scalar_t *grad_out,        // b, n
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
            v =  read_float(x + base_idx + i);
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
    __shared__ float global_grad_var;
    __shared__ float global_grad_mean;

    if (rd_mean) {
        s_var -= mean_v * mean_v;
        rsqrt_var = rsqrtf(s_var + eps);
        for (int i = 0; i < n; i += blockDim.x) {
            float var = 0;
            float mean = 0;
            if (i + threadIdx.x < n) {
                var = read_float(grad_in + base_idx + i) * -0.5 * (read_float(x + base_idx + i) - mean_v) * rsqrt_var / s_var;
                mean = - read_float(grad_in + base_idx + i) * rsqrt_var;
                
            }
            local_grad_var += blockReduceSum(var);
            local_grad_mean += blockReduceSum(mean);
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
            float v = 0;
            if (i + threadIdx.x < n) {
                v = read_float(grad_in + base_idx + i) * -0.5 * read_float(x + base_idx + i) * rsqrt_var / s_var;
            }
            local_grad_var += blockReduceSum(v);
        }
        if (threadIdx.x == 0) {
            global_grad_var = local_grad_var;
        }
    }
    __syncthreads();

    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            grad_out[base_idx + i] = read_float(grad_in + base_idx + i) * rsqrt_var + ((global_grad_mean + global_grad_var * read_float(x + base_idx + i) * 2) / n);
        }
    }
}


template<bool rd_mean>
__global__ void layernorm_backward_kernel(
    int32_t batch, int32_t n,
    const half *x,         // b, n
    const half *grad_in,   // b, n
    half *grad_out,        // b, n
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
            v =  read_float(x + base_idx + i);
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
    __shared__ float global_grad_var;
    __shared__ float global_grad_mean;

    if (rd_mean) {
        s_var -= mean_v * mean_v;
        rsqrt_var = rsqrtf(s_var + eps);
        for (int i = 0; i < n; i += blockDim.x) {
            float var = 0;
            float mean = 0;
            if (i + threadIdx.x < n) {
                var = read_float(grad_in + base_idx + i) * -0.5 * (read_float(x + base_idx + i) - mean_v) * rsqrt_var / s_var;
                mean = - read_float(grad_in + base_idx + i) * rsqrt_var;
            }
            local_grad_var += blockReduceSum(var);
            local_grad_mean += blockReduceSum(mean);
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
            float v = 0;
            if (i + threadIdx.x < n) {
                v = read_float(grad_in + base_idx + i) * -0.5 * read_float(x + base_idx + i) * rsqrt_var / s_var;
            }
            local_grad_var += blockReduceSum(v);
        }
        if (threadIdx.x == 0) {
            global_grad_var = local_grad_var;
        }
    }
    __syncthreads();

    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            grad_out[base_idx + i] = __float2half(read_float(grad_in + base_idx + i) * rsqrt_var + ((global_grad_mean + global_grad_var * read_float(x + base_idx + i) * 2) / n));
        }
    }
}

}

template<typename scalar_t>
void layernorm_forward_launcher(
    int32_t batch, int32_t n,
    const scalar_t *mat, 
    scalar_t *out,
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

void layernorm_forward_launcher(
    int32_t batch, int32_t n,
    const half *mat, 
    half *out,
    bool rd_mean,
    float eps,
    cudaStream_t stream
) {
    dim3 blocks(batch);
    dim3 threads(::min(1024, n));

    if (rd_mean) {
        ::layernorm_forward_kernel<true><<<blocks, threads, 0, stream>>>(
            batch, n, mat, out, eps
        );
    } else {
        ::layernorm_forward_kernel<false><<<blocks, threads, 0, stream>>>(
            batch, n, mat, out, eps
        );
    }
}

void layernorm_forward(int32_t batch, int32_t n, const half *mat, half *out, bool rd_mean, float eps, cudaStream_t stream) {
    layernorm_forward_launcher(batch, n, mat, out, rd_mean, eps, stream);
}
void layernorm_forward(int32_t batch, int32_t n, const float *mat, float *out, bool rd_mean, float eps, cudaStream_t stream) {
    layernorm_forward_launcher<float>(batch, n, mat, out, rd_mean, eps, stream);
}

template<typename scalar_t>
void layernorm_backward_launcher(
    int32_t batch, int32_t n,
    const scalar_t *x,         // b, n
    const scalar_t *grad_in,   // b, n
    scalar_t *grad_out,        // b, n
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

void layernorm_backward_launcher(
    int32_t batch, int32_t n,
    const half *x,         // b, n
    const half *grad_in,   // b, n
    half *grad_out,        // b, n
    bool rd_mean,
    float eps,
    cudaStream_t stream
) {
    dim3 blocks(batch);
    dim3 threads(::min(1024, n));
    if (rd_mean) {
        ::layernorm_backward_kernel<true><<<blocks, threads, 0, stream>>>(
            batch, n, x, grad_in, grad_out, eps
        );
    } else {
        ::layernorm_backward_kernel<false><<<blocks, threads, 0, stream>>>(
            batch, n, x, grad_in, grad_out, eps
        );
    }

}

void layernorm_backward(
    int32_t batch, int32_t n,
    const half *x, const half *grad_in,
    half *grad_out, bool rd_mean, float eps, cudaStream_t stream
) {
    layernorm_backward_launcher(batch, n, x, grad_in, grad_out, rd_mean, eps, stream);
}

void layernorm_backward(
    int32_t batch, int32_t n,
    const float *x, const float *grad_in,
    float *grad_out, bool rd_mean, float eps, cudaStream_t stream
) {
    layernorm_backward_launcher<float>(batch, n, x, grad_in, grad_out, rd_mean, eps, stream);
}
