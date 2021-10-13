#include "common/helper.h"
#include "common/reduce.cuh"

namespace {

__device__ inline float read_float(const float *ptr) {
    return __ldg(ptr);
}

__device__ inline float read_float(const half *ptr) {
    return __half2float( __ldg(ptr) );
}

template<typename scalar_t>
__global__  void softmax_forward_kernel(
    int32_t batch, int32_t n,
    const scalar_t *in,    // batch, n
    scalar_t *out          // batch, n
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;

    float local_max = -INFINITY;
    __shared__ float global_max;

    for (int32_t i = 0; i < n; i += blockDim.x) {
        float v = -INFINITY;
        if (i + threadIdx.x < n) {
            v = read_float(in + base_idx + i);
        }
        local_max = fmaxf(local_max,  blockReduceMax(v) );
    }
    if (threadIdx.x == 0) {
        global_max = local_max;
    }
    __syncthreads();

    float local_exp_sum = 0.0;
    __shared__ float global_exp_sum;
    for (int32_t i = 0; i < n; i += blockDim.x) {
        float v = 0;
        if (i + threadIdx.x < n) {
            v = expf( read_float(in + base_idx + i) - global_max );
            out[base_idx + i] = v;
        }
        local_exp_sum += blockReduceSum( v );
    }
    if (threadIdx.x == 0) {
        global_exp_sum = local_exp_sum;
    }
    __syncthreads();
    for (int32_t i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            out[base_idx + i] /= global_exp_sum;
        }
    }
}

__global__  void softmax_forward_kernel(
    int32_t batch, int32_t n,
    const half *in,    // batch, n
    half *out          // batch, n
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;

    float local_max = -INFINITY;
    __shared__ float global_max;

    for (int32_t i = 0; i < n; i += blockDim.x) {
        float v = -INFINITY;
        if (i + threadIdx.x < n) {
            v = read_float(in + base_idx + i);
        }
        local_max = fmaxf(local_max,  blockReduceMax(v) );
    }
    if (threadIdx.x == 0) {
        global_max = local_max;
    }
    __syncthreads();

    float local_exp_sum = 0.0;
    __shared__ float global_exp_sum;
    for (int32_t i = 0; i < n; i += blockDim.x) {
        float v = 0;
        if (i + threadIdx.x < n) {
            v = expf( read_float(in + base_idx + i) - global_max );
            out[base_idx + i] = __float2half(v);
        }
        local_exp_sum += blockReduceSum( v );
    }
    if (threadIdx.x == 0) {
        global_exp_sum = local_exp_sum;
    }
    __syncthreads();
    for (int32_t i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            out[base_idx + i] = __float2half(__half2float(out[base_idx + i]) / global_exp_sum);
        }
    }
}

template<typename scalar_t>
__global__  void softmax_backward_kernel(
    int32_t batch, int32_t n,
    const scalar_t *out,       // batch, n
    const scalar_t *grad_in,   // batch, n
    scalar_t *grad_out         // batch, n
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;


    float local_sum = 0.0;
    __shared__ float global_sum;

    for (int32_t i = 0; i < n; i += blockDim.x) {
        float v = 0;
        if (i + threadIdx.x < n) {
            v = read_float(out + base_idx + i) * read_float(grad_in + base_idx + i);
        }
        local_sum += blockReduceSum( v );
    }
    if (threadIdx.x == 0) {
        global_sum = local_sum;
    }
    __syncthreads();
    for (int32_t i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            grad_out[base_idx + i] = read_float(out + base_idx + i) * (read_float(grad_in + base_idx + i) - global_sum);
        }
    }
}
__global__  void softmax_backward_kernel(
    int32_t batch, int32_t n,
    const half *out,       // batch, n
    const half *grad_in,   // batch, n
    half *grad_out         // batch, n
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;


    float local_sum = 0.0;
    __shared__ float global_sum;

    for (int32_t i = 0; i < n; i += blockDim.x) {
        float v = 0;
        if (i + threadIdx.x < n) {
            v = read_float(out + base_idx + i) * read_float(grad_in + base_idx + i);
            local_sum += blockReduceSum( v );
        }
    }
    if (threadIdx.x == 0) {
        global_sum = local_sum;
    }
    __syncthreads();
    for (int32_t i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            grad_out[base_idx + i] = __float2half(read_float(out + base_idx + i) * (read_float(grad_in + base_idx + i) - global_sum));
        }
    }
}

}

template<typename scalar_t>
void softmax_forward_launcher(
    int32_t batch, int32_t n,
    const scalar_t *in,
    scalar_t *out,
    cudaStream_t stream
) {
    dim3 blocks(batch);
    dim3 threads(round_up(n, WARP_SZ));
    ::softmax_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        batch, n, in, out
    );
}

void softmax_forward_launcher(
    int32_t batch, int32_t n,
    const half *in,
    half *out,
    cudaStream_t stream
) {
    dim3 blocks(batch);
    dim3 threads(round_up(n, WARP_SZ));
    ::softmax_forward_kernel<<<blocks, threads, 0, stream>>>(
        batch, n, in, out
    );
}

template<typename scalar_t>
void softmax_backward_launcher(
    int32_t batch, int32_t n,
    const scalar_t *out,
    const scalar_t *grad_in,
    scalar_t *grad_out,
    cudaStream_t stream
) {
    dim3 blocks(batch);
    dim3 threads(round_up(n, WARP_SZ));
    ::softmax_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        batch, n, out, grad_in, grad_out
    );
}

void softmax_backward_launcher(
    int32_t batch, int32_t n,
    const half *out,
    const half *grad_in,
    half *grad_out,
    cudaStream_t stream
) {
    dim3 blocks(batch);
    dim3 threads(round_up(n, WARP_SZ));
    ::softmax_backward_kernel<<<blocks, threads, 0, stream>>>(
        batch, n, out, grad_in, grad_out
    );
}

void softmax_forward(int32_t batch, int32_t n, const half *in, half *out, cudaStream_t stream) {
    softmax_forward_launcher(batch, n, in, out, stream);
}

void softmax_forward(int32_t batch, int32_t n, const float *in, float *out, cudaStream_t stream) {
    softmax_forward_launcher<float>(batch, n, in, out, stream);
}

void softmax_backward(int32_t batch, int32_t n, const half *out, const half *grad_in, half *grad_out, cudaStream_t stream) {
    softmax_backward_launcher(batch, n, out, grad_in, grad_out, stream);
}

void softmax_backward(int32_t batch, int32_t n, const float *out, const float *grad_in, float *grad_out, cudaStream_t stream) {
    softmax_backward_launcher<float>(batch, n, out, grad_in, grad_out, stream);
}