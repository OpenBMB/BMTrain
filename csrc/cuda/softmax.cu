#include "common/helper.h"
#include "common/reduce.cuh"

namespace {
template<typename scalar_t>
void softmax_forward_kernel(
    int32_t batch, int32_t n,
    const __restrict__ scalar_t *in,    // batch, n
    __restrict__ scalar_t *out          // batch, n
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;

    float local_max = -INFINITY;
    __shared__ float global_max = -INFINITY;

    for (int32_t i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            local_max = fmaxf(local_max,  blockReduceMax(in[base_idx + i]) );
        }
        else {
            local_max = fmaxf(local_max,  blockReduceMax(-INFINITY) );
        }
    }
    if (threadIdx.x == 0) {
        global_max = local_max;
    }
    __syncthreads();

    float local_exp_sum = 0f;
    __shared__ float global_exp_sum = 0f;
    for (int32_t i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            float v = expf( in[base_idx + i] - global_max );
            out[base_idx + i] = v;
            local_exp_sum += blockReduceSum( v );
        }
        else {
            local_exp_sum += blockReduceSum( 0f );
        }
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

template<typename scalar_t>
void softmax_backward_kernel(
    int32_t batch, int32_t n,
    const __restrict__ scalar_t *out,       // batch, n
    const __restrict__ scalar_t *grad_in,   // batch, n
    __restrict__ scalar_t *grad_out         // batch, n
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;


    float local_sum = 0f;
    __shared__ float global_sum = 0f;

    for (int32_t i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            local_sum += blockReduceSum( out[base_idx + i] * grad_in[base_idx + i] );
        } else {
            local_sum += blockReduceSum( 0f );
        }
    }
    if (threadIdx.x == 0) {
        global_sum = local_sum;
    }
    __syncthreads();
    for (int32_t i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            grad_out[base_idx + i] = out[base_idx + i] * (grad_in[base_idx + i] - global_sum);
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
    dim3 threads(n);
    ::softmax_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
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
    dim3 threads(n);
    ::softmax_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        batch, n, out, grad_in, grad_out
    );
}