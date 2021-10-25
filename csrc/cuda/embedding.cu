#include "common/reduce.cuh"
#include <cuda_fp16.h>

// block <batch, m / 32>,   thread <32, 32>
__global__ void cu_embedding_forward(
    int32_t batch, int32_t n, int32_t m,
    const int32_t *ids,     // (batch, m)
    const half *weights,    // (vocab_size, n)
    half *out               // (batch, n, m)
) {
    __shared__ half shared[WARP_SZ][WARP_SZ + 1];

    int32_t col_in_idx = blockIdx.y * WARP_SZ + threadIdx.y;
    int32_t col_out_idx = blockIdx.y * WARP_SZ + threadIdx.x;
    const half *base_weight =  weights + (col_in_idx < m ? (ids[blockIdx.x * m + col_in_idx] * n) : 0) + threadIdx.x;

    int32_t base_out_idx = blockIdx.x * n * m + threadIdx.y * m + blockIdx.y * WARP_SZ + threadIdx.x;
    for (int i = 0; i < n; i += WARP_SZ) {
        if (i + threadIdx.x < n) {
            shared[threadIdx.y][threadIdx.x] = __ldg(base_weight + i);
        } else {
            shared[threadIdx.y][threadIdx.x] = __float2half(0);
        }
        // load multiple data from weights
        __syncthreads();
        // write multiple data to out (blockIdx.x, i + threadIdx.x, col_idx)
        if (i + threadIdx.y < n && col_out_idx < m) {
            out[ base_out_idx + i * m ] = shared[threadIdx.x][threadIdx.y];
        }
        __syncthreads();
    }
}

// block <batch, m / 1024>    thread<1024>
__global__ void cu_embedding_backward_stage1(
    int32_t batch, int32_t n, int32_t m,
    const half *grad_out,           // (batch * n, m)
    const int32_t *argsort_ids,     // (batch, n)
    const int32_t *sorted_ids,      // (batch, n)
    half *grad,                     // (vocab_size, m)
    half *aux_grad,                 // (batch, m)
    int32_t *aux_grad_idx           // (batch)
) {
    float sum = 0;
    int32_t baes_n_idx = blockIdx.x * n;
    int32_t col = blockIdx.y * WARP_SZ * WARP_SZ + threadIdx.x;

    if (col < m) {
        for (int i = 0; i < n; ++ i) {
            float v = (float)(grad_out[ argsort_ids[baes_n_idx + i] * m + col ]);
            sum += v;
            if (i + 1 == n) {
                aux_grad[blockIdx.x * m + col] = __float2half(sum);
                if (col == 0) aux_grad_idx[blockIdx.x] = __ldg(sorted_ids + baes_n_idx + i);
            }
            else if ( __ldg(sorted_ids + baes_n_idx + i) != __ldg(sorted_ids + baes_n_idx + i + 1)) {
                grad[ __ldg(sorted_ids + baes_n_idx + i) * m + col ] = __float2half(sum);
                sum = 0;
            }
        }
    }
}


// block <m / 1024>    thread<1024>
__global__ void cu_embedding_backward_stage2(
    int32_t batch, int32_t m,
    const half *aux_grad,           // (batch, m)
    const int32_t *aux_grad_idx,    // (batch)
    half *grad                      // (vocab_size, m)
) {
    float sum = 0;
    int32_t col = blockIdx.x * WARP_SZ * WARP_SZ + threadIdx.x;
    if (col < m) {
        for (int i = 0; i < batch; ++ i) {
            float v = (float)(aux_grad[i * m + col]);
            sum += v;
            if (i + 1 == batch || __ldg(aux_grad_idx + i) != __ldg(aux_grad_idx + i + 1)) {
                float v2 = (float)(grad[ __ldg(aux_grad_idx + i) * m + col ]);
                grad[ __ldg(aux_grad_idx + i) * m + col ] = __float2half(v2 + sum);
                sum = 0;
            }
        }
    }
}