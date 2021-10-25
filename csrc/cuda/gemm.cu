#include "common/reduce.cuh"
#include <cuda_fp16.h>

__global__ void cu_gemm_round(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,       // b, n, m
    const half *scale,     // b, n
    int8_t *out
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;   // mat[batch][n][m], scale[batch][n]
    float local_scale = (float)scale[blockIdx.x * n + blockIdx.y];

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x) {
        out[base_idx + i] = (int8_t)nearbyintf((float)__ldg(mat + base_idx + i) / local_scale); 
    }
}


__global__ void cu_gemm_round_transpose(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,       // b, n, m
    const half *scale,     // b, m
    int8_t *out
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;   // mat[batch][n][m], scale[batch][m]

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x) {
        out[base_idx + i] = (int8_t)nearbyintf((float)mat[base_idx + i] / (float)__ldg(scale + blockIdx.x * m + i)); 
    }
}

__global__ void cu_gemm_scale(
    int32_t batch, int32_t n, int32_t m,
    const int32_t *mat,        // b, n, m
    const half *scale_x,   // b, n
    const half *scale_y,   // b, m
    half *out,
    bool broad_cast_x, bool broad_cast_y
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;
    float scale_x_value = 0;
    if (broad_cast_x) {
        scale_x_value = (float)__ldg(scale_x + blockIdx.y);
    } else {
        scale_x_value = (float)__ldg(scale_x + blockIdx.x * n + blockIdx.y);
    }

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x){
        if (broad_cast_y) {
            out[base_idx + i] = __float2half((float)(mat[base_idx + i]) * scale_x_value * (float)__ldg(scale_y + i));
        }
        else {
            out[base_idx + i] = __float2half((float)(mat[base_idx + i]) * scale_x_value * (float)__ldg(scale_y + blockIdx.x * m + i));
        }
    }
}


__global__ void cu_gemm_calc_scale(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,        // b, n, m
    half *out  // b, n
) {
    float local_max = 0;

    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;
    for (int32_t i = 0; i < m; i += blockDim.x){
        int32_t offset = threadIdx.x + i;
        float v = 0;
        if (offset < m) {
            v = fabsf((float)(mat[base_idx + offset]));
        }
        local_max = fmaxf(v, local_max);
    }
    local_max = blockReduceMax(local_max);

    if (threadIdx.x == 0) {
        out[ blockIdx.x * n + blockIdx.y ] = __float2half(local_max / 127.0);
    }
}

__global__ void cu_gemm_calc_scale_transpose(
    int32_t batch, int32_t n, int32_t m,
    const half *in,        // b, n, m
    half *out  // b, m
) {
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t base_idx = (blockIdx.x * n + threadIdx.y) * m + col_idx;
    
    float local_max = 0.0;
    for (int32_t i = 0; i < n; i += WARP_SZ) {
        // put & transpose
        if (i + threadIdx.y < n && col_idx < m) {
            local_max = fmaxf(fabsf((float)(in[base_idx + i * m])), local_max);
        }
    }
    local_max = transposeReduceMax(local_max);
    if (threadIdx.y == 0 && col_idx < m) {
        out[blockIdx.x * m + col_idx] = __float2half(local_max / 127.0);
    }
}