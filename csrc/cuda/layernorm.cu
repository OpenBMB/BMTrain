#include "common/reduce.cuh"
#include <cuda_fp16.h>

// block <batch_idx, offset_m/32>   thread <32, 32>
__global__ void cu_layernorm_forward(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,    // (batch, n, m)
    half *out,
    float eps,
    bool rd_mean
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_total_v = 0.0;
    float local_total_v2 = 0.0;
    for (int i = 0; i < n; i += WARP_SZ) {
        float v = 0;
        if (col_idx < m && i + threadIdx.y < n) {
            v = (float)__ldg(mat + base_mat_idx + i * m);
        }

        if (rd_mean) local_total_v += v;
        local_total_v2 += v * v;
    }

    local_total_v2 = transposeReduceSum(local_total_v2) / (float)n;
    if (rd_mean) {
        local_total_v = transposeReduceSum(local_total_v) / (float)n;
        local_total_v2 -= local_total_v * local_total_v;
    }

    local_total_v2 = rsqrtf(local_total_v2 + eps);

    float local_mean =  local_total_v;
    float local_var = local_total_v2;
    if (rd_mean) {
        for (int i = 0; i < n; i += WARP_SZ) {
            if (col_idx < m && i + threadIdx.y < n) {
                out[base_mat_idx + i * m] = __float2half(((float)__ldg(mat + base_mat_idx + i * m) - local_mean) * local_var);
            }
        }
    } else {
        for (int i = 0; i < n; i += blockDim.y) {
            if (col_idx < m && i + threadIdx.y < n) {
                out[base_mat_idx + i * m] = __float2half((float)__ldg(mat + base_mat_idx + i * m) * local_var);
            }
        }
    }
}

// block <batch_idx, offset_m/32>   thread <32, 32>
__global__ void cu_layernorm_forward_v(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,    // (batch, n, m)
    half *out,          // (batch, n, m)
    half *out_var,      // (batch, m)
    float eps
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_total_v2 = 0.0;
    for (int i = 0; i < n; i += WARP_SZ) {
        float v = 0;
        if (col_idx < m && i + threadIdx.y < n) {
            v = (float)__ldg(mat + base_mat_idx + i * m);
        }
        local_total_v2 += v * v;
    }
    float local_var = rsqrtf(transposeReduceSum(local_total_v2) / (float)n + eps);

    if (threadIdx.y == 0 && col_idx < m) out_var[blockIdx.x * m + col_idx] = __float2half(local_var);

    for (int i = 0; i < n; i += blockDim.y) {
        if (col_idx < m && i + threadIdx.y < n) {
            out[base_mat_idx + i * m] = __float2half((float)__ldg(mat + base_mat_idx + i * m) * local_var);
        }
    }
}

// block <batch_idx, offset_m/32>   thread <32, 32>
__global__ void cu_layernorm_forward_mv(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,    // (batch, n, m)
    half *out,
    half *out_mean,
    half *out_var,
    float eps
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_total_v = 0.0;
    float local_total_v2 = 0.0;
    for (int i = 0; i < n; i += WARP_SZ) {
        float v = 0;
        if (col_idx < m && i + threadIdx.y < n) {
            v = (float)__ldg(mat + base_mat_idx + i * m);
        }
        local_total_v += v;
        local_total_v2 += v * v;
    }

    local_total_v = transposeReduceSum(local_total_v) / (float)n;
    local_total_v2 = rsqrtf(transposeReduceSum(local_total_v2) / (float)n - local_total_v * local_total_v + eps);

    if (threadIdx.y == 0 && col_idx < m) {
        out_var[blockIdx.x * m + col_idx] = __float2half(local_total_v2);
        out_mean[blockIdx.x * m + col_idx] = __float2half(local_total_v);
    }

    float local_mean =  local_total_v;
    float local_var = local_total_v2;
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            out[base_mat_idx + i * m] = __float2half(((float)(mat[base_mat_idx + i * m]) - local_mean) * local_var);
        }
    }
}

// block <batch_idx, offset_m/32>   thread <32, 32>
__global__ void cu_layernorm_backward_v(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,        // (batch, n, m)
    const half *grad_in,    // (batch, n, m)
    const half *var,        // (batch, m) 
    half *grad_out
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_grad_var = 0;

    float local_var = col_idx < m ? (float)__ldg(var + blockIdx.x * m + col_idx) : 0.0;
    float n_half_rsqrt_v3 = -0.5 * local_var * local_var * local_var;

    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            local_grad_var += (float)__ldg(grad_in + base_mat_idx + i * m) * n_half_rsqrt_v3 * ((float)__ldg(mat + base_mat_idx + i * m));
        }
    }

    local_grad_var = transposeReduceSum(local_grad_var);

    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            grad_out[base_mat_idx + i * m] = __float2half(
                (float)__ldg(grad_in + base_mat_idx + i * m) * local_var +
                ((local_grad_var * (float)__ldg(mat + base_mat_idx + i * m) * 2) / (float)n)
            );
        }
    }
}

// block <batch_idx, offset_m/32>   thread <32, 32>
__global__ void cu_layernorm_backward_mv(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,        // (batch, n, m)
    const half *grad_in,    // (batch, n, m)
    const half *mean,       // (batch, m)
    const half *var,        // (batch, m) 
    half *grad_out
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;


    float local_grad_var = 0;
    float local_grad_mean = 0;

    float local_mean =  col_idx < m ? (float)__ldg(mean + blockIdx.x * m + col_idx) : 0.0;
    float local_var = col_idx < m ? (float)__ldg(var + blockIdx.x * m + col_idx) : 0.0;

    float n_half_rsqrt_v3 = -0.5 * local_var * local_var * local_var;


    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            float gi = (float)__ldg(grad_in + base_mat_idx + i * m);
            local_grad_var += gi * n_half_rsqrt_v3 * ((float)__ldg(mat + base_mat_idx + i * m) - local_mean);
            local_grad_mean += -gi * local_var;
        }
    }

    local_grad_var = transposeReduceSum(local_grad_var);
    local_grad_mean = transposeReduceSum(local_grad_mean);

    local_grad_mean -= 2 * local_grad_var * local_mean;

    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            grad_out[base_mat_idx + i * m] = __float2half(
                (float)__ldg(grad_in + base_mat_idx + i * m) * local_var +
                ((local_grad_mean + local_grad_var * (float)__ldg(mat + base_mat_idx + i * m) * 2) / (float)n)
            );
        }
    }
}
