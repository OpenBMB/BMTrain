#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
// blocks <n // 1024>,      threads<min(n, 1024)>
__global__ void adam_fp32_accum(
    int32_t n,
    const half *g,        // (n)
    half *m,        // (n)
    float *v,        // (n)
    float* param,   // (n)
    half* param_h,  // (n)
    float beta1,
    float beta2,
    float eps,
    float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float local_g = __half2float(g[col]);                                       // real_g * scale
        float local_m = beta1 * __half2float(m[col]) + (1 - beta1) * local_g;       // real_m * scale
        float local_v = beta2 * v[col] + (1 - beta2) * local_g * local_g / scale;   // real_v * scale
        float local_p = param[col];
        local_p = local_p - lr * local_m / bias_correction1 / (sqrtf(local_v * scale / bias_correction2) + eps * scale) - lr * weight_decay * local_p;

        param_h[col] = __float2half(local_p);
        param[col] = local_p;
        v[col] = local_v;
        m[col] = __float2half(local_m);
    }
}

__global__ void adam_fp32_accum_bf16(
    int32_t n,
    const nv_bfloat16 *g,        // (n)
    nv_bfloat16 *m,        // (n)
    float *v,        // (n)
    float* param,   // (n)
    nv_bfloat16* param_h,  // (n)
    float beta1,
    float beta2,
    float eps,
    float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n) {
        float local_g = __bfloat162float(g[col]) / scale; // real_g
        float local_m = beta1 * __bfloat162float(m[col]) + (1 - beta1) * local_g; // real_m
        float local_v = beta2 * v[col] + (1 - beta2) * local_g * local_g; // real_v
        float local_p = param[col];
        local_p = local_p - lr * local_m / bias_correction1 / (sqrtf(local_v / bias_correction2 / scale) + eps) - lr * weight_decay * local_p; 
    
        param_h[col] = __float2bfloat16(local_p);
        param[col] = local_p;
        v[col] = local_v;
        m[col] = __float2bfloat16(local_m); 
    }
       
}

}

void adam_launcher(
    int n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_fp16,
    std::uintptr_t g_fp16,
    std::uintptr_t m_fp16,
    std::uintptr_t v_fp32,
    float beta1, float beta2,
    float eps, float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    uintptr_t stream
) {
    if (n <= 0) return;
    auto g_ptr = reinterpret_cast<half*>(g_fp16);
    auto m_ptr = reinterpret_cast<half*>(m_fp16);
    auto param_h_ptr = reinterpret_cast<half*>(param_fp16);
    auto param_fp32_ptr = reinterpret_cast<float*>(param_fp32);
    auto v_fp32_ptr = reinterpret_cast<float*>(v_fp32);
    int32_t threads = 1024;
    dim3 block_size = dim3(threads, 1, 1);
    dim3 grid_size = dim3((n + threads - 1) / threads, 1, 1);
    adam_fp32_accum<<<grid_size, block_size, 0, reinterpret_cast<cudaStream_t>(stream)>>>(n, g_ptr, m_ptr, v_fp32_ptr, param_fp32_ptr, param_h_ptr, beta1, beta2, eps, lr, scale, weight_decay, bias_correction1, bias_correction2);
}

void adam_bf16_launcher(
    int n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_bf16,
    std::uintptr_t g_bf16,
    std::uintptr_t m_fp32,
    std::uintptr_t v_fp32,
    float beta1, float beta2,
    float eps, float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    uintptr_t stream
) {
    if (n <= 0) return;
    auto g_ptr = reinterpret_cast<nv_bfloat16*>(g_bf16);
    auto m_ptr = reinterpret_cast<nv_bfloat16*>(m_fp32);
    auto param_h_ptr = reinterpret_cast<nv_bfloat16*>(param_bf16);
    auto param_fp32_ptr = reinterpret_cast<float*>(param_fp32);
    auto v_fp32_ptr = reinterpret_cast<float*>(v_fp32);
    int32_t threads = 1024;
    dim3 block_size = dim3(threads, 1, 1);
    dim3 grid_size = dim3((n + threads - 1) / threads, 1, 1);
    adam_fp32_accum_bf16<<<grid_size, block_size, 0, reinterpret_cast<cudaStream_t>(stream)>>>(n, g_ptr, m_ptr, v_fp32_ptr, param_fp32_ptr, param_h_ptr, beta1, beta2, eps, lr, scale, weight_decay, bias_correction1, bias_correction2);
}
