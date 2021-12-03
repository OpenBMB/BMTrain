#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

void adam_cpu_launcher(
    int n,
    torch::TensorAccessor<float, 1> param_fp32,
    torch::TensorAccessor<at::Half, 1> param_fp16,
    torch::TensorAccessor<at::Half, 1> g_fp16,
    torch::TensorAccessor<float, 1> m_fp32,
    torch::TensorAccessor<float, 1> v_fp32,
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            float g = c10::detail::fp16_ieee_to_fp32_value(g_fp16[i].x) / scale;
            float m = m_fp32[i];
            float v = v_fp32[i];
            float p = param_fp32[i];
            m = beta1 * m + (1 - beta1) * g;
            v = beta2 * v + (1 - beta2) * g * g;
            p = p - lr * m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) - lr * weight_decay * p;
            
            param_fp32[i] = p;
            param_fp16[i] = at::Half(p);
            m_fp32[i] = m;
            v_fp32[i] = v;
        }
    });
}

void F_adam_cpu(
    const torch::Tensor &param_fp32, 
    const torch::Tensor &param_fp16, 
    const torch::Tensor &g_fp16, 
    const torch::Tensor &m_fp32, 
    const torch::Tensor &v_fp32, 
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    int64_t step
) {
    CHECK_CONTIGUOUS(param_fp32);
    CHECK_CONTIGUOUS(param_fp16);
    CHECK_CONTIGUOUS(g_fp16);
    CHECK_CONTIGUOUS(m_fp32);
    CHECK_CONTIGUOUS(v_fp32);
    AT_ASSERTM(param_fp32.dtype() == torch::kFloat, "param_fp32 must be a float tensor");
    AT_ASSERTM(param_fp16.dtype() == torch::kHalf, "param_fp16 must be a half tensor");
    AT_ASSERTM(g_fp16.dtype() == torch::kHalf, "g_fp16 must be a half tensor");
    AT_ASSERTM(m_fp32.dtype() == torch::kFloat, "m_fp32 must be a float tensor");
    AT_ASSERTM(v_fp32.dtype() == torch::kFloat, "v_fp32 must be a float tensor");
    AT_ASSERTM(param_fp32.is_cpu(), "param_fp32 must be a cpu tensor");
    AT_ASSERTM(param_fp16.is_cpu(), "param_fp16 must be a cpu tensor");
    AT_ASSERTM(g_fp16.is_cpu(), "g_fp16 must be a cpu tensor");
    AT_ASSERTM(m_fp32.is_cpu(), "m_fp32 must be a cpu tensor");
    AT_ASSERTM(v_fp32.is_cpu(), "v_fp32 must be a cpu tensor");
    AT_ASSERTM(param_fp32.numel() == param_fp16.numel(), "param_fp32 and param_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == g_fp16.numel(), "param_fp32 and g_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == m_fp32.numel(), "param_fp32 and m_fp32 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == v_fp32.numel(), "param_fp32 and v_fp32 must have the same number of elements");

    float bias_correction1 = 1 - powf(beta1, step);
    float bias_correction2 = 1 - powf(beta2, step);

    adam_cpu_launcher(
        param_fp32.numel(),
        param_fp32.accessor<float, 1>(),
        param_fp16.accessor<at::Half, 1>(),
        g_fp16.accessor<at::Half, 1>(),
        m_fp32.accessor<float, 1>(),
        v_fp32.accessor<float, 1>(),
        beta1, beta2, 
        eps, lr, 
        scale, 
        weight_decay,
        bias_correction1,
        bias_correction2
    );
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_adam_cpu", &F_adam_cpu, "adam function cpu");
}
