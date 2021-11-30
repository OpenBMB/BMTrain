#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void adam_launcher(const torch::Tensor &param_fp32, const torch::Tensor &param_fp16, const torch::Tensor &g_fp16, const torch::Tensor &m_fp16, const torch::Tensor &v_fp32, float beta1, float beta2, float eps, float lr, float scale, float weight_decay);


#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void F_adam(const torch::Tensor &param_fp32, const torch::Tensor &param_fp16, const torch::Tensor &g_fp16, const torch::Tensor &m_fp16, const torch::Tensor &v_fp32, float beta1, float beta2, float eps, float lr, float scale, float weight_decay) {
    CHECK_INPUT(param_fp32);
    CHECK_INPUT(param_fp16);
    CHECK_INPUT(g_fp16);
    CHECK_INPUT(m_fp16);
    CHECK_INPUT(v_fp32);
    AT_ASSERTM(param_fp32.dtype() == torch::kFloat, "param_fp32 must be a float tensor");
    AT_ASSERTM(param_fp16.dtype() == torch::kHalf, "param_fp16 must be a half tensor");
    AT_ASSERTM(g_fp16.dtype() == torch::kHalf, "g_fp16 must be a half tensor");
    AT_ASSERTM(m_fp16.dtype() == torch::kHalf, "m_fp16 must be a half tensor");
    AT_ASSERTM(v_fp32.dtype() == torch::kFloat, "v_fp32 must be a float tensor");
    AT_ASSERTM(param_fp32.numel() == param_fp16.numel(), "param_fp32 and param_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == g_fp16.numel(), "param_fp32 and g_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == m_fp16.numel(), "param_fp32 and m_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == v_fp32.numel(), "param_fp32 and v_fp32 must have the same number of elements");
    adam_launcher(param_fp32, param_fp16, g_fp16, m_fp16, v_fp32, beta1, beta2, eps, lr, scale, weight_decay);
}

void adam(
    torch::TensorList &param_fp32,
    torch::TensorList &param_fp16,
    torch::TensorList &g_fp16,
    torch::TensorList &m_fp16,
    torch::TensorList &v_fp32,
    float beta1,
    float beta2,
    float eps,
    float lr,
    float scale,
    float weight_decay
) {
    AT_ASSERTM(param_fp32.size() == param_fp16.size(), "param_fp32 and param_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.size() == g_fp16.size(), "param_fp32 and g_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.size() == m_fp16.size(), "param_fp32 and m_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.size() == v_fp32.size(), "param_fp32 and v_fp32 must have the same number of elements");
    for (size_t i = 0; i < param_fp32.size(); i++) {
        F_adam(param_fp32[i], param_fp16[i], g_fp16[i], m_fp16[i], v_fp32[i], beta1, beta2, eps, lr, scale, weight_decay);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam", &adam, "adam function");
    m.def("f_adam", &F_adam, "adam function");
}
