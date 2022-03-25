#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void fused_softmax_forward_launcher(int32_t m, int32_t n, const torch::Tensor &input, torch::Tensor &softmax);
void fused_softmax_backward_launcher(int32_t m, int32_t n, const torch::Tensor &grad_output, const torch::Tensor &softmax, torch::Tensor &grad_input);
void fused_softmax_forward_inplace_launcher(int32_t m, int32_t n, torch::Tensor &x);
void fused_softmax_backward_inplace_launcher(int32_t m, int32_t n, const torch::Tensor &grad_output, torch::Tensor &x);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

void F_fused_softmax_forward(
    int32_t m, int32_t n,
    const torch::Tensor &input,     // (m, n)
    torch::Tensor &softmax          // (m, n)
) {
    CHECK_INPUT(input);
    CHECK_INPUT(softmax);
    AT_ASSERTM(input.dtype() == torch::kHalf, "input must be a half tensor");
    AT_ASSERTM(softmax.dtype() == torch::kHalf, "softmax must be a half tensor");
    AT_ASSERTM(input.numel() == softmax.numel(), "input and softmax must have the same number of elements");

    fused_softmax_forward_launcher(m, n, input, softmax);
}

void F_fused_softmax_backward(
    int32_t m, int32_t n,
    const torch::Tensor &grad_output,   // (m, n)
    const torch::Tensor &softmax,       // (m, n)
    torch::Tensor &grad_input           // (m, n)
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(softmax);
    CHECK_INPUT(grad_input);
    AT_ASSERTM(grad_output.dtype() == torch::kHalf, "grad_output must be a half tensor");
    AT_ASSERTM(softmax.dtype() == torch::kHalf, "softmax must be a half tensor");
    AT_ASSERTM(grad_input.dtype() == torch::kHalf, "grad_input must be a half tensor");
    AT_ASSERTM(grad_input.numel() == grad_output.numel(), "grad_input and grad_output must have the same number of elements");
    AT_ASSERTM(softmax.numel() == grad_output.numel(), "softmax and grad_output must have the same number of elements");

    fused_softmax_backward_launcher(m, n, grad_output, softmax, grad_input);
}

void F_fused_softmax_forward_inplace(
    int32_t m, int32_t n,
    torch::Tensor &x                    // (m, n)
) {
    CHECK_INPUT(x);
    AT_ASSERTM(x.dtype() == torch::kHalf, "x must be a half tensor");

    fused_softmax_forward_inplace_launcher(m, n, x);
}

void F_fused_softmax_backward_inplace(
    int32_t m, int32_t n,
    const torch::Tensor &grad_output,   // (m, n)
    torch::Tensor &x                    // (m, n)
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    AT_ASSERTM(grad_output.dtype() == torch::kHalf, "grad_output must be a half tensor");
    AT_ASSERTM(x.dtype() == torch::kHalf, "x must be a half tensor");
    AT_ASSERTM(x.numel() == grad_output.numel(), "x and grad_output must have the same number of elements");

    fused_softmax_backward_inplace_launcher(m, n, grad_output, x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_fused_softmax_forward", &F_fused_softmax_forward, "fused softmax forward");
    m.def("f_fused_softmax_backward", &F_fused_softmax_backward, "fused softmax backward");
    m.def("f_fused_softmax_forward_inplace", &F_fused_softmax_forward_inplace, "fused softmax forward inplace");
    m.def("f_fused_softmax_backward_inplace", &F_fused_softmax_backward_inplace, "fused softmax backward inplace");
}
