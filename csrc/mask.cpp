#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void mask_launcher(int32_t batch, int32_t n, int32_t m, const torch::Tensor &input, const torch::Tensor &mask, float value, torch::Tensor &output);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

void F_mask(
    int32_t batch, int32_t n, int32_t m,
    const torch::Tensor &input,         // (batch, n, m)
    const torch::Tensor &mask,          // (batch, m)
    const float value,
    torch::Tensor &output               // (batch, n, m)
) {
    CHECK_INPUT(input);
    CHECK_INPUT(mask);
    CHECK_INPUT(output);
    AT_ASSERTM(input.dtype() == torch::kHalf, "input must be a half tensor");
    AT_ASSERTM(mask.dtype() == torch::kInt8, "mask must be a bool tensor");
    AT_ASSERTM(output.dtype() == torch::kHalf, "output must be a bool tensor");
    AT_ASSERTM(input.numel() == output.numel(), "input and output must have the same number of elements");

    mask_launcher(batch, n, m, input, mask, value, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_mask", &F_mask, "mask");
}
