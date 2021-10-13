#include "ops/layernorm.h"
#include "torch_ops.h"

torch::Tensor layernorm_forward(
    const torch::Tensor& x,
    bool rd_mean,
    float eps
) {
    CHECK_INPUT(x);
    AT_ASSERTM(x.dtype() == torch::kFloat32 || x.dtype() == torch::kFloat16, "Only float32 and float16 are supported");
    int device_idx = x.device().index();
    auto curr_stream = at::cuda::getCurrentCUDAStream(device_idx);

    auto x_view = x.view({-1, x.size(-1)});
    auto ret = torch::empty_like(x);

    if (x.dtype() == torch::kFloat32) {
        layernorm_forward(x_view.size(0), x_view.size(1), x.data_ptr<float>(), ret.data_ptr<float>(), rd_mean, eps, curr_stream.stream());
    } else {
        layernorm_forward(x_view.size(0), x_view.size(1), (half *)x.data_ptr<at::Half>(), (half *)ret.data_ptr<at::Half>(), rd_mean, eps, curr_stream.stream());
    }
    return ret;
}

torch::Tensor layernorm_backward(
    const torch::Tensor& x,
    const torch::Tensor& grad_,
    bool rd_mean,
    float eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(grad_);
    AT_ASSERTM(x.dtype() == torch::kFloat32 || x.dtype() == torch::kFloat16, "Only float32 and float16 are supported");
    AT_ASSERTM(grad_.dtype() == x.dtype(), "grad must have the same type as x");
    AT_ASSERTM(x.device().index() == grad_.device().index(), "x and grad_ must be on the same device");
    int device_idx = x.device().index();
    auto curr_stream = at::cuda::getCurrentCUDAStream(device_idx);

    auto ret = torch::empty_like(x);
    auto x_view = x.view({-1, x.size(-1)});
    if (x.dtype() == torch::kFloat16) {
        layernorm_backward(x_view.size(0), x_view.size(1), (half *)x.data_ptr<at::Half>(), (half *)grad_.data_ptr<at::Half>(), (half *)ret.data_ptr<at::Half>(), rd_mean, eps, curr_stream.stream());
    } else {
        layernorm_backward(x_view.size(0), x_view.size(1), x.data_ptr<float>(), grad_.data_ptr<float>(), ret.data_ptr<float>(), rd_mean, eps, curr_stream.stream());
    }
    return ret;
}