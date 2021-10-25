#include "ops/softmax.h"
#include "torch_ops.h"

torch::Tensor softmax_forward(
    const torch::Tensor& x
) {
    CHECK_INPUT(x);
    AT_ASSERTM(x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32,
               "softmax only supports float32 and float16");
    int device_idx = x.device().index();
    auto curr_stream = at::cuda::getCurrentCUDAStream(device_idx);

    torch::Tensor ret = torch::empty_like(x);
    torch::Tensor x_view = x.view({-1, x.size(-1)});
    if (x.dtype() == torch::kFloat16) {
        softmax_forward(
            x_view.size(0), x_view.size(1),
            (half *)x_view.data_ptr<at::Half>(),
            (half *)ret.data_ptr<at::Half>(),
            curr_stream.stream()
        );
    } else {
        softmax_forward(
            x_view.size(0), x_view.size(1),
            x_view.data_ptr<float>(),
            ret.data_ptr<float>(),
            curr_stream.stream()
        );
    }
    return ret;
}

torch::Tensor softmax_backward(
    const torch::Tensor& out,
    const torch::Tensor& grad_
) {
    CHECK_INPUT(out); CHECK_INPUT(grad_);
    AT_ASSERTM(out.dtype() == torch::kFloat16 || out.dtype() == torch::kFloat32,
               "softmax only supports float32 and float16");
    AT_ASSERTM(out.dtype() == grad_.dtype(), "dtype mismatch");
    AT_ASSERTM(out.device().index() == grad_.device().index(), "device mismatch");
    int device_idx = out.device().index();
    auto curr_stream = at::cuda::getCurrentCUDAStream(device_idx);

    torch::Tensor ret = torch::empty_like(out);
    torch::Tensor out_view = out.view({-1, out.size(-1)});
    if (out.dtype() == torch::kFloat16) {
        softmax_backward(
            out_view.size(0), out_view.size(1),
            (half *)out.data_ptr<at::Half>(),
            (half *)grad_.data_ptr<at::Half>(),
            (half *)ret.data_ptr<at::Half>(),
            curr_stream.stream()
        );
    } else {
        softmax_backward(
            out_view.size(0), out_view.size(1),
            out.data_ptr<float>(),
            grad_.data_ptr<float>(),
            ret.data_ptr<float>(),
            curr_stream.stream()
        );
    }
    return ret;
}