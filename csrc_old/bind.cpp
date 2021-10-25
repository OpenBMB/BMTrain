#include <cstring>
#include "torch_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm", &bmm, "batch matmul");
    m.def("round", [](const torch::Tensor &x, bool transpose, const torch::Tensor &scale) -> torch::Tensor{
        return round(x, transpose, scale);
    }, "round to int8");
    m.def("calc_scale", &calc_scale, "calc scale");
    m.def("scale", &scale, "scale tensor from int32");
    m.def("layernorm_forward", &layernorm_forward, "layernorm forward");
    m.def("layernorm_backward", &layernorm_backward, "layernorm backward");
    m.def("position_bucket", &position_bucket, "position bucket");
    m.def("softmax_forward", &softmax_forward, "softmax forward");
    m.def("softmax_backward", &softmax_backward, "softmax backward");
}
