#include <torch/extension.h>
#include <cstring>
void* create_cublaslt_handle();
torch::Tensor i8linear_forward(void* handle, torch::Tensor x, torch::Tensor A);
torch::Tensor i8linear_scale(torch::Tensor x, torch::Tensor scale_1, torch::Tensor scale_2);
torch::Tensor position_bias(int num_buckets, int max_buckets, float log_maxexact, float log_scale, torch::Tensor relative_positions);
torch::Tensor round_scale(torch::Tensor x, torch::Tensor scale);
torch::Tensor normalize_forward(torch::Tensor input, float eps, bool rd_mean);
torch::Tensor softmax(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("i8_forward", &i8linear_forward, "IGEMM forward (CUDA)");
    m.def("i8_create_handle", &create_cublaslt_handle, "create cublasLt handle");
    m.def("i8_scale_2d", &i8linear_scale, "scale in two dimensions");
    m.def("i8_round_scale", &round_scale, "scale and round to int8");
    m.def("pb_calc", &position_bias, "calc position bias bucket");
    m.def("ln_normalize_forward", &normalize_forward, "normalize last dim of input");
    m.def("softmax_forward", &softmax, "fp16 softmax for 2-dim tensor");
}
