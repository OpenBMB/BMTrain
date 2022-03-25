#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
// block <batch_size, ceil(m/1024)>,  thread <1024>
__global__ void cu_mask(
    int32_t batch, int32_t n, int32_t m,
    const half *input,      // (batch, n, m)
    const int8_t *mask,     // (batch, m)
    float value,
    half *output            // (batch, n, m)
) {
    int32_t col_idx = threadIdx.x + blockIdx.y * blockDim.x;
    int32_t base_x_idx = blockIdx.x * n * m + col_idx;
    half half_value = __float2half(value);

    if (col_idx < m) {
        int8_t mask_val = mask[blockIdx.x * m + col_idx];
        for (int i = 0; i < n; i ++) {
            output[base_x_idx + i * m] = (mask_val == 0) ? half_value : input[base_x_idx + i * m];
        }
    }
}


}
void mask_launcher(
    int32_t batch, int32_t n, int32_t m,
    const torch::Tensor &input,
    const torch::Tensor &mask,
    float value,
    torch::Tensor &output
) {
    auto input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    auto mask_ptr = mask.data_ptr<int8_t>();
    auto output_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    auto stream = at::cuda::getCurrentCUDAStream();
    cu_mask<<<dim3(batch, (m+1023)/1024), 1024, 0, stream.stream()>>>(batch, n, m, input_ptr, mask_ptr, value, output_ptr);
}