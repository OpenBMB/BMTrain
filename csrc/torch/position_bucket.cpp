#include "ops/position_bucket.h"
#include "torch_ops.h"

torch::Tensor position_bucket(
    int32_t query_len, int32_t key_len, int32_t num_buckets, int32_t max_distance, bool bidirectional
) {
    int device_idx = 0;
    checkCudaStatus(cudaGetDevice(&device_idx));
    auto curr_stream = at::cuda::getCurrentCUDAStream(device_idx);

    torch::Tensor ret = torch::empty({query_len, key_len}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_idx));
    position_bucket_launcher(query_len, key_len, num_buckets, max_distance, bidirectional, ret.data_ptr<int32_t>(), curr_stream.stream());
    return ret;
}