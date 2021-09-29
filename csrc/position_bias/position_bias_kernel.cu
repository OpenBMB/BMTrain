#include "helper.h"
#include <torch/extension.h>
#include <device_launch_parameters.h>

namespace {


template<typename T>
__global__ void calc_bias_bucket_kernel(
    float log_max,
    float log_scale,
    int num_buckets,
    int max_buckets,
    const torch::PackedTensorAccessor32<T, 1, torch::RestrictPtrTraits> relative_position,
    torch::PackedTensorAccessor32<T, 1, torch::RestrictPtrTraits> out
) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < out.size(0)) {
        T t = (T)((logf(relative_position[index]) - log_max) * log_scale * num_buckets);
        if (t < max_buckets) out[index] = t;
        else out[index] = max_buckets - 1;
    }
}

}

void calc_bias_bucket(
    float log_max,
    float log_scale,
    int num_buckets,
    int max_buckets,
    torch::Tensor x,
    torch::Tensor out
) {
    int32_t thread_size = 1024;
    const dim3 threads(thread_size);
    const dim3 blocks((x.size(0) + thread_size - 1) / thread_size);
    AT_DISPATCH_INTEGRAL_TYPES(out.type(), "calc_bias_bucket_cuda", ([&] {
        calc_bias_bucket_kernel<scalar_t><<<blocks, threads>>>(
            log_max,
            log_scale,
            num_buckets,
            max_buckets,
            x.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            out.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));
}
