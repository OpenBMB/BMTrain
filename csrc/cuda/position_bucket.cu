#include "common/helper.h"

namespace {

template<bool bidirectional>
void position_bucket_kernel(
    int32_t query_len,
    int32_t key_len,
    int32_t num_buckets,
    int32_t max_distance,
    int32_t *out
) {
    int32_t max_exact = num_buckets / (bidirectional ? 4 : 2);

    for (int i = threadIdx.x; i < key_len; i += blockDim.x) {
        int32_t relative_position = blockIdx.x - i;
        int32_t bucket_offset = 0;
        if (relative_position < 0) {
            if (bidirectional) {
                bucket_offset = num_buckets / 2;
                relative_position = -relative_position;
            }
        } else {
            relative_position = 0;
        }
        if (relative_position < max_exact) {
            out[i + blockIdx.x * key_len] = relative_position + bucket_offset;
        } else {
            bucket_offset += max_exact;
            out[i + blockIdx.x * key_len] = logf(relative_position / max_exact) / logf(max_distance / max_exact) * max_exact + bucket_offset;
        }
    }
}

}

void position_bucket_launcher(
    int32_t query_len,
    int32_t key_len,
    int32_t num_buckets,
    int32_t max_distance,
    bool bidirectional,
    int32_t *out,
    cudaStream_t stream
) {
    assert(num_buckets % 4 == 0);
    dim3 blocks(query_len);
    dim3 threads(::min(key_len, 1024));

    if (bidirectional) {
        ::position_bucket_kernel<true><<<blocks, threads, 0, stream>>>(
            query_len,
            key_len,
            num_buckets,
            max_distance,
            out
        );
    } else {
        ::position_bucket_kernel<false><<<blocks, threads, 0, stream>>>(
            query_len,
            key_len,
            num_buckets,
            max_distance,
            out
        );
    }
}