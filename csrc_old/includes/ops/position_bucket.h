#pragma once
#include "common/helper.h"

void position_bucket_launcher(int32_t query_len, int32_t key_len, int32_t num_buckets, int32_t max_distance, bool bidirectional, int32_t *out, cudaStream_t stream);