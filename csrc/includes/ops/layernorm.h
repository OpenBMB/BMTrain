#pragma once
#include "common/helper.h"

void layernorm_forward(int32_t batch, int32_t n, const half *mat, half *out, bool rd_mean, float eps, cudaStream_t stream);
void layernorm_forward(int32_t batch, int32_t n, const float *mat, float *out, bool rd_mean, float eps, cudaStream_t stream);

void layernorm_backward(int32_t batch, int32_t n, const half *x, const half *grad_in, half *grad_out, bool rd_mean, float eps, cudaStream_t stream);
void layernorm_backward(int32_t batch, int32_t n, const float *x, const float *grad_in, float *grad_out, bool rd_mean, float eps, cudaStream_t stream);

