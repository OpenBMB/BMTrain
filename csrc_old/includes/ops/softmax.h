#pragma once
#include "common/helper.h"

void softmax_forward(int32_t batch, int32_t n, const half *in, half *out, cudaStream_t stream);
void softmax_forward(int32_t batch, int32_t n, const float *in, float *out, cudaStream_t stream);
void softmax_backward(int32_t batch, int32_t n, const half *out, const half *grad_in, half *grad_out, cudaStream_t stream);
void softmax_backward(int32_t batch, int32_t n, const float *out, const float *grad_in, float *grad_out, cudaStream_t stream);