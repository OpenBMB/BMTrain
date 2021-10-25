#pragma once

namespace {

__inline__ __device__ float warpReduceSum(float x) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        x += __shfl_down_sync(0xFFFFFFFF, x, offset);
    return x;
}

__inline__ __device__ float blockReduceSum(float x) {
    static __shared__ float shared[32]; // blockDim.x / warpSize
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    x = warpReduceSum(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) x = warpReduceSum(x);
    return x;
}

__inline__ __device__ float warpReduceMax(float x) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        x = fmaxf(x, __shfl_down_sync(0xFFFFFFFF, x, offset));
    return x;
}

__inline__ __device__ float blockReduceMax(float x) {
    static __shared__ float shared[32]; // blockDim.x / warpSize
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    x = warpReduceMax(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -INFINITY;
    if (wid == 0) x = warpReduceMax(x);
    return x;
}
}