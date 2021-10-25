#include "common/reduce.cuh"
#include <cuda_fp16.h>

// block <batch, n / 32, m / 32>    thread <32, 32>
__global__ void cu_transpose(
    int32_t batch, int32_t n, int32_t m,
    const half *in,
    half *out
) {
    __shared__ half shared[WARP_SZ][WARP_SZ + 1];
    int32_t row = blockIdx.y * WARP_SZ + threadIdx.y;
    int32_t col = blockIdx.z * WARP_SZ + threadIdx.x;
    int32_t offset = blockIdx.x * n * m + row * m + col;
    if (row < n && col < m) shared[threadIdx.x][threadIdx.y] = in[offset];
    __syncthreads();
    row = blockIdx.z * WARP_SZ + threadIdx.y;
    col = blockIdx.y * WARP_SZ + threadIdx.x;
    offset = blockIdx.x * n * m + row * n + col;
    if (row < m && col < n) {
        out[offset] = shared[threadIdx.y][threadIdx.x];
    }
}