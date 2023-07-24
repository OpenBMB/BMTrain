#include <cuda_fp16.h>
#include <cstdint>

namespace{
__inline__ __device__ bool isnan_(half v) {
#if __CUDA_ARCH__ >= 700 || __CUDA_ARCH__ == 600
    return __hisnan(v);
#else
    
    return !__heq(v, v);
#endif
}

__inline__ __device__ int8_t warpReduceAny(int8_t x) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        x |= __shfl_down_sync(0xFFFFFFFF, x, offset);
    return x;
}

__inline__ __device__ float blockReduceAny(int8_t x) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    x = warpReduceAny(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) x = warpReduceAny(x);
    return x;
}

// grid <min(ceil(n/1024), 1024)>,        thread<1024>
__global__ void bmt_has_nan_inf_1(
    int32_t n,
    const half* inp,        // (n,) 
    uint8_t* mid            // (1024,)
) {
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t span = blockDim.x * gridDim.x;

    int8_t r = 0;
    for (int i = gid; i < n; i += span) {
        half v = inp[i];
        if (__hisinf(v) || isnan_(v)) {
            r = 1;
            break;
        }
    }
    r = blockReduceAny(r);
    if (threadIdx.x == 0) {
        mid[blockIdx.x] = r;
    }
}

// grid <1>,        thread<1024>
__global__ void bmt_has_nan_inf_2(
    const uint8_t* mid,    // (1024,) 
    uint8_t* out
) {
    int tid = threadIdx.x;
    int8_t r = blockReduceAny(mid[tid]);
    if (tid == 0 && r > 0) {
        out[0] = 1;
    }
}

}

void has_nan_inf_launcher(
    int32_t n,
    std::uintptr_t g_fp16,
    std::uintptr_t mid,
    std::uintptr_t out,
    std::uintptr_t stream
) {
    if (n <= 0) return;
    auto g_ptr = reinterpret_cast<half*>(g_fp16);
    auto mid_ptr = reinterpret_cast<uint8_t*>(mid);
    auto out_ptr = reinterpret_cast<uint8_t*>(out);
    int32_t threads = 1024;
    dim3 block_size = dim3(threads, 1, 1);
    dim3 grid_size = dim3((n + threads - 1) / threads, 1, 1);
    dim3 clamp_grid_size = dim3(min((n + threads - 1) / threads, 1024), 1, 1);
    
    bmt_has_nan_inf_1<<<clamp_grid_size, block_size, 0, reinterpret_cast<cudaStream_t>(stream)>>>(n, g_ptr, mid_ptr);
    bmt_has_nan_inf_2<<<1, block_size, 0, reinterpret_cast<cudaStream_t>(stream)>>>(mid_ptr, out_ptr);
}