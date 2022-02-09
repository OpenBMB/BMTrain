#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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


// grid <1>,        thread<min(round_up(n, 32), 1024)>
__global__ void bmt_has_nan_inf(
    int32_t n,
    const half* inp,    // (n,) 
    uint8_t* out
) {
    int8_t r = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        half v = inp[i];
        if (__hisinf(v) || isnan_(v)) {
            r = 1;
            break;
        }
    }
    r = blockReduceAny(r);
    if (threadIdx.x == 0 && r > 0) {
        out[0] = 1;
    }
}
}

void has_nan_inf_launcher(
    const torch::Tensor &g_fp16,
    torch::Tensor out
) {
    int n = g_fp16.numel();
    auto g_ptr = reinterpret_cast<half*>(g_fp16.data_ptr<at::Half>());
    auto stream = at::cuda::getCurrentCUDAStream();

    int32_t threads = 1024;
    dim3 block_size = dim3(threads, 1, 1);
    dim3 grid_size = dim3(1, 1, 1);
    
    bmt_has_nan_inf<<<grid_size, block_size, 0, stream.stream()>>>(n, g_ptr, out.data_ptr<uint8_t>());
}