#include <cuda.h>
#include <cstdint>
#include <assert.h>
#include "watch_dog.hpp"

extern WatchDog* watchDog;

static __global__ void spin_kernel(clock_t count) {

    clock_t start = clock();
    clock_t now = start;
    #pragma unroll(1) // avoid instruct optimization.
    while ((now - start < count) || (now + (0xffffffff - start) < count)) {
        now = clock();
    }
}

void checkCUDAStatus(cudaError_t err) {
    if (err == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA Error: ") + cudaGetErrorString(err));
}

void cuda_spin(int ms, std::uintptr_t stream) {
    int dev, clockRate = 0;
    checkCUDAStatus(cudaGetDevice(&dev));
    checkCUDAStatus(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev));
    CUDAEvent event(reinterpret_cast<cudaStream_t>(stream), "cuda_sleep");
    event.recordStart();
    spin_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(ms * clock_t(clockRate));
    checkCUDAStatus(cudaGetLastError());
    event.recordEnd();
    watchDog->watch(event);
}