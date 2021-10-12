#include "ops/gemm.h"
#include <cuda_runtime.h>
#include "torch_ops.h"

torch::Tensor bmm(
    const torch::Tensor &A, // b, m, k
    bool aT,
    const torch::Tensor &B, // b, k, n
    bool bT
) {
    AT_ASSERTM(A.is_cuda(), "A must be a CUDA tensor");
    AT_ASSERTM(B.is_cuda(), "B must be a CUDA tensor");
    AT_ASSERTM(A.stride(-1) == 1, "A must be contiguous");
    AT_ASSERTM(B.stride(-1) == 1, "B must be contiguous");
    AT_ASSERTM(A.device().index() == B.device().index(), "A and B must be on the same device");
    AT_ASSERTM(A.dtype() == B.dtype(), "A and B must have the same dtype");
    auto dtype = A.dtype();
    AT_ASSERTM( dtype == torch::kFloat16 || dtype == torch::kFloat32 || dtype == torch::kInt8, "bmm not support dtype");

    int32_t m, n, k;
    if (aT) {
        k = A.size(-2);
        m = A.size(-1);
    } else {
        m = A.size(-2);
        k = A.size(-1);
    }
    if (bT) {
        AT_ASSERTM(B.size(-1) == k, "B.size(-1) must equal to k");
        n = B.size(-2);
    } else {
        AT_ASSERTM(B.size(-2) == k, "B.size(-2) must equal to k");
        n = B.size(-1);
    }

    int32_t device_idx = A.device().index();

    int32_t batch_size;
    int64_t stride_A, stride_B;

    if (A.ndimension() == 2 && B.ndimension() == 2) {
        batch_size = 1;
        stride_A = 0;
        stride_B = 0;
    } 
    else if (A.ndimension() == 3 && B.ndimension() == 2) {
        batch_size = A.size(0);
        stride_A = A.stride(0);
        stride_B = 0;
    }
    else if (A.ndimension() == 2 && B.ndimension() == 3) {
        batch_size = B.size(0);
        stride_A = 0;
        stride_B = B.stride(0);
    }
    else if (A.ndimension() == 3 && B.ndimension() == 3) {
        AT_ASSERTM(A.size(0) == B.size(0), "batch size mismatch");
        batch_size = A.size(0);
        stride_A = A.stride(0);
        stride_B = B.stride(0);
    }
    else {
        throw std::runtime_error("bmm: A and B must be 2D or 3D tensors");
    }

    torch::Tensor ret;
    auto curr_stream = at::cuda::getCurrentCUDAStream(device_idx);

    cudaSetDevice(device_idx);
    if (dtype == torch::kInt8) {
        if (A.ndimension() == 2 && B.ndimension() == 2) {
            ret = torch::empty({m, n}, torch::TensorOptions().dtype(torch::kInt32).device(device_idx));
        } else {
            ret = torch::empty({batch_size, m, n}, torch::TensorOptions().dtype(torch::kInt32).device(device_idx));
        }
        auto ctx = create_gemm_context_i8(
            n, m, k,
            bT, aT,
            B.stride(-2), A.stride(-2), ret.stride(-2),
            stride_B, stride_A, ret.stride(0)
        );
        bmm_i8_kernel(ctx, batch_size, B.data<int8_t>(), B.ndimension() == 2, A.data<int8_t>(), A.ndimension() == 2, ret.data<int32_t>(), curr_stream.stream());
        release_gemm_context(ctx);
    } else {
        if (A.ndimension() == 2 && B.ndimension() == 2) {
            ret = torch::empty({m, n}, torch::TensorOptions().dtype(dtype).device(device_idx));
        } else {
            ret = torch::empty({batch_size, m, n}, torch::TensorOptions().dtype(dtype).device(device_idx));
        }
        auto ctx = create_gemm_context_fp(
            dtype == torch::kFloat16 ? CUDA_R_16F : CUDA_R_32F,
            n, m, k,
            bT, aT,
            B.stride(-2), A.stride(-2), ret.stride(-2),
            stride_B, stride_A, ret.stride(0)
        );
        if (dtype == torch::kFloat16) {
            bmm_f16_kernel(ctx, batch_size, B.data<half>(), A.data<half>(), ret.data<half>(), curr_stream.stream());
        } else {
            bmm_f32_kernel(ctx, batch_size, B.data<float>(), A.data<float>(), ret.data<float>(), curr_stream.stream());
        }
        release_gemm_context(ctx);
    }
    return ret;
}

torch::Tensor round(
    const torch::Tensor &x,
    const torch::Tensor &scale
) {
    CHECK_INPUT(x); CHECK_INPUT(scale);
    AT_ASSERTM(x.device().index() == scale.device().index(), "x and scale must be on the same device");
    AT_ASSERTM(x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32, "round not support dtype");
    AT_ASSERTM(x.dtype() == scale.dtype(), "x and scale must have the same dtype");
    AT_ASSERTM(x.ndimension() == 3, "x must be a 3-dim tensor");
    AT_ASSERTM(scale.ndimension() == 2, "scale must be a 2-dim tensor");
    AT_ASSERTM(x.size(0) == scale.size(0), "x and scale size not matching");
    AT_ASSERTM(x.size(1) == scale.size(1), "x and scale size not matching");

    int32_t device_idx = x.device().index();
    auto curr_stream = at::cuda::getCurrentCUDAStream(device_idx);

    cudaSetDevice(device_idx);
    auto ret = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8).device(device_idx));

    if (x.dtype() == torch::kFloat16) {
        round_scale_i8(x.size(0), x.size(1), x.size(2), x.data<half>(), scale.data<half>(), ret.data<int8_t>(), curr_stream.stream());
    } else {
        round_scale_i8(x.size(0), x.size(1), x.size(2), x.data<float>(), scale.data<float>(), ret.data<int8_t>(), curr_stream.stream());
    }
    return ret;
}

torch::Tensor scale(
    const torch::Tensor &x,
    const torch::Tensor &scale_1,
    const torch::Tensor &scale_2
) {
    CHECK_INPUT(x); CHECK_INPUT(scale_1); CHECK_INPUT(scale_2);
    AT_ASSERTM(x.dtype() == torch::kInt32, "x must be int32");
    AT_ASSERTM(x.device().index() == scale_1.device().index(), "x and scale_1 must be on the same device");
    AT_ASSERTM(x.device().index() == scale_2.device().index(), "x and scale_2 must be on the same device");
    AT_ASSERTM(scale_1.dtype() == scale_2.dtype(), "scale_1 and scale_2 must have the same dtype");
    AT_ASSERTM(x.ndimension() == 3, "x must be a 3-dim tensor");
    AT_ASSERTM(scale_1.ndimension() == 2, "scale_1 must be a 2-dim tensor");
    AT_ASSERTM(scale_2.ndimension() == 2, "scale_2 must be a 2-dim tensor");
    AT_ASSERTM(x.size(1) == scale_1.size(1), "x and scale_1 size not matching");
    AT_ASSERTM(x.size(2) == scale_2.size(1), "x and scale_2 size not matching");
    auto dtype = scale_1.dtype();
    AT_ASSERTM( dtype == torch::kFloat16 || dtype == torch::kFloat32, "scale not support dtype");
    
    int32_t device_idx = x.device().index();
    auto curr_stream = at::cuda::getCurrentCUDAStream(device_idx);

    auto ret = torch::empty_like(x, torch::TensorOptions().dtype(dtype).device(device_idx));

    bool broad_cast_1 = scale_1.size(0) == 1;
    bool broad_cast_2 = scale_2.size(0) == 1;
    AT_ASSERTM(broad_cast_1 || x.size(0) == scale_1.size(0), "x and scale_1 size not matching");
    AT_ASSERTM(broad_cast_2 || x.size(0) == scale_2.size(0), "x and scale_2 size not matching");

    if (dtype == torch::kFloat16) {
        scale_i32_f16(
            x.size(0), x.size(1), x.size(2), x.data<int32_t>(), scale_1.data<half>(), scale_2.data<half>(), 
            ret.data<half>(), broad_cast_1, broad_cast_2, curr_stream.stream()
        );
    } else {
        scale_i32_f32(
            x.size(0), x.size(1), x.size(2), x.data<int32_t>(), scale_1.data<float>(), scale_2.data<float>(), 
            ret.data<float>(), broad_cast_1, broad_cast_2, curr_stream.stream()
        );
    }
    return ret;
}