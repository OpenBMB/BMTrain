
from .. import C 
import torch
CHECK_INPUT = lambda x: x.is_contiguous() and x.is_cuda
def has_inf_nan(g_fp16: torch.Tensor, out: torch.Tensor) -> None:
    assert g_fp16.dtype == torch.float16, "g_fp16 must be a half tensor"
    assert out.dtype == torch.uint8, "out must be a uint8 tensor"
    assert CHECK_INPUT(g_fp16), "g_fp16 must be contiguous and on cuda"
    assert CHECK_INPUT(out), "out must be contiguous and on cuda"
    mid = torch.zeros(1024, device=out.device, dtype=out.dtype)
    stream = torch.cuda.current_stream().cuda_stream
    C.has_nan_inf_launcher(g_fp16.numel(), g_fp16.data_ptr(), mid.data_ptr(), out.data_ptr(), stream)



def cross_entropy_forward(m: int, n: int, input: torch.Tensor, target: torch.Tensor,
                            softmax: torch.Tensor, output: torch.Tensor, ignore_index: int) -> None:
    CHECK_INPUT(input)
    CHECK_INPUT(target)
    CHECK_INPUT(softmax)
    CHECK_INPUT(output)
    assert input.dtype == torch.float16, "input must be a half tensor"
    assert target.dtype == torch.int32, "target must be an int tensor"
    assert softmax.dtype == torch.float16, "softmax must be a half tensor"
    assert output.dtype == torch.float32, "output must be a float tensor"
    assert input.numel() == softmax.numel(), "input and softmax must have the same number of elements"
    assert target.numel() == output.numel(), "target and output must have the same number of elements"
    input_ptr = input.data_ptr()
    target_ptr = target.data_ptr()
    softmax_ptr = softmax.data_ptr()
    output_ptr = output.data_ptr()
    cuda_stream = torch.cuda.current_stream().cuda_stream
    C.cross_entropy_forward_launcher(m, n, input_ptr, target_ptr, softmax_ptr, output_ptr, ignore_index, cuda_stream)

def cross_entropy_backward(m: int, n: int, grad_output: torch.Tensor, target: torch.Tensor,
                             softmax: torch.Tensor, grad_input: torch.Tensor, ignore_index: int) -> None:
    CHECK_INPUT(grad_output)
    CHECK_INPUT(target)
    CHECK_INPUT(softmax)
    CHECK_INPUT(grad_input)
    assert grad_output.dtype == torch.float32, "grad_output must be a float tensor"
    assert target.dtype == torch.int32, "target must be an int tensor"
    assert softmax.dtype == torch.float16, "softmax must be a half tensor"
    assert grad_input.dtype == torch.float16, "grad_input must be a half tensor"
    assert grad_input.numel() == softmax.numel(), "grad_input and softmax must have the same number of elements"
    assert target.numel() == grad_output.numel(), "target and grad_output must have the same number of elements"
    grad_output_ptr = grad_output.data_ptr()
    target_ptr = target.data_ptr()
    softmax_ptr = softmax.data_ptr()
    grad_input_ptr = grad_input.data_ptr()
    cuda_stream = torch.cuda.current_stream().cuda_stream
    C.cross_entropy_backward_launcher(m, n, grad_output_ptr, target_ptr, softmax_ptr, grad_input_ptr, ignore_index, cuda_stream)

def cross_entropy_forward_inplace(m: int, n: int, x: torch.Tensor, target: torch.Tensor,
                                    output: torch.Tensor, ignore_index: int) -> None:
    CHECK_INPUT(x)
    CHECK_INPUT(target)
    CHECK_INPUT(output)
    assert x.dtype == torch.float16, "x must be a half tensor"
    assert target.dtype == torch.int32, "target must be an int tensor"
    assert output.dtype == torch.float32, "output must be a float tensor"
    assert target.numel() == output.numel(), "target and output must have the same number of elements"
    cuda_stream = torch.cuda.current_stream().cuda_stream
    x_ptr = x.data_ptr()
    output_ptr = output.data_ptr()
    target_ptr = target.data_ptr()
    output_ptr = output.data_ptr()
    
    C.cross_entropy_forward_inplace_launcher(m, n, x_ptr, target_ptr, output_ptr, ignore_index, cuda_stream)

def cross_entropy_backward_inplace(m: int, n: int, grad_output: torch.Tensor, target: torch.Tensor,
                                     x: torch.Tensor, ignore_index: int) -> None:
    CHECK_INPUT(grad_output)
    CHECK_INPUT(target)
    CHECK_INPUT(x)
    assert grad_output.dtype == torch.float32, "grad_output must be a float tensor"
    assert target.dtype == torch.int32, "target must be an int tensor"
    assert x.dtype == torch.float16, "x must be a half tensor"
    assert target.numel() == grad_output.numel(), "target and grad_output must have the same number of elements"
    cuda_stream = torch.cuda.current_stream().cuda_stream
    grad_output_ptr = grad_output.data_ptr()
    target_ptr = target.data_ptr()
    x_ptr = x.data_ptr()

    C.cross_entropy_backward_inplace_launcher(m, n, grad_output_ptr, target_ptr, x_ptr, ignore_index, cuda_stream)

