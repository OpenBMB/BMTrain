
from .. import C
import torch
CHECK_INPUT = lambda x: x.is_contiguous() and x.is_cuda
def adam_cpu(param_fp32: torch.Tensor, param_fp16: torch.Tensor, g_fp16: torch.Tensor, m_fp32: torch.Tensor,
             v_fp32: torch.Tensor, beta1: float, beta2: float, eps: float, lr: float, scale: float,
             weight_decay: float, step: int) -> None:
    assert param_fp32.is_contiguous(), "param_fp32 must be contiguous"
    assert param_fp16.is_contiguous(), "param_fp16 must be contiguous"
    assert g_fp16.is_contiguous(), "g_fp16 must be contiguous"
    assert m_fp32.is_contiguous(), "m_fp32 must be contiguous"
    assert v_fp32.is_contiguous(), "v_fp32 must be contiguous"
    assert param_fp32.dtype == torch.float32, "param_fp32 must be float32 tensor"
    assert param_fp16.dtype == torch.float16, "param_fp16 must be float16 tensor"
    assert g_fp16.dtype == torch.float16, "g_fp16 must be float16 tensor"
    assert m_fp32.dtype == torch.float32, "m_fp32 must be float32 tensor"
    assert v_fp32.dtype == torch.float32, "v_fp32 must be float32 tensor"
    assert param_fp32.device == torch.device("cpu"), "param_fp32 must be a cpu tensor"
    assert param_fp16.device == torch.device("cpu"), "param_fp16 must be a cpu tensor"
    assert g_fp16.device == torch.device("cpu"), "g_fp16 must be a cpu tensor"
    assert m_fp32.device == torch.device("cpu"), "m_fp32 must be a cpu tensor"
    assert v_fp32.device == torch.device("cpu"), "v_fp32 must be a cpu tensor"
    assert param_fp32.numel() == param_fp16.numel(), "param_fp32 and param_fp16 must have the same number of elements"
    assert param_fp32.numel() == g_fp16.numel(), "param_fp32 and g_fp16 must have the same number of elements"
    assert param_fp32.numel() == m_fp32.numel(), "param_fp32 and m_fp32 must have the same number of elements"
    assert param_fp32.numel() == v_fp32.numel(), "param_fp32 and v_fp32 must have the same number of elements"
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    C.adam_cpu_launcher(
                    param_fp32.numel(),
                    param_fp32.data_ptr(),
                    param_fp16.data_ptr(),
                    g_fp16.data_ptr(),
                    m_fp32.data_ptr(),
                    v_fp32.data_ptr(),
                    beta1, beta2,
                    eps, lr,
                    scale,
                    weight_decay,
                    bias_correction1,
                    bias_correction2,
                )

def adam(param_fp32: torch.Tensor, param_fp16: torch.Tensor, g_fp16: torch.Tensor, m_fp16: torch.Tensor,
             v_fp32: torch.Tensor, beta1: float, beta2: float, eps: float, lr: float, scale: float,
             weight_decay: float, step: int) -> None:
    assert CHECK_INPUT(param_fp32), "param_fp32 must be contiguous and on cuda"
    assert CHECK_INPUT(param_fp16), "param_fp16 must be contiguous and on cuda"
    assert CHECK_INPUT(g_fp16), "g_fp16 must be contiguous and on cuda"
    assert CHECK_INPUT(m_fp16), "m_fp32 must be contiguous and on cuda"
    assert CHECK_INPUT(v_fp32), "v_fp32 must be contiguous and on cuda"
    assert param_fp32.dtype == torch.float32, "param_fp32 must be float32 tensor"
    assert param_fp16.dtype == torch.float16, "param_fp16 must be float16 tensor"
    assert g_fp16.dtype == torch.float16, "g_fp16 must be float16 tensor"
    assert m_fp16.dtype == torch.float16, "m_fp16 must be float16 tensor"
    assert v_fp32.dtype == torch.float32, "v_fp32 must be float32 tensor"
    assert param_fp32.numel() == param_fp16.numel(), "param_fp32 and param_fp16 must have the same number of elements"
    assert param_fp32.numel() == g_fp16.numel(), "param_fp32 and g_fp16 must have the same number of elements"
    assert param_fp32.numel() == m_fp16.numel(), "param_fp32 and m_fp32 must have the same number of elements"
    assert param_fp32.numel() == v_fp32.numel(), "param_fp32 and v_fp32 must have the same number of elements"
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    stream = torch.cuda.current_stream().cuda_stream
    C.adam_launcher(
        param_fp32.numel(),
        param_fp32.data_ptr(),
        param_fp16.data_ptr(),
        g_fp16.data_ptr(),
        m_fp16.data_ptr(),
        v_fp32.data_ptr(),
        beta1, beta2,
        eps, lr,
        scale,
        weight_decay,
        bias_correction1,
        bias_correction2,
        stream
    )
