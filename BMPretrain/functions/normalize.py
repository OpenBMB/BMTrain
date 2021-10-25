import torch
import BMPretrain._c as C

class NormalizeOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps, rd_mean):
        ctx.save_for_backward(x)
        ctx.eps = eps
        ctx.rd_mean = rd_mean
        return C.layernorm_forward(x, rd_mean, eps)

    @staticmethod
    def backward(ctx, grad_f : torch.Tensor):
        if not grad_f.is_contiguous():
            grad_f = grad_f.contiguous()
        x, = ctx.saved_tensors  # (..., hidden_size)
        eps = ctx.eps
        rd_mean = ctx.rd_mean
        return C.layernorm_backward(x, grad_f, rd_mean, eps), None, None