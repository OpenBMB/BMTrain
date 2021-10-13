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

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, bias=False, rd_mean=False, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        else:
            self.bias = None
        self.rd_mean = rd_mean
        self.eps = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        normed_out = NormalizeOp.apply(hidden_states, self.eps, self.rd_mean)
        # convert into float16 if necessary
        normed_out = normed_out * self.weight
        if self.bias is not None:
            normed_out = normed_out + self.bias
        return normed_out