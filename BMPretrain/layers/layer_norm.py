import torch
import BMPretrain._c as C

class NormalizeOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps, rd_mean):
        ctx.save_for_backward(x)
        ctx.eps = eps
        ctx.rd_mean = rd_mean
        last_dim = x.size(-1)
        return C.ln_normalize_forward(x.view(-1, last_dim), eps, rd_mean).view(x.size())

    @staticmethod
    def backward(ctx, grad_f):
        x, = ctx.saved_tensors  # (..., hidden_size)
        old_dtype = x.dtype
        x = x.to(torch.float32)
        eps = ctx.eps
        rd_mean = ctx.rd_mean
        var_x = (x**2).mean(dim=-1, keepdim=True)    # (...)

        if rd_mean:
            mean_x = x.mean(dim=-1, keepdim=True) # (...)
            var_x -= mean_x ** 2
            rsqrt_var = torch.rsqrt(var_x + eps)
            grad_var = (grad_f * (-1/2) * (x - mean_x) * (rsqrt_var ** 3)).sum(dim=-1, keepdim=True)  # (...)
            grad_mean = (- grad_f * rsqrt_var).sum(dim=-1, keepdim=True) - 2 * grad_var * mean_x    # (...)
        else:
            rsqrt_var = torch.rsqrt(var_x + eps)
            grad_var = (grad_f * (-1/2) * x * (rsqrt_var ** 3)).sum(dim=-1, keepdim=True) # (...)
            grad_mean = 0 # (...)
        grad_mean2 = grad_var   # (...)
        grad_x = grad_f * rsqrt_var + (grad_mean + grad_mean2 * x * 2) / x.size(-1)    # (..., hidden_size)
        return grad_x.to(old_dtype), None, None



        

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, bias=False, rd_mean=False, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
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