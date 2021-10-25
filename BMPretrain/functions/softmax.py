import torch
import BMPretrain._c as C


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ret = C.softmax_forward(x)
        ctx.save_for_backward(ret)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.saved_tensors[0]
        return C.softmax_backward(out, grad_output)
