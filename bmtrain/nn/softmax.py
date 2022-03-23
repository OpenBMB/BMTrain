from typing import Optional
import torch
from . import softmax_cuda as C

class OpFusedSoftmax(torch.autograd.Function):
    """
    Softmax dim = -1
    """
    @staticmethod
    def forward(ctx, x : torch.Tensor):
        shape = x.shape
        x = x.view(-1, shape[-1])
        softmax = torch.empty(x.size(), device=x.device, dtype=x.dtype)
        C.f_fused_softmax_forward(
            x.size(0), x.size(1),
            x,
            softmax,
        )
        softmax = softmax.view(shape)
        ctx.save_for_backward(softmax)
        return softmax
        
    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        softmax, = ctx.saved_tensors
        shape = softmax.shape
        softmax = softmax.view(-1, shape[-1])
        C.f_fused_softmax_backward_inplace(
            softmax.size(0), softmax.size(1),
            grad_output,
            softmax,
        )
        return (softmax.view(shape),)

class FusedSoftmax(torch.nn.Module):
    """Softmax on last dimension
    TODO

    Examples::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.softmax = bmt.nn.FusedSoftmax()

            def forward(self, input):
                return self.softmax(input)
    """
    def __init__(self,
                ) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = OpFusedSoftmax.apply(input)
        return output