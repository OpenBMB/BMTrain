import torch
from . import mask_cuda as C

class OpMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x : torch.Tensor, mask : torch.Tensor, value : float) -> torch.Tensor:
        assert x.dim() == 3
        batch, n, m = x.size()
        assert mask.size() == (batch, m)

        out = torch.empty(x.size(), dtype=torch.float16, device=x.device)
        mask = mask.to(torch.int8)
        C.f_mask(
            batch, n, m,
            x,
            mask,
            value,
            out,
        )
        ctx.save_for_backward(mask)
        return out

    @staticmethod
    def backward(ctx, grad_output : torch.Tensor) -> torch.Tensor:
        mask = ctx.saved_tensors[0]
        batch, n, m = grad_output.size()
        
        grad_input = torch.empty(grad_output.size(), dtype=torch.float16, device=grad_output.device)
        C.f_mask(
            batch, n, m,
            grad_output,
            mask,
            0.0,
            grad_input,
        )
        return grad_input, None, None

class Mask(torch.nn.Module):
    """Mask false position to given value and keep those positive position
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

    def forward(self, input: torch.Tensor, mask: torch.Tensor, value : float) -> torch.Tensor:
        return OpMask.apply(input, mask, value)