import torch
from ..global_var import config
from ..nccl import allGather as ncclAllGather
from ..nccl import allReduce as ncclAllReduce

class OpAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input : torch.Tensor):
        if not input.is_contiguous():
            input = input.contiguous()
        if input.storage_offset() != 0 or input.storage().size() != input.numel():
            input = input.clone()
        output = torch.empty( (config['world_size'],) + input.size(), dtype=input.dtype, device=input.device)
        
        ncclAllGather(
            input.storage(),
            output.storage(),
            config['comm']
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output[config['rank']]

def all_gather(x : torch.Tensor):
    """Gathers the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (...).
    
    Returns:
        torch.Tensor: The gathered tensor of shape (world_size, ...).
    """
    assert x.is_cuda
    return OpAllGather.apply(x)

class OpAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input : torch.Tensor, op : str):
        if not input.contiguous():
            input = input.contiguous()
        if input.storage_offset() != 0 or input.storage().size() != input.numel():
            input = input.clone()
        output = torch.empty( input.size(), dtype=input.dtype, device=input.device)
        
        ncclAllReduce(
            input.storage(),
            output.storage(),
            op,
            config['comm']
        )
        ctx.op = op

        if op in ["sum", "avg"]:
            pass
        elif op in ["max", "min"]:
            ctx.save_for_backward( input != output )
        else:
            ctx.save_for_backward( output / input )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.op == "sum":
            return grad_output, None
        elif ctx.op == "avg":
            return grad_output / config['world_size'], None
        elif ctx.op in ["max", "min"]:
            return torch.masked_fill(grad_output, ctx.saved_tensors[0], 0), None
        else:
            return grad_output * ctx.saved_tensors[0], None

def all_reduce(x : torch.Tensor, op : str = "sum"):
    """Reduces the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (...).
        op (str): The reduction operation, one of "sum", "avg", "max", "min", "prod". Default: "sum".

    Returns:
        torch.Tensor: The reduced tensor of shape (...).
    
    """
    assert x.is_cuda
    return OpAllReduce.apply(x, op)


            
