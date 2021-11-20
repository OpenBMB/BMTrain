import torch
from typing import Callable, TypeVar
from typing_extensions import ParamSpec
from functools import wraps

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, preserve_rng_state, *args):
        ctx.func = func
        ctx.preserve_rng_state = preserve_rng_state
        
        ctx.cuda_rng_state = torch.cuda.get_rng_state() if preserve_rng_state else None
        
        tensors = []
        others = []
        for arg in args:
            if torch.is_tensor(arg):
                tensors.append(arg)
                others.append(None)
            else:
                tensors.append(None)
                others.append(arg)
        ctx.nontensor_inputs = others
        ctx.save_for_backward(*tensors)

        with torch.no_grad():
            outputs = func(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        all_inputs = []
        input_reqires_grad = []
        for tensor, other in zip(ctx.saved_tensors, ctx.nontensor_inputs):
            if tensor is None:
                all_inputs.append(other)
                input_reqires_grad.append(False)
            else:
                input_reqires_grad.append( tensor.requires_grad )
                nw_tensor = tensor.detach()
                nw_tensor.requires_grad = tensor.requires_grad
                all_inputs.append(nw_tensor)

        
        with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.cuda.set_rng_state(ctx.cuda_rng_state)
            with torch.enable_grad():
                outputs = ctx.func(*all_inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        
        assert len(outputs) == len(grad_outputs)

        outputs_with_grad = []
        grad_of_output = []
        for i, output in enumerate(outputs):
            if torch.is_tensor(output) and output.requires_grad:
                outputs_with_grad.append(output)
                grad_of_output.append(grad_outputs[i])
        
        torch.autograd.backward(
            outputs_with_grad,
            grad_of_output,
        )
        grads = []
        for inp, requires_grad in zip(all_inputs, input_reqires_grad):
            if requires_grad:
                grads.append(inp.grad)
            else:
                grads.append(None)
        return (None, None) + tuple(grads)


P = ParamSpec("P")
R = TypeVar("R")
def checkpoint(func : Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args):
        return CheckpointFunction.apply(func, True, *args)
    return wrapper
