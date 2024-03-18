import torch
from .global_var import config
from .zero_context import ZeroContext

def zero_pre_forward(module, inputs):
    enter = True
    if module._mode == "PIPE" or module._mode == "1F1B":
        if not hasattr(module, "_micro_forward_idx") or module._micro_forward_idx == -1:
            module._micro_forward_idx = 0
            enter = True
        else:
            enter = False
            module._micro_forward_idx += 1
    if enter:
        zero_level = module._zero_level 
        forward_flag = 1 if zero_level == 2 else 0
        if zero_level == 2 and not module._need_release:
            forward_flag = 2 # repeating forward in same layer
        if module.all_param_no_grad: #only forward
            forward_flag = 0
        if module._mode == "1F1B":
            module._block_ctx = ZeroContext(module, module._layer_dict)
            module._block_ctx.enter(0, requires_grad=True)
        else:
            module._forward_block_ctx = ZeroContext(module, module._layer_dict)
            module._forward_block_ctx.enter(forward_flag)

def zero_post_forward(module, inputs, outputs):
    forward_flag = 1 if module._zero_level == 2 else 0
    if module.all_param_no_grad:
        forward_flag = 0
    exit = True
    if module._mode == "PIPE" or module._mode == "1F1B":
        if module._micro_forward_idx == config["micros"] - 1:
            module._micro_forward_idx = -1
            if module._mode == "1F1B":
                exit = False
            else:
                exit = True
        else:
            exit = False

    if exit:
        module._forward_block_ctx.exit(forward_flag)

def zero_pre_backward(module, grad_outputs):
    backward_flag = 2 if module._zero_level == 2 else 0
    if module._mode != "PIPE" and module._mode != "1F1B":
        module._backward_block_ctx = ZeroContext(module, module._layer_dict)
        module._backward_block_ctx.enter(backward_flag, True)
        module.release_next_module(backward_flag)
    else:
        if not hasattr(module, "_micro_backward_idx") or module._micro_backward_idx == -1:
            if module._mode == "1F1B":
                module._micro_backward_idx = 0
            else:
                module._micro_backward_idx = 0
                module._backward_block_ctx = ZeroContext(module, module._layer_dict)
                module._backward_block_ctx.enter(backward_flag,requires_grad=True)
        else:
            module._micro_backward_idx += 1

def zero_post_backward(module, grad_inputs, grad_outputs):
    backward_flag = 2 if module._zero_level == 2 else 0
    if module._mode != "PIPE" and module._mode != "1F1B":
        if module._is_first_layer: 
            module.release(backward_flag)
    else:
        if module._micro_backward_idx == config["micros"] - 1:
            if module._mode == "1F1B":
                module._block_ctx.exit(0, backward=True)
                config['load_stream'].record_event(config['load_event'])
            else:
                module.release(backward_flag)
            module._micro_backward_idx = -1

class OneStepNoGradFunc(torch.autograd.Function):
    """
        requires_grad = False for all inputs
    """
    @staticmethod
    def forward(ctx, module, placeholder, *x):
        ctx.x = x
        ctx.module = module
        ctx.rng_state = torch.cuda.get_rng_state()

        with torch.no_grad():
            out = module._module(*x)
        zero_post_forward(module, None, out)
        if not isinstance(out, torch.Tensor):
            return tuple(out)
        return out

    @staticmethod
    def backward(ctx, grads):
        zero_pre_backward(ctx.module, grads)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=True):
            torch.cuda.set_rng_state(ctx.rng_state)
            x = ctx.x
            with torch.enable_grad():
                out = ctx.module._module(*x)
                torch.autograd.backward(out, grads)
        zero_post_backward(ctx.module, grads, None)
        grads = []
        for _ in x:
            grads.append(None)
        return None, None, *grads 

class PreHookFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *x):
        ctx.module = module
        zero_pre_forward(module, x)
        return x

    @staticmethod
    def backward(ctx, *grads):
        zero_post_backward(ctx.module, grads, None)
        return None, *grads

class PostHookFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *out):
        ctx.module = module
        zero_post_forward(module, None, out)
        return out

    @staticmethod
    def backward(ctx, *grads):
        zero_pre_backward(ctx.module, grads)
        return None, *grads
