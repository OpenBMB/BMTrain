import torch
from .global_var import config
from .checkpointing import CheckpointBlockContext
from .distributed import all_gather, broadcast, all_reduce, send_activations, recv_activations 
from collections import deque,OrderedDict
from contextlib import contextmanager
from .utils import round_up, find_pre_module_helper
from .offload import Offload_Dict, offload_wrapper, offload_pre_hook, offload_post_hook
def zero_pre_forward(module, inputs):
    enter = True
    pipe = False
    if module._mode == "OFFLOAD":
        if not hasattr(module, "_offload_dict"):
            module._offload_dict = Offload_Dict()
        pack_hook, unpack_hook = offload_wrapper(module._offload_dict)
        if module.offload_level == 1:
            for n, m in module.named_modules():
                if m.__class__.__name__ == "Linear" and not hasattr(m, "_offload_hook"):
                    m._offload_hook = (pack_hook, unpack_hook)
                    m.register_forward_pre_hook(offload_pre_hook)
                    m.register_forward_hook(offload_post_hook)
        elif module.offload_level == 2:
            if not hasattr(module, "_offload_hook"):
                module._offload_hook = (pack_hook, unpack_hook)
            torch._C._autograd._push_saved_tensors_default_hooks(
                pack_hook, unpack_hook
            )
            
    if module._mode == "PIPE":
        enter = module._micro_idx == 0
        pipe = True
    if enter:
        zero_level = module._zero_level 
        forward_flag = 1 if zero_level == 2 else 0
        if zero_level == 2 and module._ref_count > 1:
            forward_flag = 2 # repeating forward in same layer
        if module.all_param_no_grad: #only forward
            forward_flag = 0
        module._forward_block_ctx = CheckpointBlockContext(module, module._layer_dict, pipe=pipe)
        module._forward_block_ctx.enter(forward_flag)

def zero_post_forward(module, inputs, outputs):
    forward_flag = 1 if module._zero_level == 2 else 0
    if module.all_param_no_grad:
        forward_flag = 0
    exit = True
    if module._mode == "PIPE":
        exit = module._micro_idx == config['micros'] - 1
    elif module._mode == "OFFLOAD":
        torch.cuda.current_stream().record_event(module.calc_event)
        pre_offload_module = find_pre_module_helper(module.pre_module())
        if pre_offload_module is not None:
            torch.cuda.current_stream().wait_event(pre_offload_module.offload_event)
        with torch.cuda.stream(config["offload_stream"]):
            config["offload_stream"].wait_event(module.calc_event)
            if not hasattr(module._offload_dict, "fp16_storage"):
                module._offload_dict.make_cpu_storage()
            module._offload_dict.record_stream(config["offload_stream"])
            module._offload_dict.d2h_memcpy()
            if len(module._next_module) > 0:
                config["offload_stream"].record_event(module.offload_event)
        if module.offload_level == 2:
            torch._C._autograd._pop_saved_tensors_default_hooks()
    if exit:
        module._forward_block_ctx.exit(forward_flag)
        module._ref_count += 1

def zero_pre_backward(module, grad_outputs):
    backward_flag = 2 if module._zero_level == 2 else 0
    if module._mode != "PIPE":
        if module._mode == "OFFLOAD" or (len(module._next_module) == 0):
            if len(module._next_module) != 0:
                current_stream = torch.cuda.current_stream()
                current_stream.wait_event(module.offload_event)
            pre_module = find_pre_module_helper(module.pre_module())
            if pre_module is not None:
                pre_module._on_device = True
                with torch.cuda.stream(config["offload_stream"]):
                    if (len(module._next_module) != 0):
                        torch.cuda.current_stream().wait_event(module.calc_event)
                    pre_module._offload_dict.h2d_memcpy()
                    torch.cuda.current_stream().record_event(pre_module.offload_event)
            if (len(module._next_module) != 0):
                module._offload_dict.record_stream(current_stream)
        module._backward_block_ctx = CheckpointBlockContext(module, module._layer_dict)
        module._backward_block_ctx.enter(backward_flag, True)
        if not module._is_last_layer: 
            module.next_module().backward_release(backward_flag)
    else:
        if module._micro_idx == config['micros'] - 1:
            module._backward_block_ctx = CheckpointBlockContext(module, module._layer_dict, pipe=True)
            module._backward_block_ctx.enter(backward_flag, True)

def zero_post_backward(module, grad_inputs, grad_outputs):
    backward_flag = 2 if module._zero_level == 2 else 0
    if module._mode != "PIPE":
        if module._mode == "OFFLOAD":
            module._on_device = False
            module._offload_dict.pop_all()
            torch.cuda.current_stream().record_event(module.calc_event)
        if module._is_first_layer: 
            module.backward_release(backward_flag)
    else:
        if module._micro_idx == 0:
            module.backward_release(backward_flag)
        module._micro_idx -= 1

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
