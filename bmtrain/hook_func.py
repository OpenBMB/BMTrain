import torch
from .global_var import config
from .checkpointing import CheckpointBlockContext

def zero_pre_forward(module, inputs):
    forward_flag = 1 if config['zero_level'] == 2 else 0
    module._forward_block_ctx = CheckpointBlockContext(module, module._layer_dict, forward_flag)
    module._forward_block_ctx.enter()

def zero_post_forward(module, inputs, outputs):
    module._forward_block_ctx.exit()

def zero_pre_backward(module, grad_outputs):
    backward_flag = 2 if config['zero_level'] == 2 else 0
    with torch.enable_grad():
        module._backward_block_ctxs[module._layer_id] = CheckpointBlockContext(module, module._layer_dict, backward_flag)
        module._backward_block_ctxs[module._layer_id].enter(True)
        if not module._is_last_layer:
            module._backward_block_ctxs[module._layer_id + 1].exit(True)
            module._backward_block_ctxs[module._layer_id + 1] = None


def zero_post_backward(module, grad_inputs, grad_outputs):
    if module._layer_id == 0:
        module._backward_block_ctxs[0].exit(True)
        module._backward_block_ctxs[0] = None

def checkpoint_pre_forward(module, inputs):
    module._inputs = inputs
    module._cuda_rng_state = torch.cuda.get_rng_state()

def checkpoint_pre_backward(module, grad_outputs):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=True):
        with torch.enable_grad():
            torch.cuda.set_rng_state(module._cuda_rng_state)
            out = module._module(*module._inputs)
            torch.autograd.backward(out, *grad_outputs)

            if module._layer_id == 0:
                module._backward_block_ctxs[0].exit(True)
                module._backward_block_ctxs[0] = None

    
