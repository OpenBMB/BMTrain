import torch
from .global_var import config
from .checkpointing import CheckpointBlockContext
from .distributed import all_gather, broadcast, all_reduce, send_activations, recv_activations 

def zero_pre_forward(module, inputs):
    enter = True
    pipe = False
    if module._mode == "PIPE":
        enter = module._micro_idx == 0
        pipe = True
    if enter:
        zero_level = config['zero_level']
        forward_flag = 1 if zero_level == 2 else 0
        if zero_level == 2 and module._ref_count > 1:
            forward_flag = 2 # repeating forward in same layer
        module._forward_block_ctx = CheckpointBlockContext(module, module._layer_dict, pipe=pipe)
        module._forward_block_ctx.enter(forward_flag)

def zero_post_forward(module, inputs, outputs):
    forward_flag = 1 if config['zero_level'] == 2 else 0
    exit = True
    if module._mode == "PIPE":
        exit = module._micro_idx == config['micros'] - 1

    if exit:
        module._forward_block_ctx.exit(forward_flag)

def zero_pre_backward(module, grad_outputs):
    backward_flag = 2 if config['zero_level'] == 2 else 0
    if module._mode != "PIPE":
        module._backward_block_ctx = CheckpointBlockContext(module, module._layer_dict)
        module._backward_block_ctx.enter(backward_flag, True)
        if not module._is_last_layer and len(module._next_module) > 0 and module._next_module[-1]._backward_block_ctx is not None:
            if module._next_module[-1]._ref_count == 1:
                module._next_module[-1]._ref_count = 0
                module._next_module.pop()._backward_block_ctx.exit(backward_flag, True)
                config['load_stream'].record_event(config['load_event'])
            else:
                module._next_module[-1]._ref_count -= 1
            
    else:
        if module._micro_idx == config['micros'] - 1:
            module._backward_block_ctx = CheckpointBlockContext(module, module._layer_dict, pipe=True)
            module._backward_block_ctx.enter(backward_flag, True)

def zero_post_backward(module, grad_inputs, grad_outputs):
    backward_flag = 2 if config['zero_level'] == 2 else 0
    if module._mode != "PIPE":
        if module._is_first_layer and module._ref_count == 1:
            module._backward_block_ctx.exit(backward_flag, True)
            module._ref_count = -1
            config['load_stream'].record_event(config['load_event'])
    else:
        if module._micro_idx == 0:
            module._ref_count = -1 if module._is_first_layer else 0
            module._backward_block_ctx.exit(backward_flag, True)
            config['load_stream'].record_event(config['load_event'])

class PipePreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, stage_id):
        pre_inputs = recv_activations(stage_id - 1, config['pipe_comm'])
        pre_inputs.requires_grad_()
        return pre_inputs 

    @staticmethod
    def backward(ctx, grads):
        return grads, None

def pipe_pre_forward(module, inputs):
    if not module._is_first_stage:
        if module._is_first_layer:
            return (PipePreFunction.apply(inputs[0], module.stage_id), ) +  inputs[1:]

def pipe_post_forward(module, inputs, outputs):
    if not module._is_last_stage:
        if module._is_last_layer:
            send_data = outputs[0] if isinstance(outputs, tuple) else outputs
            send_activations(send_data.detach(), module.stage_id + 1, config['pipe_comm'])

def pipe_pre_backward(module, grad_inputs):
    if not module._is_last_stage:
        if module._is_last_layer:
            pre_grad_inputs = recv_activations(module.stage_id + 1, config['pipe_comm'])
            return (pre_grad_inputs, ) + grad_inputs[1:]
            
def pipe_post_backward(module, grad_inputs, grad_outputs):
    if not module._is_first_stage:
        if module._is_first_layer:
            send_data = grad_inputs[0] if isinstance(grad_inputs, tuple) else grad_inputs 
            send_activations(send_data, module.stage_id - 1, config['pipe_comm'])

    module._micro_idx -= 1

class PreHookFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *x):
        ctx.module = module
        if module._mode == "PIPE":
            pipe_out = pipe_pre_forward(module, x)
            x = pipe_out if pipe_out is not None else x
       
        if module.return_hidden_states:
            module.hidden_states.append(x[0])
        zero_pre_forward(module, x)
        return x

    @staticmethod
    def backward(ctx, *grads):
        zero_post_backward(ctx.module, grads, None)
        if ctx.module._mode == "PIPE":
            pipe_post_backward(ctx.module, grads, None)
        return None, *grads

class PostHookFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *out):
        ctx.module = module
        zero_post_forward(module, None, out)
        if module._mode == "PIPE":
            pipe_post_forward(module, None, out)
        return out

    @staticmethod
    def backward(ctx, *grads):
        zero_pre_backward(ctx.module, grads)
        if ctx.module._mode == "PIPE":
            pipe_grads = pipe_pre_backward(ctx.module, grads)
            grads = pipe_grads[0] if pipe_grads is not None else grads 
            if not isinstance(grads, tuple):
                return None, grads
        return None, *grads
