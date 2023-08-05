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
        forward_flag = 1 if config['zero_level'] == 2 else 0
        module._forward_block_ctx = CheckpointBlockContext(module, module._layer_dict, forward_flag, pipe=pipe)
        module._forward_block_ctx.enter()

def zero_post_forward(module, inputs, outputs):
    exit = True
    if module._mode == "PIPE":
        exit = module._micro_idx == config['micros'] - 1

    if exit:
        module._forward_block_ctx.exit()

def zero_pre_backward(module, grad_outputs):
    backward_flag = 2 if config['zero_level'] == 2 else 0
    if module._mode != "PIPE":
        module._backward_block_ctx = CheckpointBlockContext(module, module._layer_dict, backward_flag)
        module._backward_block_ctx.enter(True)
        if not module._is_last_layer and module._next_module is not None and module._next_module._backward_block_ctx is not None:
            module._next_module._backward_block_ctx.exit(True)
            config['load_stream'].record_event(config['load_event'])
    else:
        if module._micro_idx == config['micros'] - 1:
            module._backward_block_ctx = CheckpointBlockContext(module, module._layer_dict, backward_flag, pipe=True)
            module._backward_block_ctx.enter(True)

def zero_post_backward(module, grad_inputs, grad_outputs):
    if module._mode != "PIPE":
        if module._is_first_layer:
            module._backward_block_ctx.exit(True)
            config['load_stream'].record_event(config['load_event'])
    else:
        if module._micro_idx == 0:
            module._backward_block_ctx.exit(True)
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

class PipeAllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_state):
        hidden_state_list = all_gather(hidden_state.clone(), config["pipe_comm"])
        hidden_state_list.requires_grad_()
        return hidden_state_list

    @staticmethod
    def backward(ctx, grads):
        grads = broadcast(grads, 0, config['pipe_comm'])
        topo = config['topology']
        return grads.chunk(topo.stages, dim=0)[topo.stage_id] 

class PipePostFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, last_hidden):
        last_hidden = broadcast(last_hidden, config["pipe_size"] - 1, config["pipe_comm"])
        last_hidden = last_hidden.chunk(config['topology'].stages, dim=0)
        outputs = last_hidden[config['topology'].stage_id]
        outputs.requires_grad_()
        return outputs

    @staticmethod
    def backward(ctx, grads):
        grad_list = all_gather(grads, config["pipe_comm"])
        grad_list = grad_list.flatten(start_dim=0, end_dim=1)
        return grad_list

class PreHookFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, x):
        ctx.module = module
        if module._mode == "PIPE":
            pipe_out = pipe_pre_forward(module, (x,))
            x = pipe_out[0] if pipe_out is not None else x
        zero_pre_forward(module, x)
        return x

    @staticmethod
    def backward(ctx, grads):
        zero_post_backward(ctx.module, grads, None)
        if ctx.module._mode == "PIPE":
            pipe_post_backward(ctx.module, grads, None)
        return None, grads

class PostHookFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, out):
        ctx.module = module
        zero_post_forward(module, None, out)
        if module._mode == "PIPE":
            pipe_post_forward(module, None, out)
        return out

    @staticmethod
    def backward(ctx, grads):
        zero_pre_backward(ctx.module, grads)
        if ctx.module._mode == "PIPE":
            pipe_grads = pipe_pre_backward(ctx.module, (grads, ))
            grads = pipe_grads[0] if pipe_grads is not None else grads 
        return None, grads
