import torch
from .global_var import config
from .checkpointing import CheckpointBlockContext
from .distributed import all_gather, broadcast, all_reduce, send_activations, recv_activations 

torch_version = torch.__version__
#torch_version = '1.9.0'

def zero_pre_forward(module, inputs):
    enter = True
    pipe = False
    if config['pipe_enabled']:
        enter = module._micro_idx == 0
        pipe = True
    if enter:
        forward_flag = 1 if config['zero_level'] == 2 else 0
        module._forward_block_ctx = CheckpointBlockContext(module, module._layer_dict, forward_flag, pipe=pipe)
        module._forward_block_ctx.enter()

def zero_post_forward(module, inputs, outputs):
    exit = True
    if config['pipe_enabled']:
        exit = module._micro_idx == config['micros'] - 1

    if exit:
        module._forward_block_ctx.exit()

def zero_pre_backward(module, grad_outputs):
    backward_flag = 2 if config['zero_level'] == 2 else 0
    if not config['pipe_enabled']:
        module._backward_block_ctxs[module._layer_id] = CheckpointBlockContext(module, module._layer_dict, backward_flag)
        module._backward_block_ctxs[module._layer_id].enter(True)
        if not module._is_last_layer:
            module._backward_block_ctxs[module._layer_id + 1].exit(True)
    else:
        if module._micro_idx == config['micros'] - 1:
            module._backward_block_ctxs[module._layer_id] = CheckpointBlockContext(module, module._layer_dict, backward_flag, pipe=True)
            module._backward_block_ctxs[module._layer_id].enter(True)

def zero_post_backward(module, grad_inputs, grad_outputs):
    if not config['pipe_enabled']:
        if module._layer_id == 0:
            module._backward_block_ctxs[0].exit(True)
    else:
        if module._micro_idx == 0:
            module._backward_block_ctxs[module._layer_id].exit(True)

    if torch_version < '2.0.1':
        if module._layer_id != 0:
            zero_pre_backward(module._pre_module, grad_inputs)

def pipe_pre_forward(module, inputs):
    if not module._is_first_stage:
        if module._is_first_layer:
            pre_inputs = recv_activations(module.stage_id - 1, config['pipe_comm'])
            pre_inputs.requires_grad_()
            return (pre_inputs, ) + inputs[1:]

def pipe_post_forward(module, inputs, outputs):
    if not module._is_last_stage:
        if module._is_last_layer:
            send_data = outputs[0] if isinstance(outputs, tuple) else outputs
            send_activations(send_data, module.stage_id + 1, config['pipe_comm'])

def pipe_pre_backward(module, grad_inputs):
    if module._is_last_layer:
        module._grad_list = all_gather(grad_inputs[0], config["pipe_comm"])
        module._grad_list = module._grad_list.flatten(start_dim=0, end_dim=1).chunk(module.stages, dim=0)

    if module._is_last_layer and module._is_last_stage:
        return (module._grad_list[module._micro_idx], )

    if not module._is_last_stage:
        if module._is_last_layer:
            pre_grad_inputs = recv_activations(module.stage_id + 1, config['pipe_comm'])
            return (pre_grad_inputs, ) + grad_inputs[1:]
            

def pipe_post_backward(module, grad_inputs, grad_outputs):
    if not module._is_first_stage:
        if module._is_first_layer:
            send_data = grad_inputs[0] if isinstance(grad_inputs, tuple) else grad_inputs 
            if send_data is not None:
                send_activations(send_data, module.stage_id - 1, config['pipe_comm'])

#    if module._is_first_layer:
#        if module._micro_idx == config['micros'] -1:
#            module._all_grads = []
#        grad = grad_inputs[0] 
#        module._all_grads.append(grad)
#        if module._micro_idx == 0:
#            grads = torch.cat(module._all_grads, dim=0)
#            grad = broadcast(grads, 0, config['pipe_comm'])
#            grad = grad.chunk(module.stages, dim=0)
#            return (grad[module.stage_id], ) + grad_inputs[1:]

    module._micro_idx -= 1

def checkpoint_pre_forward(module, inputs):
    if not config['pipe_enabled']:
        module._inputs = inputs
        module._cuda_rng_state = torch.cuda.get_rng_state()
    else:
        if module._micro_idx == 0:
            module._inputs = [inputs]
            module._cuda_rng_state = [torch.cuda.get_rng_state()]
        else:
            module._inputs.append(inputs)
            module._cuda_rng_state.append(torch.cuda.get_rng_state())

def checkpoint_pre_backward(module, grad_outputs):
    inputs = module._inputs if not config['pipe_enabled'] else module._inputs[module._micro_idx]
    cuda_rng_state = module._cuda_rng_state if not config['pipe_enabled'] else module._cuda_rng_state[module._micro_idx]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=True):
        with torch.enable_grad():
            torch.cuda.set_rng_state(cuda_rng_state)
            out = module._module(*inputs)
            torch.autograd.backward(out, *grad_outputs)

            if not config['pipe_enabled']:
                if module._layer_id == 0:
                    module._backward_block_ctxs[0].exit(True)
                    module._backward_block_ctxs[0] = None
            else:
                zero_post_backward(module, None, grad_outputs)
                pipe_post_backward(module, module._inputs[module._micro_idx][0].grad, None)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *args):
        inputs = args[0].detach()
        ctx.module = module
        with torch.no_grad():
            zero_pre_forward(module, args) 
            checkpoint_pre_forward(module, args) 
            outputs = module._module(inputs, *args[1:])
            outputs.requires_grad_()
            zero_post_forward(module, args, outputs) 
            return outputs

    @staticmethod
    def backward(ctx, grads):
        with torch.enable_grad():
            zero_pre_backward(ctx.module, grads)
            checkpoint_pre_backward(ctx.module, grads)
            return None, ctx.module._inputs[0].grad, None, None

def identity_post_backward(module, grad_inputs, grad_outputs):
    zero_pre_backward(module._pre_module, grad_inputs)

class IdentityLayer(torch.nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

