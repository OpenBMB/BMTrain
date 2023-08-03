import torch
from .global_var import config
from .checkpointing import CheckpointBlockContext
from .distributed import all_gather, broadcast, all_reduce, send_activations, recv_activations 

#torch_version = torch.__version__
torch_version = '1.9.0'

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
            config['load_stream'].record_event(config['load_event'])
    else:
        if module._micro_idx == config['micros'] - 1:
            module._backward_block_ctxs[module._layer_id] = CheckpointBlockContext(module, module._layer_dict, backward_flag, pipe=True)
            module._backward_block_ctxs[module._layer_id].enter(True)

def zero_post_backward(module, grad_inputs, grad_outputs):
    if not config['pipe_enabled']:
        if module._layer_id == 0:
            module._backward_block_ctxs[0].exit(True)
            config['load_stream'].record_event(config['load_event'])
        if torch_version < '2.0.1': 
            if not module._is_first_layer:
                identity_post_backward(module, grad_inputs, grad_outputs)
    else:
        if module._micro_idx == 0:
            module._backward_block_ctxs[module._layer_id].exit(True)
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

    if torch_version < '2.0.1':
        if not module._is_first_layer:
            identity_post_backward(module, grad_outputs, grad_inputs)

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
            #inputs[0].retain_grad()
            out = module._module(*inputs)
            torch.autograd.backward(out, *grad_outputs)

            if not config['pipe_enabled']:
                zero_post_backward(module, grad_outputs, (inputs[0].grad,) )
            else:
                zero_post_backward(module, grad_outputs, (inputs[0].grad,) )
                pipe_post_backward(module, (inputs[0].grad,), grad_outputs)

def identity_post_backward(module, grad_inputs, grad_outputs):
    if config['pipe_enabled']:
        pipe_grad = pipe_pre_backward(module._pre_module, grad_outputs)
        grad_outputs = pipe_grad if pipe_grad is not None else grad_outputs
    zero_pre_backward(module._pre_module, grad_outputs)
    if config['use_checkpoint']:
        checkpoint_pre_backward(module._pre_module, grad_outputs)

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

