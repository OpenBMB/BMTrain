import torch
from .global_var import config
from .checkpointing import CheckpointBlockContext
from .distributed import all_gather, broadcast, all_reduce, send_activations, recv_activations 
from collections import deque,OrderedDict
class Offload_Dict:

    def __init__(self):
        self._offload_dict = OrderedDict()
        self.offset = 0

    def add(self, tensor):
        self._offload_dict[id(tensor)] = {}
        self._offload_dict[id(tensor)]["offset"] = self.offset
        self._offload_dict[id(tensor)]["numel"] = tensor.numel()
        self._offload_dict[module_name]['dtype'] = inp.dtype

    def make_cpu_storage(self, _cuda_dict):
        with torch.cuda.stream(config["offload_stream"]):
            
def wrapper(module_name,act_cuda_dict):
    def fn(m,inps,out):
        if module_name not in act_cuda_dict:
            act_cuda_dict[module_name] = m._inp_dict
        inp = inps[0]
        act_cuda_dict[module_name]['shape'] = tuple(inp.shape)
        act_cuda_dict[module_name]['numel'] = inp.numel()
        act_cuda_dict[module_name]['inp'] = inp
        act_cuda_dict[module_name]['dtype'] = inp.dtype
        m.riginp_dict = act_cuda_dict[module_name]
    return fn


def nearest_offload_module(module):
    if module._mode == "OFFLOAD":
        return [module]
    queue = deque([(module, 0)])  # 使用队列来进行广度优先搜索
    nearest_modules = []
    nearest_depth = float('inf')
    
    while queue:
        curr_module, curr_depth = queue.popleft()
        
        if curr_depth > nearest_depth:
            break
        
        for m in curr_module._pre_module:
            if m._mode == "OFFLOAD" and not m._on_device:
                if curr_depth < nearest_depth:
                    nearest_modules = [m]
                    nearest_depth = curr_depth
                elif curr_depth == nearest_depth:
                    nearest_modules.append(m)
            else:
                queue.append((m, curr_depth + 1))
    
    return nearest_modules

def make_cpu_storage(_act_cuda_dict, _offload_dict):
    for key,val in _act_cuda_dict.items():
        if "dtype" not in val:
            print(key)
            print(val)
    fp16_total = sum([v['numel'] for v in _act_cuda_dict.values() if v['dtype'] == torch.float16])
    fp32_total = sum([v['numel'] for v in _act_cuda_dict.values() if v['dtype'] == torch.float32])
    fp16_storage = torch.HalfStorage(fp16_total).pin_memory()
    fp32_storage = torch.FloatStorage(fp32_total).pin_memory()
    fp16_offset = 0
    fp32_offset = 0
    for key,val in _act_cuda_dict.items():
        if val['dtype'] == torch.float16:
            _offload_dict[key] = {}
            _offload_dict[key]['inp'] = torch.tensor([], dtype=torch.float16, device="cpu") \
                                        .set_(fp16_storage, fp16_offset, val['shape'])

            fp16_offset += _act_cuda_dict[key]['numel']
        elif val['dtype'] == torch.float32:
            _offload_dict[key]['inp'] = torch.tensor([], dtype=torch.float32, device="cpu") \
                                        .set_(fp32_storage, fp32_offset, val['shape'])

            fp32_offset += _act_cuda_dict[key]['numel']
def d2h_memcpy(_act_cuda_dict, _offload_dict):
    for key,val in _act_cuda_dict.items():
        shape, inp = val['shape'],val['inp']
        cpu_inp = _offload_dict[key]['inp']
        _offload_dict[key]['inp'] = cpu_inp.copy_(inp, non_blocking=True)

def h2d_memcpy(_act_cuda_dict, _offload_dict):
    for key,val in _act_cuda_dict.items():
        shape, cuda_inp = val['shape'],val['inp']
        cpu_inp = _offload_dict[key]['inp']
        cuda_stor = cuda_inp.storage_type()(val['numel'])
        cuda_inp.record_stream(config["offload_stream"])
        cuda_inp.set_(cuda_stor, 0, shape)
        cuda_inp.copy_(cpu_inp, non_blocking=True)

def pack_hook(tensor):
    _offload_tensor(id_tensor) = {}
def zero_pre_forward(module, inputs):
    enter = True
    pipe = False

    if module._mode == "OFFLOAD":
        if not hasattr(module,"_act_cuda_dict"):
            torch._C._autograd._push_saved_tensors_default_hooks(
            pack_hook, unpack_hook
            )
            module._act_cuda_dict = {}
            for name, sub_module in module.named_modules():
                if sub_module.__class__.__name__ == "Linear":
                    sub_module.offload = True
                    fn = wrapper(name, module._act_cuda_dict)
                    sub_module.register_forward_hook(fn)
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
    elif module._mode == "OFFLOAD":
        torch._C._autograd._pop_saved_tensors_default_hooks()
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(config["calc_stream"])
        with torch.cuda.stream(config["offload_stream"]):
            if not hasattr(module, "_offload_dict"):
                module._offload_dict = {}
                make_cpu_storage(module._act_cuda_dict, module._offload_dict)
                d2h_memcpy(module._act_cuda_dict, module._offload_dict)
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(config["offload_stream"])
        cuda_stor = torch.UntypedStorage(1).cuda()
        for key,val in module._act_cuda_dict.items():
            module._act_cuda_dict[key]['inp'].set_(cuda_stor, 0, (1,))
            
    if exit:
        module._forward_block_ctx.exit(forward_flag)

def zero_pre_backward(module, grad_outputs):
    backward_flag = 2 if config['zero_level'] == 2 else 0
    if module._mode != "PIPE":
        if module._mode != "OFFLOAD":
            count = len([m for m in module._pre_module if m._mode=="OFFLOAD"])
            if module._is_last_layer or module._next_module[0]._mode == "OFFLOAD":
                for pre_module in nearest_offload_module(module):
                    if pre_module._mode == "OFFLOAD":
                        pre_module._on_device = True
                        with torch.cuda.stream(config["offload_stream"]):
                            h2d_memcpy(pre_module._act_cuda_dict, pre_module._offload_dict)
        else:
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(config["offload_stream"])
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
        if not module._is_first_layer and len(module._pre_module) > 0:
            module._pre_module.pop()
        if module._mode == "OFFLOAD":
            module._on_device = False
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(config["calc_stream"])
            cuda_stor = torch.UntypedStorage(1).cuda()
            with torch.cuda.stream(config["offload_stream"]):
                for key,val in module._act_cuda_dict.items():
                    inp = val['inp']
                    inp.record_stream(config["offload_stream"])
                    inp.set_(cuda_stor, 0, (1,)) 
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
