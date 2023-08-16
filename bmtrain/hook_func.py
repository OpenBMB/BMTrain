import torch
from .global_var import config
from .checkpointing import CheckpointBlockContext
from .distributed import all_gather, broadcast, all_reduce, send_activations, recv_activations 
from collections import deque,OrderedDict
class Offload_Dict:

    def __init__(self):
        self._offload_dict = OrderedDict()

    def add(self, tensor):
        tensor_id = id(tensor)
        self._offload_dict[tensor_id] = {}
        self._offload_dict[tensor_id]["numel"] = tensor.numel()
        self._offload_dict[tensor_id]['dtype'] = tensor.dtype
        self._offload_dict[tensor_id]["tensor"] = tensor
        self._offload_dict[tensor_id]["shape"] = tensor.shape
        self._device = "cuda"
        return tensor_id

    def make_cpu_storage(self):
        fp16_total = sum([v['numel'] for v in self._offload_dict.values() if v['dtype'] == torch.float16])
        fp32_total = sum([v['numel'] for v in self._offload_dict.values() if v['dtype'] == torch.float32])
        fp16_storage = torch.HalfStorage(fp16_total).pin_memory()
        fp32_storage = torch.FloatStorage(fp32_total).pin_memory()
        self.fp16_storage = fp16_storage
        self.fp32_storage = fp32_storage 
        self.fp16_total = fp16_total
        self.fp32_total = fp32_total

    def get(self, key):
        return self._offload_dict[key]["tensor"]

    def pop_all(self):
        self._offload_dict = OrderedDict()

    def h2d_memcpy(self):
        for key,val in self._offload_dict.items():
            self._offload_dict[key]['tensor'] = self._offload_dict[key]['tensor'].cuda(non_blocking=True)

    def record_stream(self, stream):
        for key, val in self._offload_dict.items():
            self._offload_dict[key]['tensor'].record_stream(stream)

    def d2h_memcpy(self):   
        fp16_offset = 0
        fp32_offset = 0
        fp16_total = sum([v['numel'] for v in self._offload_dict.values() if v['dtype'] == torch.float16])
        fp32_total = sum([v['numel'] for v in self._offload_dict.values() if v['dtype'] == torch.float32])
        assert fp16_total <= self.fp16_total
        assert fp32_total <= self.fp32_total
        fp16_storage = self.fp16_storage
        fp32_storage = self.fp32_storage
        for key,val in self._offload_dict.items():
            assert val['dtype'] in [torch.float16, torch.float32]
            storage = fp16_storage if val['dtype'] == torch.float16 else fp32_storage
            offset = fp16_offset if val['dtype'] == torch.float16 else fp32_offset
            cpu_tensor = torch.tensor([], dtype=val['dtype'], device="cpu") \
                .set_(storage, offset, val['shape'])
            self._offload_dict[key]['tensor'].record_stream(config['offload_stream'])
            self._offload_dict[key]['tensor'] = cpu_tensor.copy_(self._offload_dict[key]['tensor'], non_blocking=True)
            if val['dtype'] == torch.float16:
                fp16_offset += self._offload_dict[key]['numel']
            else:
                fp32_offset += self._offload_dict[key]['numel']

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


def offload_wrapper(offload_dict):
    def pack_hook(tensor):
        if isinstance(tensor, torch.nn.Parameter):
            return (tensor,) 
        else:
            key = offload_dict.add(tensor)
            return (tensor.device, key)
    def unpack_hook(packed):
        if len(packed) == 2:
            device, key = packed
            tensor = offload_dict.get(key)
            assert tensor.device == device
            return tensor
        else:
            tensor, = packed
            return tensor
    return pack_hook, unpack_hook

def zero_pre_forward(module, inputs):
    enter = True
    pipe = False
    if module._mode == "OFFLOAD":
        module._offload_dict = Offload_Dict()
        pack_hook, unpack_hook = offload_wrapper(module._offload_dict)
        for n, m in module.named_modules():
            if m.__class__.__name__ == "Linear":
                m._offload_hook = (pack_hook, unpack_hook)
        # torch._C._autograd._push_saved_tensors_default_hooks(
        #     pack_hook, unpack_hook
        # )
            
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
        # torch._C._autograd._pop_saved_tensors_default_hooks()
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(config["calc_stream"])
        with torch.cuda.stream(config["offload_stream"]):
            if not hasattr(module._offload_dict, "fp16_storage"):
                module._offload_dict.make_cpu_storage()
            module._offload_dict.d2h_memcpy()
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
                            pre_module._offload_dict.h2d_memcpy()
        else:
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(config["offload_stream"])
            module._offload_dict.record_stream(config["calc_stream"])
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
            with torch.cuda.stream(config["offload_stream"]):
                module._offload_dict.pop_all()
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
