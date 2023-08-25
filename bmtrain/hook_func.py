import torch
from .global_var import config
from .checkpointing import CheckpointBlockContext
from .distributed import all_gather, broadcast, all_reduce, send_activations, recv_activations 
from collections import deque,OrderedDict
from contextlib import contextmanager
from .utils import round_up

class Offload_Dict:

    def __init__(self):
        self._offload_dict = OrderedDict()

    def add(self, tensor):
        tensor = tensor.contiguous()
        tensor_id = id(tensor)
        data_ptr = tensor.storage().data_ptr()
        if data_ptr not in self._offload_dict:
            self._offload_dict[data_ptr] = {}
            self._offload_dict[data_ptr]["stor"] = tensor.storage()
            self._offload_dict[data_ptr]["size"] = tensor.storage().size()
            self._offload_dict[data_ptr]["dtype"] = tensor.storage().dtype
            self._offload_dict[data_ptr]["tensors"] = {}
        self._offload_dict[data_ptr]["tensors"][id(tensor)] = {}
        self._offload_dict[data_ptr]["tensors"][id(tensor)]["numel"] = tensor.numel()
        self._offload_dict[data_ptr]["tensors"][id(tensor)]['dtype'] = tensor.dtype
        self._offload_dict[data_ptr]["tensors"][id(tensor)]['offset'] = tensor.storage_offset()
        self._offload_dict[data_ptr]["tensors"][id(tensor)]['tensor'] = tensor
        self._offload_dict[data_ptr]["tensors"][id(tensor)]["shape"] = tensor.shape
        self._device = "cuda"
        return (data_ptr,tensor_id)
    
    def get_total(self):
        fp16_total = sum([v['size'] for v in self._offload_dict.values() if v['dtype'] == torch.float16])
        fp32_total = sum([v['size'] for v in self._offload_dict.values() if v['dtype'] == torch.float32])        
        return fp16_total,fp32_total

    def make_cpu_storage(self):
        fp16_total = sum([v['size'] for v in self._offload_dict.values() if v['dtype'] == torch.float16])
        fp32_total = sum([v['size'] for v in self._offload_dict.values() if v['dtype'] == torch.float32])
        fp16_storage = torch.HalfStorage(fp16_total).pin_memory()
        fp32_storage = torch.FloatStorage(fp32_total).pin_memory()
        self.fp16_storage = fp16_storage
        self.fp32_storage = fp32_storage 
        self.fp16_total = fp16_total
        self.fp32_total = fp32_total

    def get(self, key):
        data_ptr, tensor_id = key
        return self._offload_dict[data_ptr]['tensors'][tensor_id]["tensor"]

    def pop_all(self):
        self._offload_dict.clear()

    def h2d_memcpy(self):
        fp16_storage_cuda = self.fp16_storage.cuda(non_blocking=True)
        fp32_storage_cuda = self.fp32_storage.cuda(non_blocking=True)
        for key,val in self._offload_dict.items():
            for id_val in val['tensors'].values():
                id_val['tensor'] = torch.tensor([], dtype=id_val['dtype'],device=fp16_storage_cuda.device)
                if id_val['dtype'] == torch.float16:
                    id_val['tensor'].set_(fp16_storage_cuda, id_val['abs_offset'], id_val['shape'])
                elif id_val['dtype'] == torch.float32:
                    id_val['tensor'].set_(fp32_storage_cuda, id_val['abs_offset'], id_val['shape'])


    def record_stream(self, stream):
        for key, val in self._offload_dict.items():
            for id_val in val['tensors'].values():
                id_val['tensor'].record_stream(stream)

    def d2h_memcpy(self):   
        fp16_offset = 0
        fp32_offset = 0
        fp16_total = sum([v['size'] for v in self._offload_dict.values() if v['dtype'] == torch.float16])
        fp32_total = sum([v['size'] for v in self._offload_dict.values() if v['dtype'] == torch.float32])
        assert fp16_total <= self.fp16_total
        assert fp32_total <= self.fp32_total
        fp16_storage = self.fp16_storage
        fp32_storage = self.fp32_storage
        for key,val in self._offload_dict.items():
            assert val['dtype'] in [torch.float16, torch.float32]
            storage = fp16_storage if val['dtype'] == torch.float16 else fp32_storage
            offset = fp16_offset if val['dtype'] == torch.float16 else fp32_offset
            for id_val in val['tensors'].values():
                cpu_tensor = torch.tensor([], dtype=id_val['dtype'], device="cpu") \
                    .set_(storage, offset+id_val['offset'], id_val['shape'])
                id_val["abs_offset"] = offset+id_val['offset']
                id_val['tensor'] = cpu_tensor.copy_(id_val['tensor'], non_blocking=True)
            if val['dtype'] == torch.float16:
                fp16_offset += val['size']
            else:
                fp32_offset += val['size']
            val['stor'] = None

def nearest_offload_module(module):
    queue = deque([(module, 0)]) 
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
        elif tensor.dtype not in [torch.float16]:
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


def offload_pre_hook(module, input):
   if hasattr(module, "_offload_hook"):
        pack_hook, unpack_hook = module._offload_hook
        torch._C._autograd._push_saved_tensors_default_hooks(
            pack_hook, unpack_hook
        ) 

def offload_post_hook(module, input, output):
    if hasattr(module, "_offload_hook"):
        torch._C._autograd._pop_saved_tensors_default_hooks()

def zero_pre_forward(module, inputs):
    def find_pre_module_helper(m):
        if m._mode == "OFFLOAD":
            return m
        elif m._is_first_layer:
            return None
        else:
            return find_pre_module_helper(m._pre_module[0])
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
    elif module._mode != "OFFLOAD" and ((len(module._pre_module) > 0) and module._pre_module[0]._mode == "OFFLOAD"):
        pre_module = module._pre_module[0]
        if len(pre_module._pre_module) == 0:
            pre_offload_module = None
        else:
            pre_offload_module = find_pre_module_helper(pre_module._pre_module[0])
        if pre_offload_module is not None:
            torch.cuda.current_stream().wait_event(pre_offload_module.offload_event)
        if pre_module._mode == "OFFLOAD":
            with torch.cuda.stream(config["offload_stream"]):
                config["offload_stream"].wait_event(pre_module.calc_event)
                if not hasattr(pre_module._offload_dict, "fp16_storage"):
                    pre_module._offload_dict.make_cpu_storage()
                pre_module._offload_dict.record_stream(config["offload_stream"])
                pre_module._offload_dict.d2h_memcpy()
                if len(module._next_module) > 0:
                    config["offload_stream"].record_event(pre_module.offload_event)
            
    if module._mode == "PIPE":
        enter = module._micro_idx == 0
        pipe = True
    if enter:
        zero_level = config['zero_level']
        forward_flag = 1 if zero_level == 2 else 0
        if zero_level == 2 and module._ref_count > 1:
            forward_flag = 2 # repeating forward in same layer
        if module.all_param_no_grad: #only forward
            forward_flag = 0
        module._forward_block_ctx = CheckpointBlockContext(module, module._layer_dict, pipe=pipe)
        module._forward_block_ctx.enter(forward_flag)

def zero_post_forward(module, inputs, outputs):
    forward_flag = 1 if config['zero_level'] == 2 else 0
    if module.all_param_no_grad:
        forward_flag = 0
    exit = True
    if module._mode == "PIPE":
        exit = module._micro_idx == config['micros'] - 1
    elif module._mode == "OFFLOAD":
        if module.offload_level == 2:
            torch._C._autograd._pop_saved_tensors_default_hooks()
        torch.cuda.current_stream().record_event(module.calc_event)
    if exit:
        module._forward_block_ctx.exit(forward_flag)
        module._ref_count += 1

def zero_pre_backward(module, grad_outputs):
    def find_pre_module_helper(m):
        if m._mode == "OFFLOAD":
            return m
        else:
            if len(m._pre_module) != 0:
                return find_pre_module_helper(m._pre_module[0])
            else:
                return None
    backward_flag = 2 if config['zero_level'] == 2 else 0
    if module._mode != "PIPE":
        if module._mode != "OFFLOAD":
            count = len([m for m in module._pre_module if m._mode=="OFFLOAD"])
            if (len(module._next_module) == 0) or module._next_module[0]._mode == "OFFLOAD":
                pre_module = find_pre_module_helper(module)
                if pre_module is not None:
                    pre_module._on_device = True
                    with torch.cuda.stream(config["offload_stream"]):
                        if (len(module._next_module) != 0):
                            torch.cuda.current_stream().wait_event(module._next_module[0].calc_event)
                        pre_module._offload_dict.h2d_memcpy()
                        torch.cuda.current_stream().record_event(pre_module.offload_event)
        else:
            current_stream = torch.cuda.current_stream()
            current_stream.wait_event(module.offload_event)
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
    backward_flag = 2 if config['zero_level'] == 2 else 0
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
