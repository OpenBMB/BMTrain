import torch
from collections import OrderedDict

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