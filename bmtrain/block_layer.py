from typing import Dict, Iterable, Iterator, Union, List

from .utils import round_up
from .global_var import config
import torch
from . import nccl
from .synchronize import wait_loader
from .parameter import DistributedParameter, OpAllGather
from .checkpointing import (
        ScopedTensorInspectorContext,
        CheckpointBlockContext
)

from . import debug

from . import hook_func

import copy
import inspect
from torch.utils.checkpoint import checkpoint

def storage_type_cuda(storage_type):
    STORAGE_MAP = {
        torch.FloatStorage: torch.cuda.FloatStorage,
        torch.DoubleStorage: torch.cuda.DoubleStorage,
        torch.HalfStorage: torch.cuda.HalfStorage,
        torch.BFloat16Storage: torch.cuda.BFloat16Storage,
        torch.CharStorage: torch.cuda.CharStorage,
        torch.ByteStorage: torch.cuda.ByteStorage,
        torch.ShortStorage: torch.cuda.ShortStorage,
        torch.IntStorage: torch.cuda.IntStorage,
        torch.cuda.FloatStorage: torch.cuda.FloatStorage,
        torch.cuda.DoubleStorage: torch.cuda.DoubleStorage,
        torch.cuda.HalfStorage: torch.cuda.HalfStorage,
        torch.cuda.BFloat16Storage: torch.cuda.BFloat16Storage,
        torch.cuda.CharStorage: torch.cuda.CharStorage,
        torch.cuda.ByteStorage: torch.cuda.ByteStorage,
        torch.cuda.ShortStorage: torch.cuda.ShortStorage,
        torch.cuda.IntStorage: torch.cuda.IntStorage,
    }
    if storage_type not in STORAGE_MAP:
        raise ValueError("Unknown storage type: {}".format(storage_type))
    return STORAGE_MAP[storage_type]

def _get_param_kw(param : DistributedParameter):
    type_name = str(param.dtype).split(".")[-1]
    grad_name = "_grad" if param.requires_grad else "_nograd"
    group_name = ""
    if param.group is not None:
        group_name = "_g_" + param.group
    return type_name + grad_name + group_name

class BMTBlock(torch.nn.Module):
    """ A bmtrain block containing two memory-saving methods of ZeRO-2/3 and checkpoint.

    Checkpoint block is used to save the occupation of GPU memory in training.

    For details, please refer to `Checkpointing <https://pytorch.org/docs/stable/checkpoint.html>`_ .

    Args:
        model (torch.nn.Module): The model to be checkpointed. All kinds of modules are supported.
        use_checkpoint (boolean): use checkpoint or not. Default True.
    
    Examples:
        >>> transformer_block = TransformerBlock(...)
        >>> checkpoint_block = BMTBlock(transformer_block)
        >>> y1, ... = checkpoint_block(x)
        >>> y2, ... = transformer_block(x)
        >>> assert torch.allclose(y1, y2)
    """
    def __init__(self, inner_module : torch.nn.Module, use_checkpoint=True):
        super().__init__()
        self._module = inner_module
        self._inputs = None
        self._layer_dict = {}
        self._backward_block_ctx = None
        # build large parameter&grad here
        self._param_info = []
        self._storage_params : Dict[str, torch.nn.Parameter] = {}
        self._storage_info = {}
        self._ready = False
        # sort parameters by name
        ordered_parameters = list(self._module.named_parameters())

        # calc total number of parameters
        for name, param in ordered_parameters:
            if not isinstance(param, DistributedParameter):
                raise ValueError("All parameters in checkpoint block must be DistributedParameter.")

            storage_type = storage_type_cuda(param.storage_type())
            kw_name = _get_param_kw(param)

            if kw_name not in self._storage_info:
                self._storage_info[kw_name] = {
                    "total": 0,
                    "storage_type": storage_type,
                    "requires_grad": param.requires_grad,
                    "group": param.group
                }

            param_shape = param._original_shape

            self._storage_info[kw_name]["total"] = round_up(
                self._storage_info[kw_name]["total"] + param_shape.numel(), 
                512 // param.element_size()
                # 512 bytes aligned
            )

        offsets = {}
        # intialize storage buffers
        for kw, val in self._storage_info.items():
            val["world_size"] = config["world_size"]
            partition_size = round_up(val["total"], val["world_size"]) // val["world_size"]
            val["partition_size"] = partition_size
            val["begin"] = config['rank'] * partition_size
            val["end"] = (config['rank'] + 1) * partition_size
            offsets[kw] = 0


            storage_type = val["storage_type"]

            storage_param_buffer = storage_type(partition_size)

            dtype = storage_param_buffer.dtype
            device = storage_param_buffer.device

            # bind storage to buffer tensor
            storage_param = torch.nn.Parameter(
                torch.tensor([], dtype=dtype, device=device).set_(storage_param_buffer)
            )
            if val["requires_grad"]:
                storage_param.requires_grad_(True)
            else:
                storage_param.requires_grad_(False)

        
            self._storage_params[kw] = storage_param

        # initialize parameters in module
        for name, param in ordered_parameters:
            param_shape = param._original_shape
            kw_name = _get_param_kw(param)

            param_st = offsets[kw_name]
            offsets[kw_name] += param_shape.numel()
            param_end = offsets[kw_name]
            offsets[kw_name] = round_up(offsets[kw_name], 512 // param.element_size())

            self._param_info.append({
                "parameter": param,
                "name": name,
                "offset": param_st,
                "size": param_shape.numel(),
                "shape": param_shape,
                "kw_name": kw_name,
            })

            # copy values to buffer for normal parameter
            storage_st = self._storage_info[kw_name]["begin"]
            storage_end = self._storage_info[kw_name]["end"]
            
            # make parameter contiguous in storage
            with torch.no_grad():
                contiguous_param = OpAllGather.apply(param)

            if not (param_st >= storage_end or param_end <= storage_st):
                # copy offset in parameter storage
                offset_st = max(storage_st - param_st, 0)
                offset_end = min(storage_end - param_st, contiguous_param.numel())
                assert offset_st < offset_end

                # copy to offset in buffer storage
                to_offset_st = offset_st + param_st - storage_st
                to_offset_end = offset_end + param_st - storage_st
                
                # copy to buffer
                # PyTorch 1.11 changed the API of storage.__getitem__
                d_dtype = self._storage_params[kw_name].dtype
                d_device = self._storage_params[kw_name].device
                param.data = torch.tensor([], dtype=param.dtype, device=param.device).set_(self._storage_params[kw_name].storage(), to_offset_st, (to_offset_end - to_offset_st,))
                self._param_info[-1]["begin"] = to_offset_st
                self._param_info[-1]["end"] = (to_offset_end - to_offset_st,)
                param.data[:] = \
                    torch.tensor([], dtype=d_dtype, device=d_device).set_(contiguous_param.storage(), offset_st, (offset_end - offset_st,))[:]
                del contiguous_param
            else:
                param.data = torch.tensor([], dtype=param.dtype, device=param.device)
            # clear parameter data, but keep the dtype and device
            setattr(param, "_in_checkpoint_block", True)

        for kw in offsets.keys():
            assert offsets[kw] == self._storage_info[kw]["total"]

        self.use_checkpoint = use_checkpoint
        self._is_first_layer = True
        self._is_last_layer = True
        self._pre_module = None
        self._next_module = None
        self._mode = "BLOCK" #BLOCK or ZERO or PIPE
        self.return_hidden_states = False
        self.hidden_states = []

    def set_pre_module(self, module):
        self._pre_module = module
        module._next_module = self
    
    def forward(self, *args): 
        #input must be requires_grad, otherwise autograd.backward will make an error
        args[0].requires_grad_()
        pre_out = hook_func.PreHookFunc.apply(self, *args)
        if self.use_checkpoint:
            out = checkpoint(self._module, *pre_out)
        else:
            out = self._module(*pre_out)
        tuple_out = (out, ) if isinstance(out, torch.Tensor) else out
        post_out = hook_func.PostHookFunc.apply(self, *tuple_out)
        if isinstance(out, torch.Tensor) and isinstance(post_out, tuple):
            return post_out[0]

        if isinstance(post_out, list):
            return tuple(post_out)
        return post_out

    def __getattr__(self,name:str):
        if name=="_module":
            return self._module
        return getattr(self._module, name)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __getattribute__(self, name: str):
        if name=="_parameters":
            return self._module._parameters
        return super().__getattribute__(name)

    def __delattr__(self, name):
        object.__delattr__(self, name)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        raise RuntimeError("._save_to_state_dict() of BMTBlock should not be called")
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # gather here
        with torch.no_grad():
            with CheckpointBlockContext(self):
                return self._module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        all_keys = []
        for it in self._param_info:
            key = prefix + it["name"]
            all_keys.append(key)
            if key in state_dict:
                # load here
                input_param = state_dict[key]
                if input_param.__class__.__name__ == "DistributedTensorWrapper":
                    input_param = input_param.broadcast()
                if input_param.shape != it["shape"]:
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, it["shape"]))
                    continue
                param_st = it["offset"]
                param_end = it["offset"] + it["size"]
                kw_name = it["kw_name"]

                # not in this partition
                storage_st = self._storage_info[kw_name]["begin"]
                storage_end = self._storage_info[kw_name]["end"]
                if param_st >= storage_end:
                    continue
                if param_end <= storage_st:
                    continue
                    
                # copy to buffer
                assert input_param.numel() == it["size"]
                contiguous_param = input_param.to(it["parameter"].dtype).cuda().contiguous()
                
                offset_st = max(storage_st - param_st, 0)
                offset_end = min(storage_end - param_st, contiguous_param.numel())
                assert offset_st < offset_end

                to_offset_st = offset_st + param_st - storage_st
                to_offset_end = offset_end + param_st - storage_st
                
                # copy to buffer
                # PyTorch 1.11 changed the API of storage.__getitem__
                d_dtype = self._storage_params[kw_name].dtype
                d_device = self._storage_params[kw_name].device
                torch.tensor([], dtype=d_dtype, device=d_device).set_(self._storage_params[kw_name].storage(), to_offset_st, (to_offset_end - to_offset_st,))[:] = \
                    torch.tensor([], dtype=d_dtype, device=d_device).set_(contiguous_param.storage(), offset_st, (offset_end - offset_st,))[:]
                del contiguous_param
            elif strict:
                missing_keys.append(key)

        for name, param in self.named_parameters():
            if isinstance(param, DistributedParameter) and not param._in_checkpoint_block:
                key = prefix + name
                all_keys.append(key)
                if key in state_dict:
                    input_param = state_dict[key]
                    is_param_lazy = torch.nn.parameter.is_lazy(param)
                    # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                    if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                        input_param = input_param[0]

                    if not is_param_lazy and not isinstance(param, DistributedParameter) and input_param.shape != param.shape:
                        # local shape should match the one in checkpoint
                        error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                        'the shape in current model is {}.'
                                        .format(key, input_param.shape, param.shape))
                        continue
                    if not is_param_lazy and isinstance(param, DistributedParameter) and input_param.shape != param._original_shape:
                        error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                        'the shape in current model is {}.'
                                        .format(key, input_param.shape, param.shape))
                    try:
                        with torch.no_grad():
                            param._copy_data(input_param)
                    except Exception as ex:
                        error_msgs.append('While copying the parameter named "{}", '
                                        'whose dimensions in the model are {} and '
                                        'whose dimensions in the checkpoint are {}, '
                                        'an exception occurred : {}.'
                                        .format(key, param.size(), input_param.size(), ex.args))
                elif strict:
                    missing_keys.append(key)

        if strict:
            all_keys = set(all_keys)
            for key in state_dict.keys():
                if key.startswith(prefix) and key not in all_keys:
                    unexpected_keys.append(key)
        
    def grouped_parameters(self):
        ret = {}
        for kw, val in self._storage_info.items():
            if val["group"] not in ret:
                ret[val["group"]] = []
            ret[val["group"]].append(self._storage_params[kw])
        for kw, val in ret.items():
            yield kw, val

    def init_parameters(self):
        """
        Initialize distributed parameters in this block.
        """
        for it in self._param_info:
            param = it["parameter"]
            if isinstance(param, DistributedParameter) and param._init_method is not None:
                # initialzie here
                tmp_tensor = torch.empty(it["shape"], device=param.device, dtype=param.dtype)
                param._init_method(tmp_tensor)
                param_st = it["offset"]
                param_end = it["offset"] + it["size"]
                kw_name = it["kw_name"]

                # not in this partition
                storage_st = self._storage_info[kw_name]["begin"]
                storage_end = self._storage_info[kw_name]["end"]
                if param_st >= storage_end:
                    continue
                if param_end <= storage_st:
                    continue
                    
                # copy to buffer
                assert tmp_tensor.is_contiguous() and it["size"] == tmp_tensor.numel()
                
                offset_st = max(storage_st - param_st, 0)
                offset_end = min(storage_end - param_st, tmp_tensor.numel())
                assert offset_st < offset_end

                to_offset_st = offset_st + param_st - storage_st
                to_offset_end = offset_end + param_st - storage_st
                
                # copy to buffer
                # PyTorch 1.11 changed the API of storage.__getitem__
                d_dtype = self._storage_params[kw_name].dtype
                d_device = self._storage_params[kw_name].device
                param.data[:] = \
                    torch.tensor([], dtype=d_dtype, device=d_device).set_(tmp_tensor.storage(), offset_st, (offset_end - offset_st,))[:]
                del tmp_tensor

        
    def _named_members(self, get_members_fn, prefix='', recurse=True, **kwargs):
        r"""Helper method for yielding various names + members of modules."""
        
        #compitibity with torch 2.0
        if "remove_duplicate" in inspect.signature(torch.nn.Module._named_members).parameters and "remove_duplicate" not in kwargs:
            kwargs['remove_duplicate'] = True
        return self._module._named_members(get_members_fn, prefix, recurse, **kwargs)
    
    def named_modules(self, memo = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
            or not

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """

        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._module._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def named_children(self):
        return self._module.named_children()
    
    def train(self, mode: bool = True):
        self._module.train(mode)

    def eval(self):
        self._module.eval()
    
    def __repr__(self):
        return self._module.__repr__()
        
class TransformerBlockList(torch.nn.Module):
    r"""
    TransformerBlockList is a list of BMTBlocks.

    This is designed to reduce the communication overhead by overlapping the computation and reduce_scatter operation during backward pass.

    It is similar to `torch.nn.ModuleList` but with the difference when calling .forward() and .backward().

    Example:
        >>> module_list = [ ... ]
        >>> normal_module_list = torch.nn.ModuleList(module_list)
        >>> transformer_module_list = TransformerBlockList(module_list)
        >>> # Calling normal module list
        >>> for layer in normal_module_list:
        >>>     hidden_state = layer.forward(hidden_state, ...)
        >>> # Calling transformer module list
        >>> hidden_state = transformer_module_list(hidden_state, ...)

    """
    _modules: Dict[str, BMTBlock]

    def __init__(self, modules: Iterable[BMTBlock], num_hidden=1, sqrt=False) -> None:
        super().__init__()
        
        self._modules = {}
        for i, module in enumerate(modules):
            if not isinstance(module, BMTBlock):
                module = BMTBlock(module)

            module._mode = "ZERO"
            module._is_last_layer = True if i == len(modules) -1 else False
            module._is_first_layer = True if i == 0 else False

            self._modules[str(i)] = module
            self.add_module(str(i), module)
            if i > 0:
                self._modules[str(i)].set_pre_module(self._modules[str(i-1)])
    
        self.num_hidden = num_hidden

        if sqrt:
            length = len(self)
            num_save_needed = 0
            num_freed = 0
            save_list = [None]*length
            for i in range(length-1, -1, -1):
                if num_freed == 0 or i == 0:
                    num_save_needed += 1
                    save_list[i] = [1, -num_save_needed]
                    num_freed = num_save_needed
                else:
                    num_freed -= 1
                    save_list[i] = [0, -(num_save_needed - num_freed)]
            for i in range(length-1, -1, -1):
                save_list[i][1] += num_save_needed
            for i in range(0, length):
                save_list[i][0] = i if save_list[i][0]==1 else save_list[i-1][0]

            self.save_list = save_list
        else:
            self.save_list = [(i, i) for i in range(len(self))]
            
    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[BMTBlock]:
        return iter(self._modules.values())
    def __getitem__(self, index: Union[int, str]) -> BMTBlock:
        return self._modules[str(index)]

    def forward(self, *args, return_hidden_states = False):
        self.return_hidden_states = return_hidden_states
        hidden_states = []
        for i in range(len(self)):
            if return_hidden_states:
                self._modules[str(i)].return_hidden_states = return_hidden_states
                self._modules[str(i)].hidden_states = hidden_states
            outputs = self._modules[str(i)]._call_impl(*args)
            if not isinstance(outputs, tuple):
                outputs = (outputs, )
            args = outputs + args[self.num_hidden:]

        if return_hidden_states:
            hidden_states = [
                torch.stack(hidden_states[i::self.num_hidden], dim=0)
                for i in range(self.num_hidden)
            ]

        if return_hidden_states:
            return outputs + tuple(hidden_states[:self.num_hidden])
        else:
            return tuple(outputs[:self.num_hidden]) if self.num_hidden > 1 else outputs[0]
