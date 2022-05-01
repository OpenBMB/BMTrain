from typing import Dict, Iterable, Iterator, Tuple, Union

from .global_var import config
import torch
from . import nccl
from .synchronize import wait_loader
from .parameter import DistributedParameter, OpAllGather
from .checkpointing import ScopedTensorInspectorContext
from . import debug
from  torch.nn.modules.module import _addindent
import copy

def round_up(x, d):
    return (x + d - 1) // d * d

class OpCheckpointBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, placeholder, block : 'CheckpointBlock', preserve_rng_state, len_args, *args):
        ctx.block = block
        ctx.preserve_rng_state = preserve_rng_state
        
        ctx.cuda_rng_state = torch.cuda.get_rng_state() if preserve_rng_state else None
        
        tensors = []
        others = []
        for arg in args:
            if torch.is_tensor(arg):
                tensors.append(arg)
                others.append(None)
            else:
                tensors.append(None)
                others.append(arg)

        ctx.nontensor_inputs = others
        ctx.len_args = len_args
        ctx.save_for_backward(*tensors)

        with torch.no_grad(), ScopedTensorInspectorContext() as inspector, CheckpointBlockContext(block):
            inp_args = args[:len_args]
            inp_kwargs = {}
            for k, v in zip(args[len_args::2], args[len_args + 1::2]):
                inp_kwargs[k] = v
            outputs = ctx.block._module._call_impl(*inp_args, **inp_kwargs)
        for it in inspector.hidden_states:
            debug.append("_inspect_hidden_states", it)
        ctx.inspect_list = inspector.hidden_states
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        all_inputs = []
        input_reqires_grad = []
        len_args = ctx.len_args
        for tensor, other in zip(ctx.saved_tensors, ctx.nontensor_inputs):
            if tensor is None:
                all_inputs.append(other)
                input_reqires_grad.append(False)
            else:
                input_reqires_grad.append( tensor.requires_grad )
                nw_tensor = tensor.detach()
                nw_tensor.requires_grad = tensor.requires_grad
                all_inputs.append(nw_tensor)

        
        with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.cuda.set_rng_state(ctx.cuda_rng_state)
            with torch.enable_grad(), ScopedTensorInspectorContext() as inspector, CheckpointBlockContext(ctx.block):
                inp_args = all_inputs[:len_args]
                inp_kwargs = {}
                for k, v in zip(all_inputs[len_args::2], all_inputs[len_args + 1::2]):
                    inp_kwargs[k] = v
                outputs = ctx.block._module._call_impl(*inp_args, **inp_kwargs)
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
    
                assert len(outputs) == len(grad_outputs)

                outputs_with_grad = []
                grad_of_output = []
                for i, output in enumerate(outputs):
                    if torch.is_tensor(output) and output.requires_grad:
                        outputs_with_grad.append(output)
                        grad_of_output.append(grad_outputs[i])

                # calculate gradients for inputs, also for parameters
                torch.autograd.backward(
                    outputs_with_grad,
                    grad_of_output,
                )
            assert len(ctx.inspect_list) == len(inspector.hidden_states), "Backward step changed"
            for i, it in enumerate(inspector.hidden_states):
                assert it["name"] == ctx.inspect_list[i]["name"], "Backward step changed"
                assert it["shape"] == ctx.inspect_list[i]["shape"], "Backward step changed"
                assert it["group"] == ctx.inspect_list[i]["group"], "Backward step changed"
                
                # change the tensor in placeholder
                ctx.inspect_list[i]["tensor"] = it["tensor"]

        grads = []
        for inp, requires_grad in zip(all_inputs, input_reqires_grad):
            if requires_grad:
                grads.append(inp.grad)
            else:
                grads.append(None)
        return (None, None, None, None) + tuple(grads)

class CheckpointBlockContext:
    def __init__(self, block : 'CheckpointBlock') -> None:
        self.block = block
        self._param_buffer = {}
        self._grad_buffer = {}
        self._param_tensor = {}
        self._grad_tensor = {}

        self._need_release = False
    
    def enter(self):
        """
        gather parameters
        """
        if self.block._ready:
            return
        self.block._ready = True
        self._need_release = True

        wait_loader()
        requires_grad = torch.is_grad_enabled()

        with torch.cuda.stream(config["load_stream"]):
            for kw, val in self.block._storage_info.items():
                assert self.block._storage_params[kw].is_cuda
                assert kw not in self._grad_buffer
                assert kw not in self._param_buffer
                local_param = self.block._storage_params[kw]

                storage_type = local_param.storage_type()

                self._param_buffer[kw] = storage_type(val["partition_size"] * config["world_size"])
                self._param_tensor[kw] = torch.tensor([], dtype=self._param_buffer[kw].dtype, device=self._param_buffer[kw].device).set_(self._param_buffer[kw])

                if requires_grad and local_param.requires_grad:
                    self._grad_buffer[kw] = storage_type(val["partition_size"] * config["world_size"])
                    self._grad_tensor[kw] = torch.tensor([], dtype=self._grad_buffer[kw].dtype, device=self._grad_buffer[kw].device).set_(self._grad_buffer[kw]).zero_()

            nccl.groupStart()
            for kw, val in self.block._storage_info.items():
                nccl.allGather(
                    self.block._storage_params[kw].storage(),
                    self._param_buffer[kw],
                    config["comm"]
                )
            nccl.groupEnd()

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(config["load_stream"])
        
        # set wait stream for each storage
        for kw in self._param_tensor.keys():
            self._param_tensor[kw].record_stream(current_stream)
            if requires_grad and kw in self._grad_tensor:
                self._grad_tensor[kw].record_stream(current_stream)

        # update parameters in block
        for param in self.block._param_info:
            kw_name = param["kw_name"]
            dtype = self._param_buffer[kw_name].dtype
            device = self._param_buffer[kw_name].device
            offset = param["offset"]
            shape = param["shape"]
            param["parameter"].data = torch.tensor([], dtype=dtype, device=device).set_(self._param_buffer[kw_name], offset, shape)
            if requires_grad and kw_name in self._grad_buffer:
                param["parameter"].requires_grad_(True)
                param["parameter"].grad = torch.tensor([], dtype=dtype, device=device).set_(self._grad_buffer[kw_name], offset, shape)
            else:
                param["parameter"].requires_grad_(False)
    
    
    def __enter__(self):
        self.enter()
    
    def exit(self):
        """
        Reduce scatter gradients
        """

        if not self._need_release:
            return
        self._need_release = False
        self.block._ready = False

        requires_grad = torch.is_grad_enabled()
        if requires_grad:
            for kw, val in self.block._storage_info.items():
                local_param = self.block._storage_params[kw]

                # accumulate previous gradient
                if local_param.requires_grad:
                    if local_param.grad is None:
                        grad_storage = val["storage_type"](val["partition_size"])   # initialize gradient if not exist
                        local_param.grad = torch.tensor([], dtype=grad_storage.dtype, device=grad_storage.device).set_(grad_storage).zero_()
                    else:
                        self._grad_tensor[kw][val["begin"]:val["end"]] += local_param.grad
            
            current_stream = torch.cuda.current_stream()
            config["load_stream"].wait_stream(current_stream)   # wait for backward

            with torch.cuda.stream(config["load_stream"]):
                nccl.groupStart()
                for kw, val in self.block._storage_info.items():
                    local_param = self.block._storage_params[kw]

                    # scatter gradient
                    if local_param.requires_grad:
                        nccl.reduceScatter(
                            self._grad_buffer[kw],
                            local_param.grad.storage(),
                            "sum",
                            config["comm"]
                        )
                nccl.groupEnd()

            # set wait stream for each storage
            for kw in self._grad_tensor.keys():
                # grads can not be freed until reduce ops finish
                self._grad_tensor[kw].record_stream(config["load_stream"])

        # Release all parameters in buffer
        for param in self.block._param_info:
            dtype = param["parameter"].dtype
            device = param["parameter"].device
            param["parameter"].data = torch.tensor([], dtype=dtype, device=device)
            param["parameter"].grad = None
        
        self._grad_tensor = {}
        self._param_tensor = {}
        self._grad_buffer = {}
        self._param_buffer = {}
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # reduce scatter gradients
        self.exit()

def storage_type_cuda(storage_type):
    STORAGE_MAP = {
        torch.FloatStorage: torch.cuda.FloatStorage,
        torch.DoubleStorage: torch.cuda.DoubleStorage,
        torch.HalfStorage: torch.cuda.HalfStorage,
        torch.CharStorage: torch.cuda.CharStorage,
        torch.ByteStorage: torch.cuda.ByteStorage,
        torch.ShortStorage: torch.cuda.ShortStorage,
        torch.IntStorage: torch.cuda.IntStorage,
        torch.cuda.FloatStorage: torch.cuda.FloatStorage,
        torch.cuda.DoubleStorage: torch.cuda.DoubleStorage,
        torch.cuda.HalfStorage: torch.cuda.HalfStorage,
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

class CheckpointBlock(torch.nn.Module):
    """
    CheckpointBlock is a leaf module that `inner_module` is invisible outside of CheckpointBlock.
    Only these methods can acces `inner_module`:
    - forward
    - load_state_dict
    - state_dict

    For other methods, it looks like a black box with several parameter.

    This is desinged to reduce the number of calls to the NCCL APIs by grouping parameters inside the inner_module.

    If you want to get the parameters inside the inner_module, you can use the state_dict method.

    """
    def __init__(self, inner_module : torch.nn.Module):
        super().__init__()

        self._module = inner_module
        # build large parameter&grad here
        self._param_info = []
        self._storage_params : Dict[str, torch.nn.Parameter] = {}
        self._storage_info = {}
        self._ready = False
        # sort parameters by name
        ordered_parameters = list(self._module.named_parameters())

        # calc total number of parameters
        for name, param in ordered_parameters:
            assert isinstance(param, DistributedParameter), "All parameters in checkpoint block must be DistributedParameter."

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
            partition_size = round_up(val["total"], config['world_size']) // config['world_size']
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

            # register parameter
            self.register_parameter(kw, storage_param)

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

            if isinstance(param, DistributedParameter) and param._init_method is not None:
                # do not copy distributed parameters
                pass
            else:
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
                    torch.tensor([], dtype=d_dtype, device=d_device).set_(self._storage_params[kw_name].storage(), to_offset_st, (to_offset_end - to_offset_st,))[:] = \
                        torch.tensor([], dtype=d_dtype, device=d_device).set_(contiguous_param.storage(), offset_st, (offset_end - offset_st,))[:]
                    # self._storage_params[kw_name].storage()[to_offset_st: to_offset_end].copy_(contiguous_param.storage()[offset_st: offset_end])
                    del contiguous_param
            
            # clear parameter data, but keep the dtype and device
            param.data = torch.tensor([], dtype=param.dtype, device=param.device)
            setattr(param, "_in_checkpoint_block", True)

        for kw in offsets.keys():
            assert offsets[kw] == self._storage_info[kw]["total"]
        
    def __call__(self, *args, **kwargs):
        # gather here
        placeholder = torch.tensor([], requires_grad=torch.is_grad_enabled())
        all_inputs = list(args)
        for kw, val in kwargs.items():
            all_inputs.append(kw)
            all_inputs.append(val)
        return OpCheckpointBlock.apply(placeholder, self, True, len(args), *all_inputs)
    
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
    
    def __delattr__(self, name):
        object.__delattr__(self, name)
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        raise RuntimeError("._save_to_state_dict() of CheckpointBlock should not be called")
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # gather here
        with torch.no_grad():
            with CheckpointBlockContext(self):
                return self._module.state_dict(destination, prefix, keep_vars)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        
        all_keys = []
        for it in self._param_info:
            key = prefix + it["name"]
            all_keys.append(key)
            if key in state_dict:
                # load here
                input_param = state_dict[key]
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
                contiguous_param = input_param.to(it["parameter"].dtype).contiguous()
                
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
                # self._storage_params[kw_name].storage()[to_offset_st: to_offset_end].copy_(contiguous_param.storage()[offset_st: offset_end])
                del contiguous_param
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
                torch.tensor([], dtype=d_dtype, device=d_device).set_(self._storage_params[kw_name].storage(), to_offset_st, (to_offset_end - to_offset_st,))[:] = \
                    torch.tensor([], dtype=d_dtype, device=d_device).set_(tmp_tensor.storage(), offset_st, (offset_end - offset_st,))[:]
                # self._storage_params[kw_name].storage()[to_offset_st: to_offset_end].copy_(tmp_tensor.storage()[offset_st: offset_end])
                del tmp_tensor
        
    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        print("here in _named_members")
        memo = set()
        modules = torch.nn.Module.named_modules(self, prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

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
        # print("here in named_modules hahaha")

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
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module

        Example::

            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._module._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module
    
    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self._module.train(mode)
        return self

    def eval(self):
        r"""Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. Dropout, BatchNorm,
        etc.

        This is equivalent with `self.train(False)`.

        Returns:
            Module: self
        """
        return self.train(False)

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._module._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
    
    def __getattr__(self, attribute):
        try:
            return super().__getattr__(attribute)
        except:
            return getattr(self._module, attribute)
    

        
class OpTransformerBlockList(torch.autograd.Function):
    @staticmethod
    def forward(ctx, placeholder, self : 'TransformerBlockList', save_list, hidden_state, *args):
        tensors = []
        others = []
        for arg in args:
            if torch.is_tensor(arg):
                tensors.append(arg)
                others.append(None)
            else:
                tensors.append(None)
                others.append(arg)
    
        ctx.nontensor_inputs = others
        ctx.self = self
        ctx.save_list = copy.deepcopy(save_list)
        ctx.num_save_needed = save_list[-1][1]+1

        layer_inputs = []
        layer_inspector = []
        cuda_rng_state = []
        with torch.no_grad():
            for i in range(len(self)):
                if save_list[i][0] == i:
                    layer_inputs.append(hidden_state)
                cuda_rng_state.append( torch.cuda.get_rng_state() )
                block_ctx = CheckpointBlockContext(self._modules[str(i)])
                # gather parameter on load stream
                block_ctx.enter()
                # call inner module directly
                with ScopedTensorInspectorContext() as inspector:
                    hidden_state = self._modules[str(i)]._module._call_impl(hidden_state, *args)
                for it in inspector.hidden_states:
                    debug.append("_inspect_hidden_states", it)
                layer_inspector.append(inspector.hidden_states)
                block_ctx.exit()
        
        ctx.layer_inspector = layer_inspector
        ctx.cuda_rng_state = cuda_rng_state

        
        ctx.save_for_backward(*layer_inputs, *tensors)
        return hidden_state


    @staticmethod
    def backward(ctx, grad_hidden_state : torch.Tensor):
        def exit_prev(prev_ctx, prev_grad):
            if prev_ctx is not None:
                if prev_grad:
                    with torch.enable_grad():
                        prev_ctx.exit()
                        config["load_stream"].record_event(config["load_event"])
                else:
                    with torch.no_grad():
                        prev_ctx.exit()
                        config["load_stream"].record_event(config["load_event"])
                
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        all_inputs = []
        input_requires_grad = []
        
        layer_inputs = ctx.saved_tensors[:ctx.num_save_needed]
        save_args = ctx.saved_tensors[ctx.num_save_needed:]
        for tensor, other in zip(save_args, ctx.nontensor_inputs):
            if tensor is None:
                all_inputs.append(other)
                input_requires_grad.append(False)
            else:
                # detach for tensor inputs
                input_requires_grad.append( tensor.requires_grad )
                nw_tensor = tensor.detach()
                nw_tensor.requires_grad = tensor.requires_grad
                all_inputs.append(nw_tensor)
        
        with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=True):
            with torch.enable_grad():
                # overlap load and scatter here
                prev_ctx = None
                prev_grad = False
                for i in reversed(range(len(ctx.self))):
                    if ctx.save_list[i][0] != i:
                        with torch.no_grad():
                            st = ctx.save_list[i][0]
                            for j in range(st, i):
                                torch.cuda.set_rng_state(ctx.cuda_rng_state[j])
                                block_ctx = CheckpointBlockContext(ctx.self._modules[str(j)])
                                block_ctx.enter()
                                exit_prev(prev_ctx, prev_grad)
                                output = ctx.self._modules[str(j)]._module._call_impl(layer_inputs[ctx.save_list[j][1]], *all_inputs)
                                prev_ctx = block_ctx
                                prev_grad = False
                                layer_inputs[ctx.save_list[j+1][1]].copy_(output)
                                ctx.save_list[j+1][0] = j+1
                    torch.cuda.set_rng_state(ctx.cuda_rng_state[i])
                    ipt = layer_inputs[ctx.save_list[i][1]].detach().requires_grad_()
                    block_ctx = CheckpointBlockContext(ctx.self._modules[str(i)])
                    block_ctx.enter()
                    exit_prev(prev_ctx, prev_grad)
                    prev_ctx = block_ctx
                    prev_grad = True

                    with ScopedTensorInspectorContext() as inspector:
                        output = ctx.self._modules[str(i)]._module._call_impl(ipt, *all_inputs)
                    
                    assert len(ctx.layer_inspector[i]) == len(inspector.hidden_states), "Backward step changed"
                    for j, it in enumerate(inspector.hidden_states):
                        assert it["name"] == ctx.layer_inspector[i][j]["name"], "Backward step changed"
                        assert it["shape"] == ctx.layer_inspector[i][j]["shape"], "Backward step changed"
                        assert it["group"] == ctx.layer_inspector[i][j]["group"], "Backward step changed"
                        
                        # change the tensor in placeholder
                        ctx.layer_inspector[i][j]["tensor"] = it["tensor"]
                    torch.autograd.backward(
                        [output],
                        [grad_hidden_state]
                    )
                    grad_hidden_state = ipt.grad
                
                exit_prev(prev_ctx, prev_grad)

        grads = []
        for inp, requires_grad in zip(all_inputs, input_requires_grad):
            if requires_grad:
                grads.append(inp.grad)
            else:
                grads.append(None)
        return (None, None, None, grad_hidden_state) + tuple(grads)
    
class TransformerBlockList(torch.nn.Module):
    r"""
    TransformerBlockList is a list of CheckpointBlocks.

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
    _modules: Dict[str, CheckpointBlock]

    def __init__(self, modules: Iterable[CheckpointBlock], sqrt=False) -> None:
        super().__init__()
        
        self._modules = {}
        for i, module in enumerate(modules):
            if not isinstance(module, CheckpointBlock):
                module = CheckpointBlock(module)
            self._modules[str(i)] = module
            self.add_module(str(i), module)

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
    
    def __getitem__(self, index: Union[int, str]) -> CheckpointBlock:
        return self._modules[str(index)]

    def forward(self, hidden_state, *args):
        placeholder = torch.tensor([], requires_grad=torch.is_grad_enabled())
        return OpTransformerBlockList.apply(placeholder, self, self.save_list, hidden_state, *args)
