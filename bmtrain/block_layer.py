from typing import Dict, Iterable

from .global_var import config
import torch
from . import nccl
from .synchronize import wait_loader
from .parameter import DistributedParameter, OpAllGather
from .checkpointing import ScopedTensorInspectorContext
from . import debug

def round_up(x, d):
    return (x + d - 1) // d * d

class OpCheckpointBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block : 'CheckpointBlock', preserve_rng_state, *args):
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
        ctx.save_for_backward(*tensors)

        with torch.no_grad(), ScopedTensorInspectorContext() as inspector, CheckpointBlockContext(block):
            outputs = ctx.block._module._call_impl(*args)
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
                outputs = ctx.block._module._call_impl(*all_inputs)
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
        return (None, None) + tuple(grads)

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

                storage_type = self.block._storage_params[kw].storage_type()

                self._param_buffer[kw] = storage_type(val["total"])
                self._param_tensor[kw] = torch.tensor([], dtype=self._param_buffer[kw].dtype, device=self._param_buffer[kw].device).set_(self._param_buffer[kw])

                if requires_grad and val["requires_grad"]:
                    self._grad_buffer[kw] = storage_type(val["total"])
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
                param["parameter"].grad = torch.tensor([], dtype=dtype, device=device).set_(self._grad_buffer[kw_name], offset, shape)
    
    
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
                self._grad_tensor[kw][val["begin"]:val["end"]] += local_param.grad
            
            current_stream = torch.cuda.current_stream()
            config["load_stream"].wait_stream(current_stream)   # wait for backward

            with torch.cuda.stream(config["load_stream"]):
                nccl.groupStart()
                for kw, val in self.block._storage_info.items():
                    local_param = self.block._storage_params[kw]

                    # scatter gradient
                    if val["requires_grad"]:
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
            if val["requires_grad"]:
                storage_grads_buffer = storage_type(partition_size)

            dtype = storage_param_buffer.dtype
            device = storage_param_buffer.device

            # bind storage to buffer tensor
            storage_param = torch.nn.Parameter(
                torch.tensor([], dtype=dtype, device=device).set_(storage_param_buffer)
            )
            if val["requires_grad"]:
                storage_param.grad = torch.tensor([], dtype=dtype, device=device).set_(storage_grads_buffer).zero_()

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
                    self._storage_params[kw_name].storage()[to_offset_st: to_offset_end].copy_(contiguous_param.storage()[offset_st: offset_end])
                    del contiguous_param
            
            # clear parameter data, but keep the dtype and device
            param.data = torch.tensor([], dtype=param.dtype, device=param.device)
            setattr(param, "_in_checkpoint_block", True)

        for kw in offsets.keys():
            assert offsets[kw] == self._storage_info[kw]["total"]
        
    def __call__(self, *args):
        # gather here
        return OpCheckpointBlock.apply(self, True, *args)
    
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
                self._storage_params[kw_name].storage()[to_offset_st: to_offset_end].copy_(contiguous_param.storage()[offset_st: offset_end])
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
                self._storage_params[kw_name].storage()[to_offset_st: to_offset_end].copy_(tmp_tensor.storage()[offset_st: offset_end])
                del tmp_tensor
        
class OpTransformerBlockList(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self : 'TransformerBlockList', hidden_state, *args):
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
        
        layer_inputs = []
        layer_inspector = []
        cuda_rng_state = []
        with torch.no_grad():
            for i in range(len(self)):
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
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        all_inputs = []
        input_requires_grad = []
        
        layer_inputs = ctx.saved_tensors[:len(ctx.self)]
        save_args = ctx.saved_tensors[len(ctx.self):]
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
                for i in list(range(len(ctx.self)))[::-1]:
                    torch.cuda.set_rng_state(ctx.cuda_rng_state[i])
                    ipt = layer_inputs[i].detach().requires_grad_()
                    block_ctx = CheckpointBlockContext(ctx.self._modules[str(i)])
                    block_ctx.enter()
                    if prev_ctx is not None:
                        prev_ctx.exit()
                        config["load_stream"].record_event(config["load_event"])
                    prev_ctx = block_ctx

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
                
                if prev_ctx is not None:
                    prev_ctx.exit()
                    config["load_stream"].record_event(config["load_event"])

        grads = []
        for inp, requires_grad in zip(all_inputs, input_requires_grad):
            if requires_grad:
                grads.append(inp.grad)
            else:
                grads.append(None)
        return (None, grad_hidden_state) + tuple(grads)
    
class TransformerBlockList(torch.nn.Module):
    _modules: Dict[str, CheckpointBlock]

    def __init__(self, modules: Iterable[CheckpointBlock]) -> None:
        super().__init__()
        
        self._modules = {}
        for i, module in enumerate(modules):
            if not isinstance(module, CheckpointBlock):
                module = CheckpointBlock(module)
            self._modules[str(i)] = module
            self.add_module(str(i), module)

    def __len__(self) -> int:
        return len(self._modules)

    def forward(self, hidden_state, *args):
        return OpTransformerBlockList.apply(self, hidden_state, *args)
