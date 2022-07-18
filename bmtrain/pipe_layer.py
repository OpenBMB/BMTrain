from bmtrain import layer
import pip
import torch
import copy
from typing import Dict, Iterable, Iterator, Tuple, Union, List
from .store import allgather_object
from torch.autograd import backward
from .pipe_comm import recv_activations,send_activations,broadcast,gather_input
from .global_var import config
import torch
from . import nccl
from .synchronize import wait_loader
from .parameter import DistributedParameter, OpAllGather
from .checkpointing import ScopedTensorInspectorContext
from . import debug
from  torch.nn.modules.module import _addindent
import copy
from .pipe_comm import gather_input
from .block_layer import CheckpointBlockContext,CheckpointBlock,round_up,_get_param_kw
class OpPipeTransformerBlockList(torch.autograd.Function):
    @staticmethod
    def forward(ctx, placeholder, self : 'PipelineTransformerBlockList', save_list, hidden_state, *args):
        args = list(args)
        with PipeContext(self, hidden_state, args) as pipe_input:
            hidden_state = pipe_input[0]
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
            ctx.layers_dict=[{} for _ in range(len(self))]
            layer_inputs = []
            layer_inspector = []
            cuda_rng_state = []
            with torch.no_grad():
                for idx,layer_id in enumerate(self.layer_ids):
                    if save_list[idx][0] == idx:
                        layer_inputs.append(hidden_state)
                    cuda_rng_state.append( torch.cuda.get_rng_state() )
                    #TODO  for micro batch in pipe,the checkpoint activation needs to save until all micro batch are finished when forward
                    block_ctx = CheckpointBlockContext(self._modules[str(layer_id)], ctx.layers_dict[idx], 1, pipe=True)
                    # gather parameter on load stream
                    block_ctx.enter()
                    # call inner module directly
                    with ScopedTensorInspectorContext() as inspector:
                        hidden_state = self._modules[str(layer_id)]._module._call_impl(hidden_state, *args)
                    for it in inspector.hidden_states:
                        debug.append("_inspect_hidden_states", it)
                    layer_inspector.append(inspector.hidden_states)
                    block_ctx.exit()
            
            ctx.layer_inspector = layer_inspector
            ctx.cuda_rng_state = cuda_rng_state

            
            ctx.save_for_backward(*layer_inputs, *tensors)
            pipe_input[0] = hidden_state
        return pipe_input[0]


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
        with PipeContext(ctx.self, grad_hidden_state, backward=True) as pipe_input:
            grad_hidden_state = pipe_input[0]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=True):
                with torch.enable_grad():
                    # overlap load and scatter here
                    prev_ctx = None
                    prev_grad = False
                    for idx,layer_id in enumerate(ctx.self.layer_ids[::-1]):
                        torch.cuda.set_rng_state(ctx.cuda_rng_state[idx])
                        ipt = layer_inputs[ctx.save_list[idx][1]].detach().requires_grad_()
                        block_ctx = CheckpointBlockContext(ctx.self._modules[str(layer_id)], ctx.layers_dict[idx], 2, pipe=True)
                        block_ctx.enter()
                        exit_prev(prev_ctx, prev_grad)
                        prev_ctx = block_ctx
                        prev_grad = True

                        with ScopedTensorInspectorContext() as inspector:
                            output = ctx.self._modules[str(layer_id)]._module._call_impl(ipt, *all_inputs)

                        assert len(ctx.layer_inspector[idx]) == len(inspector.hidden_states), "Backward step changed"
                        for j, it in enumerate(inspector.hidden_states):
                            assert it["name"] == ctx.layer_inspector[idx][j]["name"], "Backward step changed"
                            assert it["shape"] == ctx.layer_inspector[idx][j]["shape"], "Backward step changed"
                            assert it["group"] == ctx.layer_inspector[idx][j]["group"], "Backward step changed"
                            
                            # change the tensor in placeholder
                            ctx.layer_inspector[idx][j]["tensor"] = it["tensor"]
                        torch.autograd.backward(
                            [output],
                            [grad_hidden_state]
                        )
                        grad_hidden_state = ipt.grad
                    exit_prev(prev_ctx, prev_grad)

            pipe_input[0] = grad_hidden_state
        grads = []
        for inp, requires_grad in zip(all_inputs, input_requires_grad):
            if requires_grad:
                grads.append(inp.grad)
            else:
                grads.append(None)
        return (None, None, None, pipe_input[0]) + tuple(grads)
    
class PipelineTransformerBlockList(torch.nn.Module):
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
        rank = config['rank']
        topo = config['topology']
        self.layer_ids = []
        pipe_group = topo.pp_group
        self.stages = topo.stages
        self.stage_id = topo.stage_id
        self.pipe_idx = topo.pipe_idx 
        self.layer_ids = self.partition_modules(modules)
        self.next_rank = pipe_group[self.pipe_idx, self.stage_id + 1] if self.stage_id < config['pipe_size'] - 1 else -1
        self.prev_rank = pipe_group[self.pipe_idx, self.stage_id - 1] if self.stage_id > 0 else -1
        # self.micro_batches = config['num_micro_batches']
            
        for idx,module in enumerate(modules):
            if not isinstance(module, CheckpointBlock):
                module = CheckpointBlock(module)
            self._modules[str(idx)] = module
        self.save_list = [(i, i) for i in range(len(self.layer_ids))]
            
    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: Union[int, str]) -> CheckpointBlock:
        return self._modules[str(index)]

    def forward(self, hidden_state, *args):
        placeholder = torch.tensor([], requires_grad=torch.is_grad_enabled())
        hidden_state = OpPipeTransformerBlockList.apply(placeholder, self, self.save_list, hidden_state, *args)
        return hidden_state
    def partition_modules(self, modules: Iterable[CheckpointBlock])->  List[int]:
        if not isinstance(modules, list):
            modules = list(modules)
        part_len = round_up(len(modules),self.stages) // self.stages
        start = self.stage_id * part_len
        end = min((self.stage_id + 1) * part_len, len(modules))
        idxs = range(start, end)
        for i in range(len(modules)):
            if i not in idxs:
                for name,param in modules[i]._module.named_parameters():
                    param.data = torch.tensor([], dtype = param.dtype, device = param.device)
                for kw, val in modules[i]._storage_info.items():
                    val["begin"] = self.stage_id
                    val["end"] = self.stage_id + 1
                    val["partition_size"] = 1
                    val["total"] = val["world_size"]
                    dtype = modules[i]._storage_params[kw].dtype
                    device = modules[i]._storage_params[kw].device
                    modules[i]._storage_params[kw] = \
                        torch.nn.Parameter(torch.tensor([0], dtype = dtype, device=device))
            else:
                contiguous_params = {}
                for kw, val in modules[i]._storage_info.items():
                    storage_type = val["storage_type"]
                    contiguous_params[kw] = storage_type(val["partition_size"]*val["world_size"])
                    val["world_size"] = config["world_size"] // config["pipe_size"]
                    partition_size = round_up(val["total"], val["world_size"]) // val["world_size"]
                    val["partition_size"] = partition_size
                    val["begin"] = config['zero_rank'] * partition_size
                    val["end"] = (config['zero_rank'] + 1) * partition_size
                    assert config["world_size"] % config["pipe_size"] == 0, "world_size needs to be divisible by pipe_size"
                    nccl.allGather(
                        modules[i]._storage_params[kw].storage(),
                        contiguous_params[kw],
                        config["comm"]
                    )
                    storage_param_buffer = storage_type(partition_size)
                    dtype = storage_param_buffer.dtype
                    device = storage_param_buffer.device
                    modules[i]._storage_params[kw] = torch.nn.Parameter(
                        torch.tensor([], dtype=dtype, device=device).set_(storage_param_buffer)
                    )
                    if val["requires_grad"]:
                        modules[i]._storage_params[kw].requires_grad_(True)
                    else:
                        modules[i]._storage_params[kw].requires_grad_(False)
                ordered_parameters = list(modules[i]._module.named_parameters())
                for idx,named_param in enumerate(ordered_parameters):
                    name,param = named_param
                    param_info = modules[i]._param_info[idx]
                    kw_name = _get_param_kw(param)
                    storage_info = modules[i]._storage_info[kw_name]
                    storage_st = storage_info["begin"]
                    storage_end = storage_info["end"]
                    param_st = param_info["offset"]
                    param_end = param_st + param_info["size"]
                    storage = contiguous_params[kw_name]
                    contiguous_param = torch.tensor([],dtype = storage.dtype, device=storage.device).set_(storage,param_info["offset"],param_info["shape"])
                    if not (param_st >= storage_end or param_end <= storage_st):
                        # copy offset in parameter storage
                        offset_st = max(storage_st - param_st, 0)
                        offset_end = min(storage_end - param_st, contiguous_param.numel())
                        assert offset_st < offset_end
                        to_offset_st = offset_st + param_st - storage_st
                        to_offset_end = offset_end + param_st - storage_st
                        d_dtype = modules[i]._storage_params[kw_name].dtype
                        d_device = modules[i]._storage_params[kw_name].device
                        param.data = torch.tensor([], dtype=param.dtype, device=param.device).set_(modules[i]._storage_params[kw_name].storage(), to_offset_st, (to_offset_end - to_offset_st,))
                        param_info["begin"] = to_offset_st
                        param_info["end"] = (to_offset_end - to_offset_st,)
                        param.data[:] = \
                            torch.tensor([], dtype=d_dtype, device=d_device).set_(contiguous_param.storage(), offset_st, (offset_end - offset_st,))[:]
                    else:
                        param.data = torch.tensor([], dtype=param.dtype, device=param.device)
                del contiguous_param
        return idxs
    def __init__(self, module, hidden_state, args, backward=False):
        self.module = module
        self.stage_id = module.stage_id
        self.stages = module.stages
        self.next_rank = module.next_rank
        self.prev_rank = module.prev_rank
        self.hidden_state = [hidden_state]
        self.backward = backward
        self.send_buffer = {}
        self.args = args
    def enter(self):
        if self.backward:
            if self.stage_id != self.stages -1:
                self.hidden_state[0] = recv_activations(self.stage_id + 1, config['pipe_comm'])
        else:
            if self.stage_id != 0:
                self.hidden_state[0] = recv_activations(self.stage_id - 1, config['pipe_comm'])
        return self.hidden_state
    def exit(self):
        if self.backward:
            if self.stage_id != 0:
                send_activations(self.hidden_state[0], self.stage_id - 1, config['pipe_comm'])
        else:
            if self.stage_id != self.stages - 1:
                send_activations(self.hidden_state[0], self.stage_id + 1, config['pipe_comm'])
            broadcast(self.hidden_state[0], self.stages - 1, config['pipe_comm'], inplace=True)
    def __enter__(self):
        return self.enter()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()