from collections import OrderedDict
import copy
import torch
import copy
from typing import Dict, Iterable, Iterator, Tuple, Union, List
import torch

from .distributed import all_gather, broadcast, all_reduce, send_activations, recv_activations 
from .global_var import config
from . import nccl
from .checkpointing import (
        ScopedTensorInspectorContext,
        CheckpointBlockContext
)
from . import debug
from .block_layer import CheckpointBlock, round_up, _get_param_kw

class PipePreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_state, *args):
        hidden_state_list = all_gather(hidden_state.clone(), config["pipe_comm"])
        hidden_state_list.requires_grad_()

        batch_related = args[-1]
        batch_related_origin = [True if i in args[-1] else False for i in range(len(args[:-1]))]
        batch_related_rule = []
        args = args[:-1]

        batch_size = hidden_state.shape[0]
        num_micros = config["micros"]
        args_list = [[] for _ in range(num_micros)]
        input_requires_grad = []
        for arg in args:
            if torch.is_tensor(arg):
                arg_all = all_gather(arg, config['pipe_comm'])
                if arg.dim() == hidden_state.dim() and arg.shape[0] == batch_size:
                    batch_related_rule.append(True)
                    arg_all = arg_all.flatten(0, 1).chunk(num_micros, dim=0)
                    arg_all = [tensor.requires_grad_(arg.requires_grad) for tensor in arg_all]
                else:
                    batch_related_rule.append(False)
                    arg_all = [arg_all[0].requires_grad_(arg.requires_grad) for i in range(num_micros)]
                input_requires_grad.append(arg.requires_grad)
            else:
                batch_related_rule.append(False)
                arg_all = [arg for _ in range(num_micros)]
                input_requires_grad.append(False)
            for i in range(num_micros):
                args_list[i].append(arg_all[i])
        ctx.input_requires_grad = input_requires_grad
        ctx.args_list = args_list
        if len(batch_related) == 0:
            ctx.batch_related = batch_related_rule
        else:
            ctx.batch_related = batch_related_origin
        return hidden_state_list, args_list

    @staticmethod
    def backward(ctx, grads, arg_grads):
        grads = broadcast(grads, 0, config['pipe_comm'])
        topo = config['topology']
        arg_grads = []
        num_micros = config['micros']
        for idx,requires_grad in enumerate(ctx.input_requires_grad):
            if requires_grad:
                grad = torch.cat([ctx.args_list[m][idx].grad for m in range(num_micros)], dim=0)
                grad = all_reduce(grad, "sum", config["pipe_comm"])
                split_size = topo.stages if ctx.batch_related[idx] else num_micros
                grad = grad.chunk(split_size)
                if ctx.batch_related[idx]:
                    arg_grads.append(grad[topo.stage_id])
                else:
                    arg_grads.append(grad[0])
            else:
                arg_grads.append(None)
        arg_grads.append(None) #for append(batch_related)
        return grads.chunk(topo.stages, dim=0)[topo.stage_id], *arg_grads

class PipePostFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, last_hidden, hidden_states=None, forward_stage_ranges=None, backward_stage_ranges=None, last_hidden_shape=None, return_hidden_states=False):
        topo = config['topology']
        ctx.return_hidden_states = return_hidden_states
        last_hidden = broadcast(last_hidden, config["pipe_size"] - 1, config["pipe_comm"])
        last_hidden = last_hidden.chunk(topo.stages, dim=0)
        output = last_hidden[topo.stage_id]
        output.requires_grad_()

        if return_hidden_states:
            ctx.stage_id = topo.stage_id
            ctx.stages = topo.stages
            ctx.backward_stage_ranges = backward_stage_ranges
            middle_hiddens = []
            for stage_id in range(ctx.stages):
                if ctx.stage_id == stage_id:
                    middle_hidden = hidden_states
                else:
                    middle_shape = (forward_stage_ranges[stage_id],) + last_hidden_shape
                    middle_hidden = torch.zeros(middle_shape, device=hidden_states.device, dtype=hidden_states.dtype)
                middle_hidden = broadcast(middle_hidden, stage_id, config["pipe_comm"])
                middle_hidden = middle_hidden.chunk(ctx.stages, dim=1)
                middle_hidden = middle_hidden[ctx.stage_id].clone()
                middle_hiddens.append(middle_hidden)
            middle_hiddens = torch.cat(middle_hiddens, dim=0)
            middle_hiddens.requires_grad_()
            return output, middle_hiddens
        else:
             return output

    @staticmethod
    def backward(ctx, grads, grad_middle=None):
        grad_list = all_gather(grads, config["pipe_comm"])
        grad_list = grad_list.flatten(start_dim=0, end_dim=1)

        if ctx.return_hidden_states:
            for stage_id in range(ctx.stages):
                layer_range = ctx.backward_stage_ranges[stage_id]
                grad_middle_state = grad_middle[layer_range]
                grad_middle_state = all_gather(grad_middle_state.transpose(0,1), config["pipe_comm"])
                grad_middle_state = grad_middle_state.flatten(start_dim=0, end_dim=1).transpose(0, 1)
                if ctx.stage_id == stage_id:
                    grad_hidden_state_list = grad_middle_state
            return grad_list, grad_hidden_state_list, None, None, None, None
        else:
             return grad_list

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

    def __init__(self, modules: Iterable[CheckpointBlock], num_hidden=1) -> None:
        super().__init__()
        self.num_hidden = num_hidden 
        self._modules = {}
        rank = config['rank']
        topo = config['topology']
        self.layer_ids = []
        pipe_group = topo.pp_group
        self.stages = topo.stages
        self.stage_id = topo.stage_id
        self.pipe_idx = topo.pipe_idx 
        for idx, module in enumerate(modules):
            if not isinstance(module, CheckpointBlock):
                module = CheckpointBlock(module)

            module._mode = "PIPE"
            module.stage_id = self.stage_id
            module.stages = self.stages

            self._modules[str(idx)] = module

        self.layer_ids = self.get_range_by_stage_id(self.stage_id)

        for i,layer_id in enumerate(self.layer_ids):
            self._modules[str(layer_id)].layer_id = layer_id
            self._modules[str(layer_id)]._is_first_stage = True if self.stage_id == 0 else False
            self._modules[str(layer_id)]._is_last_stage = True if self.stage_id == self.stages-1 else False
            self._modules[str(layer_id)]._is_first_layer = True if i == 0 else False
            self._modules[str(layer_id)]._is_last_layer = True if i == len(self.layer_ids)-1 else False
#if i > 0:
#self._modules[str(layer_id)].set_pre_module(self._modules[str(layer_id-1)])

        self.partition_modules(self.layer_ids)
        self.next_rank = pipe_group[self.pipe_idx, self.stage_id + 1] if self.stage_id < config['pipe_size'] - 1 else -1
        self.prev_rank = pipe_group[self.pipe_idx, self.stage_id - 1] if self.stage_id > 0 else -1
        # self.micro_batches = config['num_micro_batches']

        self.save_list = [(i, i) for i in range(len(self.layer_ids))]
            
    def __len__(self) -> int:
        return len(self._modules) 

    def __iter__(self) -> Iterator[CheckpointBlock]:
        return iter(self._modules.values())

    def __getitem__(self, index: Union[int, str]) -> CheckpointBlock:
        return self._modules[str(index)]

    def forward(self, hidden_state, *args, batch_related=[], return_hidden_states=False):
        self.return_hidden_states = return_hidden_states
        batch_size = hidden_state.shape[0]
        num_micros = config["micros"]
        args = args + (batch_related, )
        hidden_state_list, args_list = PipePreFunction.apply(hidden_state, *args)

        hidden_state_list = hidden_state_list.flatten(0, 1).chunk(num_micros, dim=0)
        outputs = []
        hidden_states = []

        for micro_idx, (hidden_state, arg) in enumerate(zip(hidden_state_list, args_list)):
            micro_hidden_states = []
            for idx,layer_id in enumerate(self.layer_ids):
                self._modules[str(layer_id)]._micro_idx = micro_idx
                if return_hidden_states:
                    self._modules[str(layer_id)].return_hidden_states = return_hidden_states
                    self._modules[str(layer_id)].hidden_states = micro_hidden_states
                hidden_state = self._modules[str(layer_id)](hidden_state, *arg)
            outputs.append(hidden_state)
            if return_hidden_states:
                hidden_states.append(torch.stack(micro_hidden_states, dim=0))

        last_hidden = torch.cat(outputs, dim=0)
        last_hidden_shape = last_hidden.shape

        if return_hidden_states:
            hidden_states = torch.cat(hidden_states, dim=1) 
            forward_stage_ranges = []
            backward_stage_ranges = []
            for stage_id in range(self.stages):
                forward_stage_ranges.append(self.get_part_len_by_stage_id(stage_id))
                backward_stage_ranges.append(self.get_range_by_stage_id(stage_id))
            outputs, hidden_states = PipePostFunction.apply(last_hidden, hidden_states, forward_stage_ranges, backward_stage_ranges, last_hidden_shape, return_hidden_states)
            return outputs, hidden_states 
        else:
            outputs = PipePostFunction.apply(last_hidden)
            return outputs

    def get_range_by_stage_id(self, stage_id : int) -> List[int]:
        part_lens = [0]+[self.get_part_len_by_stage_id(i) for i in range(stage_id+1)]
        start = sum(part_lens[:stage_id+1])
        end = start + part_lens[stage_id+1]
        return range(start, end)

    def get_part_len_by_stage_id(self, stage_id : int) -> int:
        return len(self) // self.stages + (stage_id < (len(self) % self.stages))

    def get_stage_by_layer_id(self, layer_id : int) -> int:
        part_len = len(self) // self.stages
        rest = len(self) % self.stages
        if layer_id // (part_len + 1) < rest:
            return layer_id // (part_len + 1)
        else:
            return rest + (layer_id - rest * (part_len+1)) // part_len

    def partition_modules(self, idxs) -> None:
        for i in range(len(self)):
            contiguous_params = {}
            for kw, val in self[i]._storage_info.items():
                storage_type = val["storage_type"]
                contiguous_params[kw] = storage_type(round_up(val["total"], config["world_size"] // config["pipe_size"]))
                nccl.allGather(
                    self[i]._storage_params[kw].storage(),
                    contiguous_params[kw],
                    config["comm"]
                )

            if i not in idxs:
                for name, param in self[i]._module.named_parameters():
                    param.data = torch.tensor([], dtype = param.dtype, device = param.device)
                for kw, val in self[i]._storage_info.items():
                    val["begin"] = self.stage_id
                    val["end"] = self.stage_id + 1
                    val["partition_size"] = 1
                    val["total"] = val["world_size"]
                    dtype = self[i]._storage_params[kw].dtype
                    device = self[i]._storage_params[kw].device
                    self[i]._storage_params[kw] = \
                        torch.nn.Parameter(torch.tensor([0], dtype = dtype, device=device))
            else:
                for kw, val in self[i]._storage_info.items():
                    storage_type = val["storage_type"]
                    val["world_size"] = config["world_size"] // config["pipe_size"]
                    partition_size = round_up(val["total"], val["world_size"]) // val["world_size"]
                    val["partition_size"] = partition_size
                    val["begin"] = config['zero_rank'] * partition_size
                    val["end"] = (config['zero_rank'] + 1) * partition_size
                    storage_param_buffer = storage_type(partition_size)
                    dtype = storage_param_buffer.dtype
                    device = storage_param_buffer.device
                    self[i]._storage_params[kw] = torch.nn.Parameter(
                        torch.tensor([], dtype=dtype, device=device).set_(storage_param_buffer)
                    )
                    if val["requires_grad"]:
                        self[i]._storage_params[kw].requires_grad_(True)
                    else:
                        self[i]._storage_params[kw].requires_grad_(False)
                ordered_parameters = list(self[i]._module.named_parameters())
                for idx, named_param in enumerate(ordered_parameters):
                    name, param = named_param
                    param_info = self[i]._param_info[idx]
                    kw_name = _get_param_kw(param)
                    storage_info = self[i]._storage_info[kw_name]
                    storage_st = storage_info["begin"]
                    storage_end = storage_info["end"]
                    param_st = param_info["offset"]
                    param_end = param_st + param_info["size"]
                    if not (param_st >= storage_end or param_end <= storage_st):
                        # copy offset in parameter storage
                        offset_st = max(storage_st - param_st, 0)
                        offset_end = min(storage_end - param_st, param_info["size"])
                        assert offset_st < offset_end
                        to_offset_st = offset_st + param_st - storage_st
                        to_offset_end = offset_end + param_st - storage_st
                        d_dtype = self[i]._storage_params[kw_name].dtype
                        d_device = self[i]._storage_params[kw_name].device
                        param.data = torch.tensor([], dtype=param.dtype, device=param.device).set_(self[i]._storage_params[kw_name].storage(), to_offset_st, (to_offset_end - to_offset_st,))
                        param_info["begin"] = to_offset_st
                        param_info["end"] = (to_offset_end - to_offset_st,)
                        param.data[:] = \
                            torch.tensor([], dtype=d_dtype, device=d_device).set_(contiguous_params[kw], storage_st+to_offset_st, (to_offset_end - to_offset_st,))[:]
                    else:
                        param.data = torch.tensor([], dtype=param.dtype, device=param.device)
            del contiguous_params
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, module in self._modules.items():
            idx = int(name)
            name = prefix + name + '.'
            
            dst = OrderedDict() # creates an temporary ordered dict
            dst._metadata = OrderedDict()

            if idx in self.layer_ids:
                with torch.no_grad():
                    with CheckpointBlockContext(module, pipe=True):
                        module._module.state_dict(destination=dst, prefix=name, keep_vars=False)
                if config["zero_rank"] == 0:
                    if config["rank"] == 0:
                        destination.update(dst)
                    else:
                        assert list(dst.keys()) == [name+n for n, parameter in module._module.named_parameters()]
                        for key, tensor in dst.items():
                            send_activations(tensor.cuda(), 0, config['pipe_comm'])
            if config['rank'] == 0 and idx not in self.layer_ids:
                for n, parameter in module._module.named_parameters():
                    destination[name+n] = recv_activations(self.get_stage_by_layer_id(idx), config['pipe_comm'])

