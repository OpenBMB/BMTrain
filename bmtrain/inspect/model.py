import torch
from ..store import broadcast_object
from ..pipe_layer import PipelineTransformerBlockList
from ..block_layer import Block
from ..parameter import DistributedParameter
from .. import nccl
from ..global_var import config
import fnmatch

def _gather_value(value : torch.Tensor, partition_size, origin_size):
    global_size = partition_size * config['world_size']

    global_buffer = torch.empty(global_size, dtype=value.dtype, device=value.device)

    if value.numel() != partition_size:
        tmp_buf = torch.zeros(partition_size, dtype=value.dtype, device=value.device)
        tmp_buf[:value.numel()] = value[:]
        nccl.allGather(
            tmp_buf,
            global_buffer,
            config['comm']
        )
    else:
        nccl.allGather(
            value,
            global_buffer,
            config['comm']
        )

    output_tensor = global_buffer[:origin_size.numel()].view(origin_size)

    return output_tensor

def inspect_pipeline_transformer_block_list(pipe_model: PipelineTransformerBlockList, param_name : str, _prefix : str = ''):
    ret = []
    for name, model in pipe_model._modules.items():
        idx = int(name)
        prefix = _prefix + name + '.'

        # fast check
        pass_fast_check = False
        for param in model._param_info:
            abs_name = prefix + param["name"]
            if fnmatch.fnmatch(abs_name, param_name):
                pass_fast_check = True
                break
        if not pass_fast_check:
            continue

        if idx in pipe_model.layer_ids:
            _param_buffer = {}
            _grad_buffer = {}
            for kw, val in model._storage_info.items():
                local_param = model._storage_params[kw]
                _param_buffer[kw] = torch.empty(
                    val["partition_size"] * val['world_size'],
                    dtype=local_param.dtype, device=local_param.device
                )
                if local_param.grad is not None:
                    _grad_buffer[kw] = torch.empty(
                        val["partition_size"] * val['world_size'],
                        dtype=local_param.dtype, device=local_param.device
                    )
            
            nccl.groupStart()
            for kw, val in model._storage_info.items():
                nccl.allGather(
                    model._storage_params[kw],
                    _param_buffer[kw],
                    val["zero_comm"]
                )
                if model._storage_params[kw].grad is not None:
                    nccl.allGather(
                        model._storage_params[kw].grad,
                        _grad_buffer[kw],
                        val["zero_comm"]
                    )

            nccl.groupEnd()
            for param in model._param_info:
                abs_name = prefix + param["name"]
                if fnmatch.fnmatch(abs_name, param_name):
                    kw_name = param["kw_name"]
                    offset = param["offset"]
                    shape = param["shape"]
                    numel = shape.numel()
                    p = _param_buffer[kw_name][offset:offset+numel].view(shape)
                    if kw_name in _grad_buffer:
                        g = _grad_buffer[kw_name][offset:offset+numel].view(shape)
                        info = {
                            "name": abs_name,
                            "shape": tuple(shape),
                            "std": p.std().cpu().item(),
                            "mean": p.mean().cpu().item(),
                            "grad_std": g.std().cpu().item(),
                            "grad_mean": g.mean().cpu().item(),
                            "max": p.max().cpu().item(),
                            "min": p.min().cpu().item(),
                        }
                    else:
                        info = {
                            "name": abs_name,
                            "shape": tuple(shape),
                            "std": p.std().cpu().item(),
                            "mean": p.mean().cpu().item(),
                            "grad_std": 0.,
                            "grad_mean": 0.,
                            "max": p.max().cpu().item(),
                            "min": p.min().cpu().item(),
                        }
                    broadcast_object(info, config["pipe_comm"], pipe_model.get_stage_by_layer_id(idx))
                    ret.append(info)
        else:
            for param in model._param_info:
                abs_name = prefix + param["name"]
                if fnmatch.fnmatch(abs_name, param_name):
                    info = broadcast_object({}, config["pipe_comm"], pipe_model.get_stage_by_layer_id(idx))
                    ret.append(info)

    return ret


def inspect_block(model : Block, param_name : str, prefix : str = ''):
    # fast check
    pass_fast_check = False
    for param in model._param_info:
        abs_name = prefix + param["name"]
        if fnmatch.fnmatch(abs_name, param_name):
            pass_fast_check = True
            break
    if not pass_fast_check:
        return []

    _param_buffer = {}
    _grad_buffer = {}
    for kw, val in model._storage_info.items():
        local_param = model._storage_params[kw]
        _param_buffer[kw] = torch.empty(
            val["partition_size"] * config['world_size'],
            dtype=local_param.dtype, device=local_param.device
        )
        if local_param.grad is not None:
            _grad_buffer[kw] = torch.empty(
                val["partition_size"] * config['world_size'],
                dtype=local_param.dtype, device=local_param.device
            )
    
    nccl.groupStart()
    for kw, val in model._storage_info.items():
        nccl.allGather(
            model._storage_params[kw],
            _param_buffer[kw],
            config["comm"]
        )
        if model._storage_params[kw].grad is not None:
            nccl.allGather(
                model._storage_params[kw].grad,
                _grad_buffer[kw],
                config["comm"]
            )

    nccl.groupEnd()
    ret = []
    for param in model._param_info:
        abs_name = prefix + param["name"]
        if fnmatch.fnmatch(abs_name, param_name):
            kw_name = param["kw_name"]
            offset = param["offset"]
            shape = param["shape"]
            numel = shape.numel()
            p = _param_buffer[kw_name][offset:offset+numel].view(shape)
            if kw_name in _grad_buffer:
                g = _grad_buffer[kw_name][offset:offset+numel].view(shape)
                ret.append({
                    "name": abs_name,
                    "shape": tuple(shape),
                    "std": p.std().cpu().item(),
                    "mean": p.mean().cpu().item(),
                    "grad_std": g.std().cpu().item(),
                    "grad_mean": g.mean().cpu().item(),
                    "max": p.max().cpu().item(),
                    "min": p.min().cpu().item(),
                })
            else:
                ret.append({
                    "name": abs_name,
                    "shape": tuple(shape),
                    "std": p.std().cpu().item(),
                    "mean": p.mean().cpu().item(),
                    "grad_std": 0.,
                    "grad_mean": 0.,
                    "max": p.max().cpu().item(),
                    "min": p.min().cpu().item(),
                })
    return ret

@torch.no_grad()
def inspect_model(model : torch.nn.Module, param_name : str, prefix : str = ''):
    """Inspect the model and return the summary of the parameters.

    Args:
        model (torch.nn.Module): The model to be inspected.
        param_name (str): The name of the parameter to be inspected. The wildcard '*' can be used to match multiple parameters.
        prefix (str): The prefix of the parameter name.
        
    Returns:
        list: The summary of the parameters.
    
    Example:
        >>> result_linear = bmt.inspect.inspect_model(model, "*.linear*")
        >>> result_layernorm = bmt.inspect.inspect_model(model, "*.layernorm*")
        >>> text_summray = bmt.inspect.format_summary(result_linear + result_layernorm)
        >>> bmt.print_rank(text_summary)
        name   shape     max     min     std     mean    grad_std  grad_mean
        ...

    """
    if isinstance(model, PipelineTransformerBlockList):
        return inspect_pipeline_transformer_block_list(model, param_name, prefix)
    elif isinstance(model, Block):
        return inspect_block(model, param_name, prefix)
    else:
        ret = []
        for name, param in model._parameters.items():
            if fnmatch.fnmatch(prefix + name, param_name):
                if isinstance(param, DistributedParameter):
                    p = _gather_value(param.data, param._partition_size, param._original_shape)
                else:
                    p = param
                if p is None:
                    continue
                stats = {
                    'name': prefix + name,
                    'shape': tuple(p.size()),
                    "std": p.std().cpu().item(),
                    "mean": p.mean().cpu().item(),
                    "max": p.max().cpu().item(),
                    "min": p.min().cpu().item(),
                }
                if param.grad is not None:
                    if isinstance(param, DistributedParameter):
                        g = _gather_value(param.grad.data, param._partition_size, param._original_shape)
                    else:
                        g = param.grad
                    stats["grad_std"] = g.std().cpu().item()
                    stats["grad_mean"] = g.mean().cpu().item()
                else:
                    stats["grad_std"] = 0.
                    stats["grad_mean"] = 0.
                ret.append(stats)
        for name, module in model._modules.items():
            ret.extend(inspect_model(module, param_name, prefix + name + '.'))
        return ret
