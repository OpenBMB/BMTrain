import torch
from ..block_layer import CheckpointBlock
from ..parameter import DistributedParameter
from .. import nccl
from ..global_var import config
import fnmatch

def _gather_value(value : torch.Tensor, partition_size, origin_size):
    global_size = partition_size * config['world_size']

    storage = value.storage_type()(global_size)
    
    nccl.allGather(
        value.storage(),
        storage,
        config['comm']
    )

    output_tensor = torch.tensor([], dtype=value.dtype, device="cuda")
    output_tensor.set_(storage, 0, origin_size)

    return output_tensor

def inspect_checkpoint_block(model : CheckpointBlock, param_name : str, prefix : str = ''):
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
        storage_type = model._storage_params[kw].storage_type()

        _param_buffer[kw] = storage_type(val["total"])
        _grad_buffer[kw] = storage_type(val["total"])
    
    nccl.groupStart()
    for kw, val in model._storage_info.items():
        nccl.allGather(
            model._storage_params[kw].storage(),
            _param_buffer[kw],
            config["comm"]
        )
        nccl.allGather(
            model._storage_params[kw].grad.storage(),
            _grad_buffer[kw],
            config["comm"]
        )
    nccl.groupEnd()
    ret = []
    for param in model._param_info:
        abs_name = prefix + param["name"]
        if fnmatch.fnmatch(abs_name, param_name):
            kw_name = param["kw_name"]
            dtype = _param_buffer[kw_name].dtype
            device = _param_buffer[kw_name].device
            offset = param["offset"]
            shape = param["shape"]
            p = torch.tensor([], dtype=dtype, device=device).set_(_param_buffer[kw_name], offset, shape)
            g = torch.tensor([], dtype=dtype, device=device).set_(_grad_buffer[kw_name], offset, shape)
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
    return ret

@torch.no_grad()
def inspect_model(model : torch.nn.Module, param_name : str, prefix : str = ''):
    if isinstance(model, CheckpointBlock):
        return inspect_checkpoint_block(model, param_name, prefix)
    else:
        ret = []
        for name, param in model._parameters.items():
            if fnmatch.fnmatch(prefix + name, param_name):
                if isinstance(param, DistributedParameter):
                    p = _gather_value(param.data, param.storage().size(), param._original_shape)
                else:
                    p = param
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
                        g = _gather_value(param.grad.data, param.storage().size(), param._original_shape)
                    else:
                        g = param.grad
                    stats["grad_std"] = g.std().cpu().item()
                    stats["grad_mean"] = g.mean().cpu().item()
                else:
                    stats["grad_std"] = None
                    stats["grad_mean"] = None
                ret.append(stats)
        for name, module in model._modules.items():
            ret.extend(inspect_model(module, param_name, prefix + name + '.'))
        return ret
