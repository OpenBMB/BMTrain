from typing import Iterable
import torch

from bmpretrain.utils import print_rank

from .block_layer import CheckpointBlock
from .parameter import DistributedParameter
from .global_var import config


def init_distributed_parameter(params : Iterable[torch.nn.Parameter]):
    for param in params:
        if not isinstance(param, DistributedParameter):
            continue
        if param._init_method is None:
            continue
        with torch.no_grad():
            partition_size = param.storage().size()
            global_size = partition_size * config['world_size']
            
            tmp_storage = param.storage_type()(global_size)
            tmp_tensor = torch.tensor([], dtype=param.dtype, device="cuda")
            tmp_tensor.set_(tmp_storage, 0, param._original_shape)

            param._init_method(tmp_tensor)

            param.storage().copy_(tmp_storage[partition_size * config['rank'] : partition_size * (config['rank'] + 1)])

def iterate_parameters(model : torch.nn.Module):
    for kw, val in model._parameters.items():
        yield val

def init_parameters(model : torch.nn.Module):
    modules = model.named_modules()
    for module_prefix, module in modules:
        if isinstance(module, CheckpointBlock):
            module.init_parameters()
        else:
            init_distributed_parameter( iterate_parameters(module) )
            