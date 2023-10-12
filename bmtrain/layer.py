import torch
from .parameter import DistributedParameter
import itertools

class DistributedModule(torch.nn.Module):
    """
    DistributedModule is a subclass of torch.nn.Module that overrides the `__getattr__` method to gather distributed parameters automatically.
    
    """

    def __getattr__(self, name: str):
        ret = super().__getattr__(name)
        # gather distributed parameters if not in CheckpointBlock
        if isinstance(ret, DistributedParameter): 
            return ret.gather()
        return ret

