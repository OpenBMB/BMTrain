import torch
from ..layer import DistributedModule
from ..parameter import DistributedParameter
import torch.nn.functional as F
import math


class LayerNorm(DistributedModule):
    def __init__(self, normalized_shape, eps = 1e-6, bias : bool = True, dtype = torch.half) -> None:
        super().__init__()
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = DistributedParameter(torch.ones(normalized_shape, dtype=dtype), group="layernorm")
        if bias:
            self.bias = DistributedParameter(torch.zeros(normalized_shape, dtype=dtype), group="layernorm")
        else:
            self.bias = None
    
    def forward(self, x):
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
        )