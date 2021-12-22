import torch
from ..layer import DistributedModule
from ..parameter import DistributedParameter, ParameterInitializer
import torch.nn.functional as F
import math

class Linear(DistributedModule):
    def __init__(self, in_features, out_features, bias : bool = True, dtype = torch.half) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = DistributedParameter(torch.empty(out_features, in_features, dtype=dtype), init_method=ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=1))
        if bias:
            self.bias = DistributedParameter(torch.empty(out_features, dtype=dtype), init_method=ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=1))
        else:
            self.bias = None
    
    def forward(self, x):
        return F.linear(
            x,
            self.weight,
            self.bias
        ) / math.sqrt(self.in_features)