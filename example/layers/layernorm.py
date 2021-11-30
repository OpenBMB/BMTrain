import torch
import bmpretrain as bmp
from cpm_kernels.torch.layernorm import OpLayerNormMean, OpLayerNormNoMean


class LayerNorm(bmp.DistributedModule):
    def __init__(self, hidden_size : int, eps : float = 1e-5, bias=True, dtype=torch.half):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = bmp.DistributedParameter(torch.ones(hidden_size, dtype=dtype))
        self.bias = bmp.DistributedParameter(torch.zeros(hidden_size, dtype=dtype)) if bias else None
    
    def forward(self, x : torch.Tensor):
        """
        Args:
            x: (batch_size, hidden_size, seq_len)       fp16
        
        Returns:
            out : (batch_size, hidden_size, seq_len)    fp16
        """
        assert x.size(1) == self.hidden_size
        
        if self.bias is not None:
            return  OpLayerNormMean.apply(x, self.eps, self.weight, self.bias)
        else:
            return OpLayerNormNoMean.apply(x, self.eps, self.weight)
    