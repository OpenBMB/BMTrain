import torch
import cpm_kernels.torch as ct
import bmpretrain as bmp
import math

class FeedForward(bmp.DistributedModule):
    def __init__(self, dim_model : int, dim_ff : int, int8=True, dtype=torch.half):
        super().__init__()
        self.w_0 = bmp.DistributedParameter(torch.empty(dim_ff, dim_model, dtype=dtype), init_method=bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=1))
        self.w_1 = bmp.DistributedParameter(torch.empty(dim_ff, dim_model, dtype=dtype), init_method=bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=1))
        self.w_out = bmp.DistributedParameter(torch.empty(dim_model, dim_ff, dtype=dtype), init_method=bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=1))

        self.int8 = int8
        self.dim_model = dim_model
        self.dim_ff = dim_ff
    
    def forward(self, x):
        """
        Args:
            x : (batch, hidden_size, seq_len)       fp16
        Returns:
            out : (batch, hidden_size, seq_len)     fp16
        """
        # (1#batch, dim_ff, dim_model) @ (batch, dim_model, seq_len) = (batch, dim_ff, seq_len)
        gelu_score = ct.gelu(
            ct.bmm(self.w_0.unsqueeze(0), False, x, False, int8=self.int8)
        )
        hidden_out = ct.bmm(self.w_1.unsqueeze(0), False, x, False, int8=self.int8)
        
        # (batch, dim_ff, seq_len)
        x = ct.element_mul(gelu_score, hidden_out)

        # (1#batch, dim_model, dim_ff) @ (batch, dim_ff, seq_len) = (batch, dim_model, seq_len)
        x = ct.bmm(self.w_out.unsqueeze(0), False, x, False, int8=self.int8)
        return x
