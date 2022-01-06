import torch
import torch.nn.functional as F
import bmpretrain as bmp

class Linear(bmp.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmp.DistributedParameter(torch.empty(out_features, in_features, dtype=dtype, device="cuda"), init_method=torch.nn.init.xavier_normal_)
        if bias:
            self.bias = bmp.DistributedParameter(torch.empty(out_features, dtype=dtype, device="cuda"), init_method=torch.nn.init.zeros_)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )