import bmtrain as bmt
import torch
from bmtrain import config
from bmtrain.block_layer import CheckpointBlockContext,  CheckpointBlock, TransformerBlockList
from typing import List
import torch.nn.functional as F

class Linear(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, init_weight = None, init_bias = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.out = {}
        if init_weight:
            self.weight = bmt.DistributedParameter(torch.tensor(init_weight, dtype=torch.float, device="cuda").reshape(out_features, in_features))
        else:
            self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=torch.float, device="cuda"), init_method=torch.nn.init.xavier_normal_)

        if init_bias:
            self.bias = bmt.DistributedParameter(torch.tensor(init_bias, dtype=torch.float, device="cuda").reshape(out_features,))
        else:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=torch.float, device="cuda"), init_method=torch.nn.init.zeros_)
    
    def forward(self, input):
        ret = F.linear(input, self.weight, self.bias)
        return ret

bmt.init_distributed()
a = Linear(256, 256)
b = Linear(256, 256)
m = TransformerBlockList([CheckpointBlock(a), CheckpointBlock(b)])
a.bias.requires_grad_(False)
bmt.init_parameters(m)
inp = torch.rand((1, 10, 256)).cuda()*100
if bmt.rank() == 0:
    print("===============================================forward===========================================")
bmt.synchronize()
logits = m(inp)
bmt.synchronize()
loss = logits.sum()
loss.backward()
bmt.print_rank(
    bmt.inspect.format_summary(
        bmt.inspect.inspect_model(m, '*')
    )
)
print(a.weight.grad)
print(a.bias.grad)