import torch
import bmpretrain
import os
import cpm_kernels.torch as ct

from bmpretrain.utils import print_rank

class DistributedLinear(bmpretrain.DistributedModule):
    def __init__(self, dim_in, dim_out, int8 : bool = True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.int8 = int8

        self.weight = bmpretrain.DistributedParameter(torch.empty(dim_in, dim_out, dtype=torch.half))
    
    def forward(self, x : torch.Tensor):
        last_dim = x.size(-1)
        x_viewd = x.view(-1, last_dim)
        out = ct.bmm(x_viewd.unsqueeze(0), False, self.weight.unsqueeze(0), False, self.int8)
        mean_out = out.mean()
        var_out = out.var()
        return (out - mean_out) / var_out

class TestMLP(bmpretrain.DistributedModule):
    def __init__(self):
        super().__init__()

        self.mlp = torch.nn.ModuleList([
            DistributedLinear(4096, 4096) for _ in range(16)
        ])
    
    def forward(self, x):
        cnt = 0
        for layer in self.mlp:
            x = layer(x)
            bmpretrain.wait_loader()
            print_rank("Layer %d\n" % cnt, torch.cuda.memory_summary())
            cnt += 1
        return x

def main():    
    bmpretrain.init_distributed()

    model = TestMLP()

    print_rank("Model mem\n", torch.cuda.memory_summary())
    states = model.state_dict()
    print_rank("Model after getting state dict\n", torch.cuda.memory_summary())
    for kw in states.keys():
        torch.nn.init.normal_(states[kw])
    model.load_state_dict(states)
    print_rank("Model after loading state dict\n", torch.cuda.memory_summary())
    bmpretrain.print_rank(model.mlp[0].weight)
    bmpretrain.synchronize()
    bmpretrain.print_rank(model.mlp[0].weight, rank=1)
    bmpretrain.synchronize()
    value = torch.randn(4, 4096, device="cuda", dtype=torch.half)

    print_rank("Start mem\n", torch.cuda.memory_summary())
    with torch.no_grad():
        out = model(value)
    
    for i in range(bmpretrain.world_size()):
        if i == bmpretrain.rank():
            bmpretrain.print_block("Rank %d output" % bmpretrain.rank(), "%s" % out)
        bmpretrain.synchronize()
    

if __name__ == "__main__":
    main()