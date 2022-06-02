import torch
import bmtrain
from tqdm import tqdm

from bmtrain.utils import print_rank

class DistributedLinear(bmtrain.DistributedModule):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.weight = bmtrain.DistributedParameter(torch.empty(dim_in, dim_out, dtype=torch.float))
    
    def forward(self, x : torch.Tensor):
        last_dim = x.size(-1)
        x_viewd = x.view(-1, last_dim)
        out = torch.matmul(x_viewd, self.weight).view(x.size()[:-1] + (self.dim_out,))
        var_out = out.var(dim=-1, keepdim=True)
        return out / var_out + x

class TestMLP(bmtrain.DistributedModule):
    def __init__(self):
        super().__init__()

        self.mlp = torch.nn.ModuleList([
            DistributedLinear(8, 8) for _ in range(16)
        ])
    
    @bmtrain.checkpoint
    def forward_segments(self, layer_ids, hidden_state : torch.Tensor):
        for idx in layer_ids:
            hidden_state = self.mlp[idx](hidden_state)
        return hidden_state
    
    def forward(self, x):
        for segments in [
            [0, 1], [2, 3],
            [4, 5], [6, 7],
            [8, 9], [10, 11],
            [12, 13], [14, 15]
        ]:
            x = self.forward_segments(segments, x)
            bmtrain.wait_loader()
        return x

def main():
    bmtrain.init_distributed()

    model = TestMLP()

    print_rank("Model mem\n", torch.cuda.memory_summary())
    state = model.state_dict()
    for kw, param in state.items():
        torch.nn.init.normal_(param, std=0.02)
    model.load_state_dict(state)    # split parameters automatically
    print_rank("Model after loading state dict\n", torch.cuda.memory_summary())
    
    for i in range(bmtrain.world_size()):
        v = torch.randn(4, 8, dtype=torch.float, device="cuda")
        if i == bmtrain.rank():
            raw_data = v

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for iter in tqdm(range(100)):
        # load data
        data = raw_data.clone().requires_grad_()
        optimizer.zero_grad()
        output = model(data)
        loss = ((output - bmtrain.rank()) ** 2).mean()
        print_rank("Iter %d, loss: " % iter, bmtrain.sum_loss(loss).item())
        
        loss.backward()
        optimizer.step()
        bmtrain.wait_optimizer() # loader stream wait for optimizer

    print_rank(torch.cuda.memory_summary())
    bmtrain.synchronize()
    print_rank(torch.cuda.memory_summary(), rank=1)

    

if __name__ == "__main__":
    main()