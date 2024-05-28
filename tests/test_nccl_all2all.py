
from utils import *

import bmtrain as bmt
import torch
from bmtrain import nccl
import math
import os

def test_main(dtype):
    # x shape (2,8)
    refx = torch.tensor([[(bmt.rank()*2+y)*10+x for x in range(8)] for y in range(2)], dtype=dtype, device="cuda")
    refy = torch.tensor([[y*10+bmt.rank()*2+x for x in range(2)] for y in range(8)], dtype=dtype, device="cuda")

    x = refx.clone()
    bmt.print_rank("x")
    for r in range(4):
        bmt.print_rank(x, rank=r)
        bmt.synchronize()

    x = torch.cat(x.chunk(4, dim=1), dim=0).contiguous()
    y = torch.zeros((8,2), dtype=dtype, device="cuda")
    nccl.all2all(x.storage(), y.storage(), bmt.config['comm'])
    bmt.print_rank("y")
    for r in range(4):
        bmt.print_rank(y, rank=r)
        if bmt.rank() == r: assert (y == refy).all()
        bmt.synchronize()

    x = torch.zeros((8,2), dtype=dtype, device="cuda")
    nccl.all2all(y.storage(), x.storage(), bmt.config['comm'])
    x = torch.cat(x.chunk(4, dim=0), dim=1).contiguous()
    bmt.print_rank("x")
    for r in range(4):
        bmt.print_rank(x, rank=r)
        if bmt.rank() == r: assert (x == refx).all()
        bmt.synchronize()

class Attention(bmt.DistributedModule):
    def __init__(self, 
            dim_model : int, dim_head : int,
            num_heads : int, bias : bool = True,
            sequence_parallel : bool = False,
            dtype = None
        ) -> None:
        super().__init__()

        self.project_q = bmt.nn.Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.project_k = bmt.nn.Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.project_v = bmt.nn.Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.project_out = bmt.nn.Linear(dim_head * num_heads, dim_model, bias=bias, dtype=dtype)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_model = dim_model

        self.sequence_parallel = sequence_parallel
    
    def forward(self, 
            hidden : torch.Tensor,        # (batch_size, seq_q, dim_model)
        ) -> torch.Tensor:
        batch_size, seq, dim_model = hidden.size()

        h_q : torch.Tensor = self.project_q(hidden)
        h_k : torch.Tensor = self.project_k(hidden)
        h_v : torch.Tensor = self.project_v(hidden)

        if self.sequence_parallel:
            assert batch_size == 1
            h_q = h_q.view(seq, -1)
            h_k = h_k.view(seq, -1)
            h_v = h_v.view(seq, -1)
            h_q = torch.cat(h_q.chunk(bmt.world_size(), dim=1), dim=0).contiguous()
            h_k = torch.cat(h_k.chunk(bmt.world_size(), dim=1), dim=0).contiguous()
            h_v = torch.cat(h_v.chunk(bmt.world_size(), dim=1), dim=0).contiguous()
            h_q = bmt.distributed.all_to_all(h_q, bmt.config['comm'])
            h_k = bmt.distributed.all_to_all(h_k, bmt.config['comm'])
            h_v = bmt.distributed.all_to_all(h_v, bmt.config['comm'])
            seq = seq * bmt.world_size()
            h_q = h_q.view(batch_size, seq, -1)
            h_k = h_k.view(batch_size, seq, -1)
            h_v = h_v.view(batch_size, seq, -1)

        h_q = h_q.view(batch_size, seq, -1, self.dim_head)
        h_k = h_k.view(batch_size, seq, -1, self.dim_head)
        h_v = h_v.view(batch_size, seq, -1, self.dim_head)

        h_q = h_q.permute(0, 2, 1, 3).contiguous()
        h_k = h_k.permute(0, 2, 1, 3).contiguous()
        h_v = h_v.permute(0, 2, 1, 3).contiguous()

        h_q = h_q.view(-1, seq, self.dim_head)
        h_k = h_k.view(-1, seq, self.dim_head)
        h_v = h_v.view(-1, seq, self.dim_head)

        score = torch.bmm(
            h_q, h_k.transpose(1, 2)
        )
        score = score / math.sqrt(self.dim_head)

        score = score.view(batch_size, -1, seq, seq)

        score = score.view(-1, seq, seq)

        h_out = torch.bmm(
            score, h_v
        )
        h_out = h_out.view(batch_size, -1, seq, self.dim_head)
        h_out = h_out.permute(0, 2, 1, 3).contiguous()
        h_out = h_out.view(batch_size, seq, -1)

        if self.sequence_parallel:
            h_out = h_out.view(seq, -1)
            h_out = bmt.distributed.all_to_all(h_out, bmt.config['comm'])
            h_out = torch.cat(h_out.chunk(bmt.world_size(), dim=0), dim=1).contiguous()
            seq = seq // bmt.world_size()
            h_out = h_out.view(batch_size, seq, -1)

        attn_out = self.project_out(h_out)
        return attn_out

def test_ulysses(dtype):
    model1 = Attention(dim_model=768, dim_head=32, num_heads=8, dtype=dtype, sequence_parallel=False)
    bmt.init_parameters(model1)
    bmt.save(model1, "test.pt")
    model2 = Attention(dim_model=768, dim_head=32, num_heads=8, dtype=dtype, sequence_parallel=True)
    bmt.load(model2, "test.pt")

    xx = torch.randn((1, 128, 768), dtype=dtype, device="cuda").requires_grad_()
    x_sp = xx.clone().chunk(bmt.world_size(), dim=1)[bmt.rank()].detach().requires_grad_()

    yy = model1(xx)
    y_sp = model2(x_sp)

    gg = torch.randn((1, 128, 768), dtype=dtype, device="cuda")
    g = gg.chunk(bmt.world_size(), dim=1)[bmt.rank()]

    yy.backward(gg)
    y_sp.backward(g)

    for r in range(bmt.world_size()):
        if bmt.rank() == r:
            print(r)
            print(y_sp)
            print(yy.chunk(bmt.world_size(), dim=1)[bmt.rank()])
            assert torch.allclose(y_sp, yy.chunk(bmt.world_size(), dim=1)[bmt.rank()])
            print(x_sp.grad)
            print(xx.grad.chunk(bmt.world_size(), dim=1)[bmt.rank()])
            assert torch.allclose(x_sp.grad, xx.grad.chunk(bmt.world_size(), dim=1)[bmt.rank()], rtol=1e-3, atol=1e-1)
        bmt.synchronize()
    
    if bmt.rank() == 0: os.remove("test.pt")

if __name__ == "__main__":
    bmt.init_distributed()

    # test_main(torch.half)
    test_ulysses(torch.half)
