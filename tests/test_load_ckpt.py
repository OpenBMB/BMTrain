from utils import *
import torch
import torch.nn.functional as F
import bmtrain as bmt
import os
from collections import OrderedDict

class Linear_Normal(torch.nn.Module):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device="cuda"))
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=dtype, device="cuda"))
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class Linear_BMT(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=dtype), init_method=torch.nn.init.xavier_normal_)
        if bias:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=dtype), init_method=torch.nn.init.zeros_)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

def test_save_load(m):
    bmt.save(m, "test.pt", non_blocking=False)
    bmt.load(m, "test.pt")
    bmt.save(m, "test.pt", non_blocking=True)
    bmt.load(m, "test.pt")
    bmt.save(m, "test.pt", non_blocking=False, save_gather=True)
    bmt.load(m, "test.pt", load_gather=True)
    bmt.clean("test.pt")


def test_main():
    # Transformer BlockList
    m = Linear_Normal(256, 256).cuda()
    m2 = bmt.TransformerBlockList([bmt.Block(Linear_BMT(256, 256))])
    m2_state = m.state_dict().copy()
    m2_state["0.weight"] = m2_state.pop("weight")
    m2_state["0.bias"] = m2_state.pop("bias")
    test_save_load(m2)
    m2.load_state_dict(m2_state)
    for key in m.state_dict():
        bmt_key = f"0.{key}"
        assert bmt_key in m2.state_dict(), "wrong key in bmtrain model"
        assert (m2.state_dict()[bmt_key].cuda() == m.state_dict()[key]).all() , "wrong param in bmtrain model"
    print("Transformer Blocklist load_state_dict ,state_dict, bmt.load/save test passed")

    # Block 
    m3 = bmt.Block(Linear_BMT(256, 256))
    m3.load_state_dict(m.state_dict())
    for key in m.state_dict():
        assert key in m3.state_dict(), "wrong key in bmtrain model"
        assert (m.state_dict()[key] == m3.state_dict()[key].cuda()).all(), "wrong param in bmtrain model"
    test_save_load(m2)
    print("Block load_state_dict ,state_dict, bmt.load/save test passed")

    # normal Distributed module
    m4 = Linear_BMT(256, 256)
    m4.load_state_dict(m.state_dict())
    for key in m.state_dict():
        assert key in m4.state_dict(), "wrong key in bmtrain model"
        assert (m.state_dict()[key] == m4.state_dict()[key].cuda()).all(), "wrong param in bmtrain model"
    test_save_load(m2)
    print("bmt.distributedmodule load_state_dict, state_dict, bmt.load/save test passed")

if __name__ == "__main__":
    bmt.init_distributed()

    test_main()
