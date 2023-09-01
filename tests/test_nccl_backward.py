from utils import *
import os
import bmtrain as bmt
import torch
import time
def test_main():
    x = torch.full((1,), bmt.rank() + 1, dtype=torch.half, device="cuda").requires_grad_(True)
    y = bmt.distributed.all_reduce(x, "prod").view(-1)
    loss = (y * y).sum() / 2
    loss.backward()
    ref = y
    for i in range(bmt.world_size()):
        if i != bmt.rank(): ref *= i+1
    print(x.grad)
    assert_eq(x.grad, ref)

if __name__ == "__main__":
    os.environ["WORLD_SIZE"] = "2"
    bmt.init_distributed()

    test_main()