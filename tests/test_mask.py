import torch
import bmtrain as bmt
import torch
import random
import time
import copy

def test_main():
    mask = bmt.nn.Mask()

    for i in range(1, 101):
        x1 = torch.rand(32, 32, 32, 5*i)
        m = (torch.randint(0, 10, (32, 32, 5*i))==0).cuda()
        x2 = x1.clone().detach()

        x1 = x1.cuda().half().requires_grad_()
        x2 = x2.cuda().requires_grad_()

        y1 = mask(x1.view(x1.size(0), x1.size(1), -1), m.view(m.size(0), -1), -100)
        y2 = torch.masked_fill(x2, (m==False).view(m.size(0), 1, m.size(1), m.size(2)), -100)

        print(f"forward: {(y1.view(y2.shape)-y2).abs().max()}")

        grad1 = torch.rand(32, 32, 32, 5*i)
        grad2 = grad1.clone().detach()

        grad1 = grad1.cuda().half().view(y1.shape)
        grad2 = grad2.cuda()

        y1.backward(grad1)
        y2.backward(grad2)

        print(f"backward: {(x1.grad-x2.grad).abs().max()}" )

def benchmark_memory():
    mask = bmt.nn.Mask()
    x = torch.rand(32, 128, 512*512).half().cuda().requires_grad_(True)
    m = torch.randint(0, 2, (32, 512*512)).bool().cuda()
    mm = torch.randint(0, 2, (32, 512*512)).bool().cuda()
    print(torch.cuda.memory_summary())
    x = mask(x, m, -100)
    print(torch.cuda.memory_summary())
    x = mask(x, mm, 100)
    print(torch.cuda.memory_summary())

if __name__ == "__main__":
    test_main()
    # benchmark_memory() # TODO