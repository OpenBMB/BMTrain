import torch
import bmtrain as bmt
import torch
import random
import time
import copy

def test_main():
    softmax_func1 = bmt.nn.FusedSoftmax()
    softmax_func2 = torch.nn.Softmax(dim=-1)

    for i in [5]*10 + [10]*10 + [50]*10 + [100]*10 + [500]*10:
        x1 = torch.rand(32, 32, 32, 5*i)
        mask = torch.randint(0, 10, (32, 32, 32, 5*i))==0
        x1.masked_fill_(mask, -1e9)
        x2 = x1.clone().detach()

        x1 = x1.cuda().half().requires_grad_()
        x2 = x2.cuda().requires_grad_()

        y1 = softmax_func1(x1)
        y2 = softmax_func2(x2)

        print(f"forward: {(y1-y2).abs().max()}")

        grad1 = torch.rand(32, 32, 32, 5*i)
        grad2 = grad1.clone().detach()

        grad1 = grad1.cuda().half()
        grad2 = grad2.cuda()

        y1.backward(grad1)
        y2.backward(grad2)

        print(f"backward: {(x1.grad-x2.grad).abs().max()}" )

def benchmark_memory():
    softmax_func = bmt.nn.FusedSoftmax()
    # softmax_func = torch.nn.Softmax(dim=-1)
    x = torch.rand(32, 128, 512, 512).cuda().half().requires_grad_(True)
    print(torch.cuda.memory_summary())
    x = softmax_func(x)
    print(torch.cuda.memory_summary())

if __name__ == "__main__":
    test_main()
    # benchmark_memory()