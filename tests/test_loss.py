import torch
import bmtrain as bmt
import torch
import random
import time
import copy

def run(x, tgt, loss_func, bigmodel=None, scale=32768, use_float=False):
    x = x.clone().detach()
    bigmodel = copy.deepcopy(bigmodel)
    if use_float:
        x = x.float()
        if bigmodel is not None:
            bigmodel = bigmodel.float()
    x = x.requires_grad_()
    # print(torch.cuda.memory_summary())
    if bigmodel is None:
        loss = loss_func(x, tgt)
    else:
        t = bigmodel(x)
        loss = loss_func(t, tgt)
        # loss = loss_func(bigmodel(x), tgt)
    # print(torch.cuda.memory_summary())
    (loss * scale).backward()
    # print(torch.cuda.memory_summary())
    return loss, x.grad

def check(x, tgt, loss_func1, loss_func2, bigmodel=None):
    loss_1, grad_1 = run(x, tgt, loss_func1, bigmodel=bigmodel)
    loss_2, grad_2 = run(x, tgt, loss_func2, bigmodel=bigmodel, use_float=True)
    assert(grad_1.isnan().sum()==0)
    assert(grad_2.isnan().sum()==0)
    print((loss_1 - loss_2).abs().item())
    print((grad_1 - grad_2).abs().max().item())
    print()

def simple_test():
    loss_func1 = bmt.loss.FusedCrossEntropy()
    loss_func2 = torch.nn.CrossEntropyLoss()

    N = 32 * 512
    for i in range(1, 10):
        C = i * 10
        x = torch.randn(N, C).cuda().half()
        tgt = torch.randint(0, C, (N,)).cuda().long()
        check(x, tgt, loss_func1, loss_func2)
    for i in range(1, 10):
        C = i * 100
        x = torch.randn(N, C).cuda().half()
        tgt = torch.randint(0, C, (N,)).cuda().long()
        check(x, tgt, loss_func1, loss_func2)
    for i in range(1, 31):
        C = i * 1000
        x = torch.randn(N, C).cuda().half()
        tgt = torch.randint(0, C, (N,)).cuda().long()
        check(x, tgt, loss_func1, loss_func2)

def test_other():
    N = 32 * 512
    for i in range(1, 11):
        C = i * 10
        weight = [i+1 for i in range(C)]
        random.shuffle(weight)
        weight = torch.tensor(weight, device="cuda")
        loss_func1 = bmt.loss.FusedCrossEntropy(weight=weight.clone().half())
        loss_func2 = torch.nn.CrossEntropyLoss(weight=weight.clone().float())

        x = torch.randn(N, C).cuda().half()
        tgt = torch.randint(0, C, (N,)).cuda().long()
        mask = torch.randint(0, 2, (N,)).cuda().bool()
        tgt[mask] = -100
        check(x, tgt, loss_func1, loss_func2)

def test_inplace():
    loss_func1 = bmt.loss.FusedCrossEntropy(inplace=True)
    loss_func2 = torch.nn.CrossEntropyLoss()
    N = 32 * 512

    for i in range(1, 11):
        C = i * 10
        bigmodel = torch.nn.Linear(5, C).cuda().half()
        x = torch.randn(N, 5).cuda().half()
        tgt = torch.randint(0, C, (N,)).cuda().long()
        check(x, tgt, loss_func1, loss_func2, bigmodel=bigmodel)


def benchmark_memory():
    N = 32 * 512
    C = 50000

    bigmodel = torch.nn.Linear(1, C).cuda().half()
    x = torch.randn(N, 1).cuda().half()
    tgt = torch.randint(0, C, (N,)).cuda().long()

    # loss_func = bmt.loss.FusedCrossEntropy(inplace=True)
    loss_func = bmt.loss.FusedCrossEntropy(inplace=False)
    # loss_func = torch.nn.CrossEntropyLoss()

    loss, grad = run(x, tgt, loss_func, bigmodel=bigmodel)
    print(loss, grad)

if __name__ == "__main__":
    test_other()
    test_inplace()
    simple_test()
    # benchmark_memory()