import torch
import bmtrain as bmt

def main():
    loss_func1 = bmt.loss.FusedCrossEntropy()
    loss_func2 = torch.nn.CrossEntropyLoss()

    N = 32 * 512
    C = 30000

    for _ in range(32):
        x = torch.randn(N, C).cuda().half()
        x1 = x.clone().requires_grad_()
        x2 = x.clone().requires_grad_()
        tgt = torch.randint(0, C, (N,)).cuda().long()
        
        loss_1 = loss_func1(x1, tgt)
        loss_2 = loss_func2(x2, tgt)
        print((loss_1 - loss_2).abs())
        (loss_1 * 32768).backward()
        (loss_2 * 32768).backward()
        print(x1.grad)
        print((x1.grad - x2.grad).abs().max())

if __name__ == "__main__":
    main()