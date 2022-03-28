import bmtrain as bmt
import torch

def main():
    bmt.init_distributed()
    x = torch.full((1,), bmt.rank() + 1, dtype=torch.half, device="cuda").requires_grad_(True)
    y = bmt.distributed.all_reduce(x, "prod").view(-1)
    bmt.print_rank(y)
    loss = (y * y).sum() / 2
    loss.backward()
    print(x.grad)


if __name__ == "__main__":
    main()