import torch
import bmtrain as bmt

class InnerModule(bmt.DistributedModule):
    def __init__(self):
        super().__init__()

        self.drop = torch.nn.Dropout(p=0.5)
    
    def forward(self, x):
        bmt.print_rank(x)
        return self.drop(x)

class OutterModule(bmt.DistributedModule):
    def __init__(self) -> None:
        super().__init__()

        self.blk = bmt.TransformerBlockList([
            bmt.CheckpointBlock(InnerModule())
            for _ in range(5)
        ])
    
    def forward(self, x):
        bmt.print_rank(self.training)
        return self.blk(x)

def main():
    bmt.init_distributed()

    model = OutterModule()

    x = torch.ones(32, device="cuda")
    y = model(x)
    bmt.print_rank(y)

if __name__ == "__main__":
    main()