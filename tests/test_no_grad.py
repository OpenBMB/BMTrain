import torch
import bmtrain as bmt

class Layer(torch.nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.linear = bmt.nn.Linear(32, 32)
        self.count = 0
    def forward(self, x):
        self.count += 1
        return self.linear(x)

def test_no_grad():
    x = torch.randn(32, 32, device='cuda')

    layer1 = bmt.Block(Layer())
    layer2 = bmt.Block(Layer())
    layer1.linear.weight.requires_grad_(False)
    layer1.linear.bias.requires_grad_(False)
    y = layer1(x)
    assert y.requires_grad == False
    y = layer2(y)
    y.sum().backward()
    assert layer1.count == 1
    assert layer2.count == 2

def test_all_input_no_grad():
    linear1 = bmt.nn.Linear(32, 32)
    linear2 = bmt.nn.Linear(32, 32)

    x = torch.randn(32,32, device='cuda')

    linear1 = bmt.Block(linear1)
    linear2 = bmt.Block(linear2)
    y = linear1(x)
    y = linear2(y)
    y.sum().backward()
    assert linear1.weight.grad is not None
    assert linear1.bias.grad is not None
    assert x.grad is None

if __name__ == '__main__':
    bmt.init_distributed()

    test_no_grad()
    test_all_input_no_grad()
