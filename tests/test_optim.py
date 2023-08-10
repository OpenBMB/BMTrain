from utils import *
import torch
import bmtrain as bmt

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, 128)
        self.param = torch.nn.Parameter(torch.empty(1237))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

def main(dtype):
    model1 = TestModule()
    model2 = TestModule()
    model3 = TestModule()

    state_dict = model1.state_dict()
    for kw in state_dict.keys():
        state_dict[kw] = torch.randn_like(state_dict[kw])

    model1.load_state_dict(state_dict)
    model2.load_state_dict(state_dict)
    model3.load_state_dict(state_dict)

    model1 = model1.cuda().to(dtype)
    model2 = model2.cuda().to(dtype)
    model3 = model3.cuda()

    opt1 = bmt.optim.AdamOptimizer(model1.parameters(), lr=1)
    opt2 = bmt.optim.AdamOffloadOptimizer(model2.parameters(), lr=1)
    opt3 = torch.optim.Adam(model3.parameters(), lr=1)

    for _ in range(100):
        opt1.zero_grad()
        opt2.zero_grad()
        opt3.zero_grad()

        for p1, p2, p3 in zip(model1.parameters(), model2.parameters(), model3.parameters()):
            grad = torch.randn_like(p1)
            p1.grad = grad.to(dtype)
            p2.grad = grad.to(dtype)
            p3.grad = grad.float()

        opt1.step()
        opt2.step()
        opt3.step()

        for p1, p2, p3 in zip(model1.parameters(), model2.parameters(), model3.parameters()):
            diff1 = torch.abs(p1 - p2).max().item() 
            diff2 = torch.abs(p1 - p3).max().item()
            diff3 = torch.abs(p2 - p3).max().item()
            print(f"{diff1:4.6f}, {diff2:4.6f}, {diff3:4.6f}")
            assert_lt(diff1, 1)
            assert_lt(diff2, 1)
            assert_lt(diff3, 1)

if __name__ == "__main__":
    main(torch.float16)
    main(torch.bfloat16)
