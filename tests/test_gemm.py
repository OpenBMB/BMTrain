import torch
import torch.nn.functional as F
import BMPretrain.layers as layers
import random, math
import unittest



def calc_scale(x):
    return x.abs().max(dim=-1, keepdim=True)[0] / 127


def scale_round(x, scale):
    return torch.round(x / scale) * scale

class ScaleOnlyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x : torch.Tensor):
        scale = calc_scale(x)
        return scale_round(x, scale)
    
    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        return grad_outputs

def real_bmm(a, aT, b, bT, int8):
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)

    if aT:
        a = a.transpose(-1, -2)
    if bT:
        b = b.transpose(-1, -2)
    if int8:
        old_type = a.dtype
        a = ScaleOnlyFunc.apply(a.to(torch.float32))
        b = ScaleOnlyFunc.apply(b.to(torch.float32))
        return torch.matmul(a, b).to(old_type)
    else:
        return torch.matmul(a, b)
    
TEST_CASES = [
    (1, 4, 4, 4, torch.half),
    (2, 4, 4, 4, torch.half),
    (1, 8, 8, 8, torch.half),
    (1, 16, 16, 16, torch.float),
    (1, 32, 32, 32, torch.float),
    (1, 64, 64, 64, torch.float),
    (1, 128, 128, 128, torch.float),
    (4, 128, 128, 128, torch.float),
    (8, 128, 128, 128, torch.float),
    (8, 512, 4096, 2048, torch.float),
    (8, 512, 4096, 8192, torch.half),
    (8, 512, 4096, 2048, torch.half),
    (8, 512, 10240, 4096, torch.half),
    (8, 1024, 512, 4096, torch.half),
]

def generate_matrix(batch, m, k, n, dtype):
    tp = random.randint(0, 2)
    if tp == 0:
        a = torch.randn(batch, m, k, device="cuda").to(dtype) / math.sqrt(k)
        b = torch.randn(batch, k, n, device="cuda").to(dtype) / math.sqrt(k)
    elif tp == 1:
        a = torch.randn(1, m, k, device="cuda").to(dtype) / math.sqrt(k)
        b = torch.randn(batch, k, n, device="cuda").to(dtype) / math.sqrt(k)
    else:
        a = torch.randn(batch, m, k, device="cuda").to(dtype) / math.sqrt(k)
        b = torch.randn(1, k, n, device="cuda").to(dtype) / math.sqrt(k)
    if random.randint(0, 1) == 1:
        a = a.transpose(-1, -2).contiguous()
        aT = True
    else:
        aT = False
    if random.randint(0, 1) == 1:
        b = b.transpose(-1, -2).contiguous()
        bT = True
    else:
        bT = False

    return a, aT, b, bT

class TestGEMM(unittest.TestCase):
    def test_gemm_forward(self):
        with torch.cuda.device(0):
            for args in TEST_CASES[::-1]:
                batch, m, k, n, dtype = args
                if dtype == torch.float32:
                    threshold = 0.01
                elif dtype == torch.float16:
                    threshold = 0.1
                else:
                    raise RuntimeError("Unknown dtype %s" % dtype)

                for _ in range(10):
                    a, aT, b, bT = generate_matrix(batch, m, k, n, dtype)
                    int8 = random.randint(0, 1) == 1
                    out = layers.bmm(a, aT, b, bT, int8)
                    ans = real_bmm(a, aT, b, bT, int8)
                    diff = (out - ans).abs().max()
                    self.assertLess(diff, threshold, "diff is too big : %lf" % diff)

    def test_gemm_backward(self):
        with torch.cuda.device(1):
            for args in TEST_CASES:
                batch, m, k, n, dtype = args

                if dtype == torch.float32:
                    threshold = 0.1
                elif dtype == torch.float16:
                    threshold = 0.5
                else:
                    raise RuntimeError("Unknown dtype %s" % dtype)
                
                for _ in range(10):
                    a, aT, b, bT = generate_matrix(batch, m, k, n, dtype)
                    int8 = random.randint(0, 1) == 1

                    a1 = a.clone().detach(); a2 = a
                    b1 = b.clone().detach(); b2 = b

                    a1.requires_grad_(); a2.requires_grad_()
                    b1.requires_grad_(); b2.requires_grad_()

                    ans = real_bmm(a2, aT, b2, bT, int8)
                    out = layers.bmm(a1, aT, b1, bT, int8)
                    diff = (out - ans).abs()
                    (out ** 2).sum().backward()
                    (ans ** 2).sum().backward()

                    diff1 = (a1.grad - a2.grad).abs().max()
                    diff2 = (b1.grad - b2.grad).abs().max()
                    diff = torch.max(diff1, diff2)
                    self.assertLess(diff, threshold, "diff is too big : %lf" % diff)
