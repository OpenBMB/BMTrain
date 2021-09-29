import torch
import torch.nn.functional as F
import BMPretrain.layers as layers
import sys
import unittest


@torch.jit.script
def calc_scale(x):
    return x.abs().max(dim=-1, keepdim=True)[0] / 127

@torch.jit.script
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

def real_quantized_linear(x : torch.Tensor, weight : torch.Tensor):
    return F.linear(ScaleOnlyFunc.apply(x), ScaleOnlyFunc.apply(weight))
    
TEST_CASES = [
    (1, 4, 4, 4, torch.half),
    (2, 4, 4, 4, torch.half),
    (1, 8, 8, 8, torch.half),
    (1, 16, 16, 16, torch.float),
    (1, 32, 32, 32, torch.float),
    (1, 64, 64, 64, torch.double),
    (1, 128, 128, 128, torch.double),
    (4, 128, 128, 128, torch.float),
    (8, 128, 128, 128, torch.double),
    (32, 512, 4096, 2048, torch.float),
    (32, 512, 4096, 8192, torch.half),
    (32, 512, 4096, 2048, torch.half),
    (32, 512, 10240, 4096, torch.half),
    (32, 1024, 512, 4096, torch.half),
]

class TestGEMM(unittest.TestCase):
    def test_gemm_forward(self):
        with torch.cuda.device(0):
            for args in TEST_CASES[::-1]:
                batch, m, k, n, dtype = args
                if dtype == torch.float32:
                    threshold = 5e-3
                elif dtype == torch.float16:
                    threshold = 0.1
                elif dtype == torch.float64:
                    threshold = 5e-5
                else:
                    raise RuntimeError("Unknown dtype %s" % dtype)

                a = torch.randn(batch, m, k, device="cuda").to(dtype)
                b = torch.randn(n, k, device="cuda").to(dtype)
                out = layers.simple_quantized_linear(a, b) / k
                ans = real_quantized_linear(a, b) / k
                diff = (out - ans).abs().max()
                self.assertLess(diff, threshold, "diff is too big : %lf" % diff)

    def test_gemm_backward(self):
        with torch.cuda.device(1):
            for args in TEST_CASES[::-1]:
                batch, m, k, n, dtype = args

                if dtype == torch.float32:
                    threshold = 0.1
                elif dtype == torch.float16:
                    threshold = 0.5
                elif dtype == torch.float64:
                    threshold = 0.05
                else:
                    raise RuntimeError("Unknown dtype %s" % dtype)
                
                a1 = torch.randn(batch, m, k, device="cuda").to(dtype)
                b1 = torch.randn(n, k, device="cuda").to(dtype)

                a2 = a1.clone()
                b2 = b1.clone()

                a1.requires_grad_(); b1.requires_grad_()
                a2.requires_grad_(); b2.requires_grad_()

                out = layers.simple_quantized_linear(a1, b1) / k
                ans = real_quantized_linear(a2, b2) / k

                (out ** 2).sum().backward()
                (ans ** 2).sum().backward()

                diff1 = (a1.grad - a2.grad).abs().max()
                diff2 = (b1.grad - b2.grad).abs().max()
                diff = torch.max(diff1, diff2)
                self.assertLess(diff, threshold, "diff is too big : %lf" % diff)