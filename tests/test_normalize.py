import BMPretrain.layers as layers
import unittest
import torch

def normalize(x, rd_mean, eps=1e-5):
    old_dtype = x.dtype
    x = x.float()
    var = (x**2).mean(dim=-1, keepdim=True)
    if rd_mean:
        mean = x.mean(dim=-1, keepdim=True)
        var -= mean**2
        x = (x - mean) * torch.rsqrt(var + eps)
    else:
        x = x * torch.rsqrt(var + eps)
    x = x.to(old_dtype)
    return x

class TorchLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, bias, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        else:
            self.bias = None
        self.eps = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        hidden_states = self.weight * hidden_states
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
        return hidden_states

class TestNormalize(unittest.TestCase):
    def test_rdmean(self):
        with torch.cuda.device(0):
            for shape in [
                (1, 32),
                (4, 128),
                (16, 128),
                (4, 16, 128),
                (123, 321, 512),
                (123, 321, 768),
                (233, 1, 321, 1024),
                (4, 16, 4096),
            ]:
                x = torch.randn(*shape, device="cuda").half()
                x1 = x.clone().requires_grad_()
                x2 = x.requires_grad_()
                y1 = normalize(x1, rd_mean=True)
                y2 = layers.NormalizeOp.apply(x2, 1e-5, True)
                diff = (y1 - y2).abs()
                self.assertTrue(diff.max() < 5e-3)

                rd = torch.randn( *shape, device="cuda").half()
                (y1 * rd).mean().backward()
                (y2 * rd).mean().backward()
                
                diff_grad = (x1.grad - x2.grad).abs()
                self.assertTrue(diff_grad.max() < 5e-3)
    
    def test_nordmean(self):
        with torch.cuda.device(1):
            for shape in [
                (1, 32),
                (4, 128),
                (16, 128),
                (4, 16, 128),
                (123, 321, 512),
                (123, 321, 768),
                (233, 1, 321, 1024),
                (4, 16, 4096),
            ]:
                x = torch.randn(*shape, device="cuda").half()
                x1 = x.clone().requires_grad_()
                x2 = x.requires_grad_()
                y1 = normalize(x1, rd_mean=False)
                y2 = layers.NormalizeOp.apply(x2, 1e-5, False)
                diff = (y1 - y2).abs()
                self.assertTrue(diff.max() < 5e-3)

                rd = torch.randn( *shape, device="cuda").half()
                (y1 * rd).sum().backward()
                (y2 * rd).sum().backward()
                
                diff_grad = (x1.grad - x2.grad).abs()
                self.assertTrue(diff_grad.max() < 5e-3)
        
    def test_layernorm_unbias(self):
        with torch.cuda.device(2):
            for shape, eps in [
                (768, 1e-5),
                (768, 1e-6),
                (1024, 1e-3),
                (1024, 1e-6)
            ]:
                l1 = TorchLayerNorm(shape, False, eps)
                l2 = layers.LayerNorm(shape, False, False, eps)
                state_dict = {
                    "weight": torch.randn(shape),
                }
                l1.load_state_dict(state_dict)
                l2.load_state_dict(state_dict)
                                
                l1 = l1.to("cuda").half()
                l2 = l2.to("cuda").half()

                for _ in range(16):
                    x_raw = torch.randn((128, 512, shape), device="cuda").half()
                    x1 = x_raw.clone().requires_grad_()
                    x2 = x_raw.requires_grad_()
                    y1 = l1(x1)
                    y2 = l2(x2)

                    diff = (y1 - y2).abs()
                    self.assertTrue(diff.max() < 2e-2)

                    rd = torch.randn( x_raw.size(), device="cuda").half()
                    (y1 * rd).sum().backward()
                    (y2 * rd).sum().backward()
                    
                    diff_grad = (x1.grad - x2.grad).abs()
                    self.assertTrue(diff_grad.max() < 5e-2)
    
    def test_layernorm_bias(self):
        with torch.cuda.device(3):
            for shape, eps in [
                (768, 1e-5),
                (768, 1e-6),
                (1024, 1e-3),
                (1024, 1e-6)
            ]:
                l1 = TorchLayerNorm(shape, True, eps)
                l2 = layers.LayerNorm(shape, True, False, eps)
                state_dict = {
                    "weight": torch.randn(shape),
                    "bias": torch.randn(shape)
                }
                l1.load_state_dict(state_dict)
                l2.load_state_dict(state_dict)
                
                l1 = l1.to("cuda").half()
                l2 = l2.to("cuda").half()

                for _ in range(16):
                    x_raw = torch.randn((123, 312, shape), device="cuda").half()
                    x1 = x_raw.clone().requires_grad_()
                    x2 = x_raw.requires_grad_()
                    y1 = l1(x1)
                    y2 = l2(x2)

                    diff = (y1 - y2).abs()
                    self.assertTrue(diff.max() < 2e-2)

                    rd = torch.randn( x_raw.size(), device="cuda").half()
                    (y1 * rd).sum().backward()
                    (y2 * rd).sum().backward()
                    
                    diff_grad = (x1.grad - x2.grad).abs()
                    self.assertTrue(diff_grad.max() < 5e-2)
    