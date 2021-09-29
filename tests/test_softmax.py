import BMPretrain.layers as layers
import unittest
import torch


@torch.jit.script
def softmax(x):
    mx_x = torch.max(x, dim=-1, keepdim=True)[0]
    x = x - mx_x
    x = torch.exp(x)
    x = x / torch.sum(x, dim=-1, keepdim=True)
    return x

class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        with torch.cuda.device(0):
            for shape in [
                (1, 32),
                (4, 128),
                (16, 128),
                (4, 16, 128),
                (123, 321, 512),
                (123, 321, 768),
                (233, 1, 321, 1024),
                (4, 16, 123),
                (4, 16, 321),
            ]:
                x = torch.randn(*shape, device="cuda").half()
                x1 = x.clone().requires_grad_()
                x2 = x.requires_grad_()
                y1 = softmax(x1)
                y2 = layers.Softmax.apply(x2)
                diff = (y1 - y2).abs()
                self.assertTrue(diff.max() < 5e-3)

                rd = torch.randn( *shape, device="cuda").half()
                (y1 * rd).mean().backward()
                (y2 * rd).mean().backward()
                
                diff_grad = (x1.grad - x2.grad).abs()
                self.assertTrue(diff_grad.max() < 5e-3)
    