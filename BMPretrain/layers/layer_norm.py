import torch
from ..functions.normalize import NormalizeOp

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, bias=False, rd_mean=False, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        else:
            self.bias = None
        self.rd_mean = rd_mean
        self.eps = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        normed_out = NormalizeOp.apply(hidden_states, self.eps, self.rd_mean)
        # convert into float16 if necessary
        normed_out = normed_out * self.weight
        if self.bias is not None:
            normed_out = normed_out + self.bias
        return normed_out