import torch
from ..functions.softmax import Softmax

def softmax(x : torch.Tensor) -> torch.Tensor:
    return Softmax.apply(x)