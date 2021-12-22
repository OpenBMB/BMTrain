import torch
from ..layer import DistributedModule
from ..parameter import DistributedParameter, ParameterInitializer
import torch.nn.functional as F

class Embedding(DistributedModule):
    def __init__(self, vocab_size : int, embedding_size : int, dtype=torch.half):
        super().__init__()
        self.weight = DistributedParameter(torch.empty(vocab_size, embedding_size, dtype=dtype), init_method=ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=1))

    def forward(self, ids : torch.Tensor):
        return F.embedding(
            ids,
            self.weight
        )

    def projection(self, hidden : torch.Tensor):
        return F.linear(
            hidden,
            self.weight
        )