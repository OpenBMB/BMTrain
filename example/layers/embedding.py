import torch
import bmpretrain as bmp
from cpm_kernels.torch.embedding import OpEmbedding

class Embedding(bmp.DistributedModule):
    def __init__(self, vocab_size : int, embedding_size : int, dtype=torch.half):
        super().__init__()
        self.weight = bmp.DistributedParameter(torch.empty(vocab_size, embedding_size, dtype=dtype), init_method=bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=1))

    def forward(self, ids : torch.Tensor):
        """
        Args:
            ids : (batch_size, seq_len)                         int32
        Returns:
            embedding : (batch_size, embedding_size, seq_len)   fp16
        """
        return OpEmbedding.apply(ids, self.weight)
