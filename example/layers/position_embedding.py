import torch
import bmpretrain as bmp
from cpm_kernels.torch.position_embedding import OpPositionEmbedding

class PositionEmbedding(bmp.DistributedModule):
    def __init__(self, num_heads, num_buckets, max_distance, bidirectional=True, dtype=torch.half):
        super().__init__()
        self.weight = bmp.DistributedParameter(torch.randn(num_heads, num_buckets, dtype=dtype), init_method=bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=0.02))

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional


    def forward(self, key_len, query_len):
        """
        Args:
            key_len: int
            query_len : int
        Returns:
            out : (num_heads, key_len, query_len)   fp16
        """
        return OpPositionEmbedding.apply(query_len, key_len, self.num_buckets, self.max_distance, self.num_heads, self.weight, self.bidirectional)

