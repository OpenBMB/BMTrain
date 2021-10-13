import torch
import BMPretrain._c as C
import math

class PositionBiasEmbedding(torch.nn.Module):
    """Position bias embedding.

    This is used to add bias to the positional embedding.
    """

    def __init__(self, num_buckets, num_heads, max_distance, bidirectional=True):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        self.embedding = torch.nn.Embedding(self.num_buckets, self.num_heads)

    def compute_bias(self, query_length, key_length):
        """ Compute binned relative position bias """
        # device = self.embedding.weight.device
        relative_position_bucket = C.position_bucket(query_length, key_length, self.num_buckets, self.max_distance, self.bidirectional)
        relative_position_bucket = relative_position_bucket.to()
        values = self.embedding(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values
    
    def forward(self, query_length, key_length):
        return self.compute_bias(query_length, key_length)