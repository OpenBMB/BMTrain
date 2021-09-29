import BMPretrain.layers as layers
import unittest
import torch
import math


class TorchRelativePositionEmbedding(torch.nn.Module):
    def __init__(self, num_buckets, num_heads, max_distance, bidirectional=True) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        self.embedding = torch.nn.Embedding(self.num_buckets, self.num_heads)

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """ Compute binned relative position bias """
        context_position = torch.arange(query_length, dtype=torch.long, device="cuda")[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device="cuda")[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self.relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.embedding(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values
    
    def forward(self, query_length, key_length):
        return self.compute_bias(query_length, key_length)

class TestNormalize(unittest.TestCase):
    def test_position_bias(self):

        for args in [
            (32, 12, 128, False),
            (32, 12, 128, True),
            (32, 24, 256, True),
            (128, 64, 128, True),
            (16, 64, 256, False),
            (16, 16, 512, True),
        ]:
            l1 = TorchRelativePositionEmbedding(*args)
            l2 = layers.PositionBiasEmbedding(*args)

            state_dict = {
                'embedding.weight': torch.randn(args[0], args[1])
            }
            
            l1.load_state_dict(state_dict)
            l2.load_state_dict(state_dict)

            l1 = l1.cuda()
            l2 = l2.cuda()

            a = l1(128, 128)
            torch.cuda.synchronize()
            b = l2(128, 128)
            diff = (a - b).abs()
            self.assertTrue(diff.max() < 1e-6)
    