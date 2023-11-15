import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

import bmtrain as bmt
from bmtrain.global_var import config
from bmtrain.distributed import all_reduce, all_gather
from .parallel_linear_func import OpParallelLinear 

class Projection(bmt.DistributedModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
    ):
        super().__init__()

        self.dim_model = embedding_size
        self.weight = bmt.DistributedParameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )

    def projection(self, x: torch.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501
        logits = F.linear(x, self.weight)
        return logits

class VPProjection(bmt.DistributedModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
    ):
        super().__init__()

        self.dim_model = embedding_size
        assert vocab_size % bmt.config["tp_size"] == 0
        self.vocab_size_per_partition = vocab_size // bmt.config["tp_size"]
        self.start_index = bmt.config["tp_rank"] * self.vocab_size_per_partition
        self.end_index = (bmt.config["tp_rank"] + 1) * self.vocab_size_per_partition
        self.weight = bmt.DistributedParameter(
            torch.empty(self.vocab_size_per_partition, embedding_size, dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            tp_split_dim=0,
            tp_mode=True,
        )

    def projection(self, x: torch.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501
        return bmt.nn.OpParallelLinear.apply(x, self.weight, None, False, False, False, None, 1)