import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

import bmtrain as bmt
from bmtrain.global_var import config
from bmtrain.distributed import all_reduce, all_gather
from .parallel_linear_func import ParallelLinearFunc 

class ParallelEmbedding(bmt.DistributedModule):
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
        assert vocab_size % config['tp_size'] == 0
        self.vocab_size_per_partition = vocab_size // config['tp_size']
        self.start_index = config['topology'].tp_id * self.vocab_size_per_partition
        self.end_index = (config['topology'].tp_id+1) * self.vocab_size_per_partition
        self.weight = bmt.DistributedParameter(
            torch.empty(self.vocab_size_per_partition, embedding_size, dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            tp_mode=True,
            tp_split_dim=0,
        )

    def forward(self, ids: torch.Tensor, gather_input=True):
        """
        Args:
            ids (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            gather_input (bool) : whether gather input is required between  tensor parallel group)
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        if config['tp_size'] > 1:
            if gather_input:
                ids = all_gather(ids, comm=config['tp_comm'])
            input_mask = (ids < self.start_index) | (ids >= self.end_index) 
            ids = ids.clone() - self.start_index
            ids[input_mask] = 0 

        embeds = F.embedding(ids, self.weight)

        if config['tp_size'] > 1:
            embeds[input_mask, :] = 0.0
            embeds = all_reduce(embeds, op="sum", comm=config['tp_comm'])
            embed_list = embeds.chunk(config['tp_size'], dim=0)
            embeds = embed_list[config['topology'].tp_id].flatten(0,1)
            
        return embeds.clone()

    def projection(self, x: torch.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501
        gather_input = True
        split_input = False
        reduce_output_type = None 
        gather_output = False 
        out = ParallelLinearFunc.apply(x , self.weight, None, gather_input, gather_output, split_input, reduce_output_type)
        return out
