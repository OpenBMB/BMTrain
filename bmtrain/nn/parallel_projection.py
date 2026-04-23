import torch
import bmtrain as bmt
from bmtrain.global_var import config
from .linear import OpLinear
from .parallel_linear_func import OpParallelLinear


class Projection(bmt.DistributedModule):
    """Output projection: linear map from hidden size to full vocabulary (reference / non-TP).

    Args:
        vocab_size: number of classes / vocabulary size.
        embedding_size: hidden dimension (input features).
        dtype: parameter dtype.
        init_mean, init_std: arguments for :func:`torch.nn.init.normal_` on the full weight matrix.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1.0,
    ):
        super().__init__()
        self.dim_model = embedding_size
        self.vocab_size = vocab_size
        self.weight = bmt.DistributedParameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),
            init_method=bmt.ParameterInitializer(
                torch.nn.init.normal_, mean=init_mean, std=init_std
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return OpLinear.apply(x, self.weight, None)


class VPProjection(bmt.DistributedModule):
    """Vocabulary-parallel output projection (weight sharded on dim 0).

    Each rank accepts a slice of the hidden dimension; tensors are gathered before linear.
    Matches :class:`VPEmbedding` forward with ``projection=True``.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1.0,
    ):
        super().__init__()
        assert vocab_size % config["tp_size"] == 0
        self.dim_model = embedding_size
        self.vocab_size_per_partition = vocab_size // config["tp_size"]
        self.weight = bmt.DistributedParameter(
            torch.empty(self.vocab_size_per_partition, embedding_size, dtype=dtype),
            init_method=bmt.ParameterInitializer(
                torch.nn.init.normal_, mean=init_mean, std=init_std
            ),
            tp_split_dim=0,
            tp_mode=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each rank holds a slice of the last (hidden) dim. all_gather returns (tp_size, *x.shape);
        # permute so batch leads, then reshape to full embedding_size on the last dim for F.linear.
        shape = x.shape
        g = bmt.distributed.all_gather(x, comm=config["tp_comm"])
        x = g.permute(1, 0, *range(2, g.ndim)).reshape(*shape[:-1], -1)
        return OpParallelLinear.apply(
            x, self.weight, None, False, False, False, None, 1
        )
