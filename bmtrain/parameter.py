from typing import Callable, Iterable, Optional
import torch
from .utils import round_up
from .global_var import config
from . import nccl
from .distributed import all_gather


class DistributedParameter(torch.nn.Parameter):
    r"""
    DistributedParameter is a subclass of torch.nn.Parameter.

    It scatters the tensor to all the nodes and gathers them when needed.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient.
        init_method (Callable[['DistributedParameter'], None], optional): the method to initialize the parameter.
        group (str, optional): the group name of the parameter.

    **Note**: DistributedParameter must be on the CUDA device. It will transfer the data to device automatically when `__init__` called.

    """

    _original_shape: torch.Size
    _start_partition: int
    _end_partition: int
    _init_method: Optional[Callable[["DistributedParameter"], None]]
    _in_block: bool
    _group: Optional[str]

    def __new__(
        cls,
        data: torch.Tensor,
        requires_grad: bool = True,
        init_method: Optional[Callable[["DistributedParameter"], None]] = None,
        group: Optional[str] = None,
        tp_mode: bool = False,
        tp_split_dim: int = -1,
    ):
        if not config["initialized"]:
            raise RuntimeError("BMTrain is not initialized")

        num_of_elements = data.numel()

        if tp_mode:
            comm = config["tp_zero_comm"]
        else:
            comm = config["zero_comm"]
        world_size = nccl.commCount(comm)
        rank = nccl.commRank(comm)
        cuda_storage_size = round_up(num_of_elements, world_size) // world_size

        original_shape = data.size()
        tp_original_shape = original_shape
        if tp_mode and tp_split_dim >= 0:
            tp_original_shape = list(original_shape)
            tp_original_shape[tp_split_dim] *= config["tp_size"]

        storage_tensor = torch.empty(cuda_storage_size, dtype=data.dtype, device="cuda")

        start_of_partition = cuda_storage_size * rank
        end_of_partition = min(num_of_elements, cuda_storage_size * (rank + 1))

        cuda_tensor_size = max(end_of_partition - start_of_partition, 0)

        cuda_tensor = storage_tensor[:cuda_tensor_size]
        # Detach, flatten, clone on CUDA so we read plain tensor storage. Copying from
        # `nn.Parameter(...).view(-1)[...]` alone can yield zeros for the common pattern
        # `self.weight = nn.Parameter(torch.empty(...)); init_(self.weight)` (see tests).
        with torch.no_grad():
            src = data.detach()
            if src.device.type != "cuda":
                src = src.cuda()
            src_flat = src.reshape(-1).contiguous().clone()
        cuda_tensor.copy_(src_flat[start_of_partition:end_of_partition])
        ret = torch.Tensor._make_subclass(cls, cuda_tensor, requires_grad)

        setattr(ret, "_original_shape", original_shape)
        setattr(ret, "_start_partition", start_of_partition)
        setattr(ret, "_end_partition", end_of_partition)
        setattr(ret, "_init_method", init_method)
        setattr(ret, "_in_block", False)
        setattr(ret, "_group", group if not tp_mode else "tp")

        setattr(ret, "_tp_mode", tp_mode)
        setattr(ret, "_zero_comm", comm)
        setattr(ret, "_tp_split_dim", tp_split_dim)
        setattr(ret, "_tp_original_shape", tp_original_shape)
        setattr(ret, "_partition_size", cuda_storage_size)
        return ret

    @property
    def group(self):
        """The group name of the distributed parameter."""

        return self._group

    def gather(self) -> torch.Tensor:
        """Gather the data from ZeRO distributed nodes.

        Return:
            torch.Tensor: The gathered data.

        """
        with torch.cuda.stream(config["load_stream"]):
            output_tensor = OpAllGather.apply(self)
        current_stream = torch.cuda.current_stream()
        output_tensor.record_stream(current_stream)
        current_stream.wait_stream(config["load_stream"])
        return output_tensor

    def gather_all(self) -> torch.Tensor:
        """Gather the data from ZeRO and Tensor Parallel distributed nodes.

        Return:
            torch.Tensor: The gathered data.

        """
        zero_param = self.gather()
        if config["tp_size"] > 1 and self._tp_split_dim >= 0:
            output_tensor = all_gather(zero_param, config["tp_comm"])
            if self._tp_split_dim == 1:
                output_list = output_tensor.chunk(config["tp_size"], dim=0)
                output = torch.cat(output_list, dim=output_list[0].dim() - 1).flatten(
                    0, 1
                )
                return output
            else:
                return output_tensor.flatten(0, 1)
        else:
            return zero_param

    def tp_gather(self) -> torch.Tensor:
        """Gather the data from Tensor Parallel distributed nodes.

        Return:
            torch.Tensor: The gathered data.

        """
        if config["tp_size"] > 1 and self._tp_split_dim >= 0:
            output_tensor = all_gather(self, config["tp_comm"])
            if self._tp_split_dim == 1:
                output_list = output_tensor.chunk(config["tp_size"], dim=0)
                output = torch.cat(output_list, dim=output_list[0].dim() - 1).flatten(
                    0, 1
                )
                return output
            else:
                return output_tensor.flatten(0, 1)
        else:
            return self

    def _copy_data(self, data: torch.Tensor):
        """Copy data to self.data.

        Detach → move to CUDA → flatten → clone to ensure we read from a
        materialized contiguous buffer, avoiding stale-storage issues when the
        source comes from nn.Parameter wrappers.
        """
        with torch.no_grad():
            src = data.detach()
            if src.device.type != "cuda":
                src = src.cuda()
            flat = src.reshape(-1).contiguous().clone()
        self.data.copy_(flat[self._start_partition : self._end_partition])


class OpAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: DistributedParameter):
        assert isinstance(value, DistributedParameter)
        comm = value._zero_comm
        world_size = nccl.commCount(comm)
        ctx.comm = comm
        ctx.world_size = world_size

        partition_size = value._partition_size
        global_size = partition_size * world_size

        global_buffer = torch.empty(global_size, dtype=value.dtype, device="cuda")
        # Pad local data to partition_size; value.numel() may be smaller on the last rank.
        local_buffer = torch.empty(partition_size, dtype=value.dtype, device="cuda")
        local_buffer[:value.numel()].copy_(value)

        nccl.allGather(local_buffer, global_buffer, comm)

        output_tensor = global_buffer[:value._original_shape.numel()].view(value._original_shape)

        ctx.partition_size = partition_size
        ctx.tensor_size = value.size(0)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        expected_size = ctx.partition_size * ctx.world_size
        # Pad or truncate grad to match the allGather buffer size (partition_size * world_size),
        # because the original tensor may have been smaller due to non-divisible partitioning.
        grad_flat = grad_output.reshape(-1)
        if grad_flat.numel() < expected_size:
            padded = torch.zeros(expected_size, dtype=grad_output.dtype, device=grad_output.device)
            padded[:grad_flat.numel()].copy_(grad_flat)
            grad_flat = padded
        elif grad_flat.numel() > expected_size:
            grad_flat = grad_flat[:expected_size].contiguous()

        grad_partition = torch.empty(ctx.partition_size, dtype=grad_output.dtype, device=grad_output.device)
        nccl.reduceScatter(grad_flat, grad_partition, "sum", ctx.comm)
        return grad_partition[:ctx.tensor_size]


class ParameterInitializer:
    """
    ParameterInitializer is a helper class that is used to initialize the distributed parameters.

    Similar to functools.partial .

    """

    def __init__(self, func: Callable, *args, **kwargs) -> None:
        self.func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self, param: DistributedParameter):
        self.func(param, *self._args, **self._kwargs)
