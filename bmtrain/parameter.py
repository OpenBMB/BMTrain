from typing import Callable, Iterable, Optional
import torch
from .utils import round_up
from .global_var import config
from . import nccl

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
    
    _original_shape : torch.Size
    _start_partition : int
    _end_partition : int
    _init_method : Optional[Callable[['DistributedParameter'], None]]
    _in_checkpoint_block : bool
    _group : Optional[str]

    def __new__(cls,
            data : torch.Tensor, 
            requires_grad : bool = True, 
            init_method : Optional[Callable[['DistributedParameter'], None]] = None,
            group : Optional[str] = None
        ):
        num_of_elements = data.numel()

        cuda_tensor = torch.tensor([], dtype=data.dtype, device="cuda") 
        cuda_storage_size = round_up(num_of_elements, config["world_size"]) // config["world_size"]

        original_shape = data.size()

        cuda_storage = cuda_tensor.storage_type()(cuda_storage_size)

        start_of_partition = cuda_storage_size * config["rank"]
        end_of_partition = min(num_of_elements, cuda_storage_size * (config["rank"] + 1))

        # FX: cuda_tensor_size < 0 if num_of_elements is too small
        cuda_tensor_size = max(end_of_partition - start_of_partition, 0)

        cuda_tensor.set_(cuda_storage, 0, (cuda_tensor_size,))
        cuda_tensor.copy_(data.view(-1)[start_of_partition: end_of_partition])
        ret = torch.Tensor._make_subclass(cls, cuda_tensor, requires_grad)
        
        setattr(ret, "_original_shape", original_shape)
        setattr(ret, "_start_partition", start_of_partition)
        setattr(ret, "_end_partition", end_of_partition)
        setattr(ret, "_init_method", init_method)
        setattr(ret, "_in_checkpoint_block", False)
        setattr(ret, "_group", group)
        return ret
    
    @property
    def group(self):
        """The group name of the distributed parameter."""

        return self._group

    def gather(self) -> torch.Tensor:
        """Gather the data from all the distributed nodes.

        Return:
            torch.Tensor: The gathered data.
        
        """
        with torch.cuda.stream(config['load_stream']):
            output_tensor = OpAllGather.apply(self)
        current_stream = torch.cuda.current_stream()
        output_tensor.record_stream( current_stream )
        current_stream.wait_stream(config['load_stream'])
        return output_tensor

    def _copy_data(self, data : torch.Tensor):
        self.data.copy_(data.view(-1)[self._start_partition : self._end_partition])
    

class OpAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value : DistributedParameter):
        assert isinstance(value, DistributedParameter)

        partition_size = value.storage().size()
        global_size = partition_size * config['world_size']

        storage = value.storage_type()(global_size)
        
        nccl.allGather(
            value.storage(),
            storage,
            config['comm']
        )

        output_tensor = torch.tensor([], dtype=value.dtype, device="cuda")
        output_tensor.set_(storage, 0, value._original_shape)
    
        ctx.partition_size = partition_size
        ctx.tensor_size = value.size(0)
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_storage = grad_output.storage_type()(ctx.partition_size)
        grad_output_storage = grad_output.storage()
        if grad_output_storage.size() == ctx.partition_size * config['world_size']:
            pass
        else:
            grad_output_storage.resize_(ctx.partition_size * config['world_size'])
        nccl.reduceScatter(
            grad_output_storage,
            grad_storage,
            'sum',
            config['comm']
        )
        grad_tensor = torch.tensor([], dtype=grad_output.dtype, device="cuda")
        grad_tensor.set_(grad_storage, 0, (ctx.tensor_size,))
        return grad_tensor

class ParameterInitializer:
    """
    ParameterInitializer is a helper class that is used to initialize the distributed parameters.

    Similar to functools.partial .

    """
    def __init__(self, func : Callable, *args, **kwargs) -> None:
        self.func = func
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, param : DistributedParameter):
        self.func(param, *self._args, **self._kwargs)
