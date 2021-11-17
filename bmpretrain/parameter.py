import torch
from .utils import round_up, print_rank
from .global_var import config
from . import nccl

class DistributedParameter(torch.nn.Parameter):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`locally-disable-grad-doc` for more details. Default: `True`
    """
    
    _original_shape : torch.Size
    _start_partition : int
    _end_partition : int

    def __new__(cls, data : torch.Tensor, requires_grad=True):
        num_of_elements = data.numel()

        cuda_tensor = torch.tensor([], dtype=data.dtype, device="cuda") 
        cuda_storage_size = round_up(num_of_elements, config["world_size"]) // config["world_size"]

        original_shape = data.size()

        cuda_storage = cuda_tensor.storage_type()(cuda_storage_size)

        start_of_partition = cuda_storage_size * config["rank"]
        end_of_partition = min(num_of_elements, cuda_storage_size * (config["rank"] + 1))
        cuda_tensor_size = end_of_partition - start_of_partition

        cuda_tensor.set_(cuda_storage, 0, (cuda_tensor_size,))
        cuda_tensor.copy_(data.view(-1)[start_of_partition: end_of_partition])
        ret = torch.Tensor._make_subclass(cls, cuda_tensor, requires_grad)
        
        setattr(ret, "_original_shape", original_shape)
        setattr(ret, "_start_partition", start_of_partition)
        setattr(ret, "_end_partition", end_of_partition)
        return ret

    def gather(self) -> torch.Tensor:
        return OpAllGather.apply(self)
    

    def _copy_data(self, data : torch.Tensor):
        self.data.copy_(data.view(-1)[self._start_partition : self._end_partition])
    

class OpAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value : DistributedParameter):
        assert isinstance(value, DistributedParameter)

        current_stream = torch.cuda.current_stream()
        with torch.cuda.stream(config['load_stream']):
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
        output_tensor.record_stream( current_stream )
        current_stream.wait_stream(config['load_stream'])
        ctx.partition_size = partition_size
        ctx.tensor_size = value.size(0)
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        grad_storage = grad_output.storage_type()(ctx.partition_size)
        grad_output_storage = grad_output.storage()
        if grad_output_storage.size() == ctx.partition_size * config['world_size']:
            pass
        else:
            grad_output_storage.resize_(ctx.partition_size * config['world_size'])
        # use default stream here because pytorch backward uses default stream
        nccl.reduceScatter(
            grad_output_storage,
            grad_storage,
            'sum',
            config['comm']
        )
        grad_tensor = torch.tensor([], dtype=grad_output.dtype, device="cuda")
        grad_tensor.set_(grad_storage, 0, (ctx.tensor_size,))
        return grad_tensor
        