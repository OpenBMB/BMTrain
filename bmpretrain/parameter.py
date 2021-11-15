import torch
import torch.distributed as dist
from .utils import round_up
from .global_var import config
from .distributed import all_gather


class DistributedParameter(torch.nn.Parameter):
    _original_shape : torch.Size

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
        return ret

    def gather(self):
        return all_gather(self)