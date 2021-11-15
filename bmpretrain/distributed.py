import torch
from .global_var import config
from . import nccl

def all_gather(value : torch.Tensor) -> torch.Tensor:
    
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
    return output_tensor

        