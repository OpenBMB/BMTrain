import datetime
import torch
import random
import torch.distributed as dist
import os
from .global_var import config

from . import nccl


def init_distributed(
        init_method : str = "env://",
        seed : int = 0,
    ):
    """Initialize distributed training.
    This function will initialize the distributed training, set the random seed and global configurations.
    It must be called before any other distributed functions.

    Args:
        seed (int): The random seed.

    **init_distributed** reads the following environment variables: 
    
    * `WORLD_SIZE`: The total number gpus in the distributed training.
    * `RANK`: The global rank of the current gpu. From 0 to `WORLD_SIZE - 1`.
    * `MASTER_ADDR`: The address of the master node.
    * `MASTER_PORT`: The port of the master node.
    * `LOCAL_RANK`: The local rank of the current gpu.
    
    Normally, all the environments variables above are setted by the pytorch distributed launcher.

    **Note**: Do not use any functions in torch.distributed package including `torch.distributed.init_process_group` .

    **Note**: If your training script is stuck here , it means some of your distributed workers are not connected to the master node.

    """
    torch.backends.cudnn.enabled = False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE","1"))
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"]="localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"]="10010"
    addr = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    master = addr+":"+port
    timeout = datetime.timedelta(seconds=1800)
    rendezvous_iterator = dist.rendezvous(
        init_method, rank, world_size, timeout=timeout
    )   

    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(timeout)
    store = dist.PrefixStore("bmtrain", store)
    torch.cuda.set_device(local_rank)
    config["initialized"] = True
    config["local_rank"] = local_rank
    config["local_size"] = local_size
    config["rank"] = rank
    config["world_size"] = world_size
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    
    if rank == 0:
        unique_id : bytes = nccl.getUniqueId()
        store.set("BMTRAIN_UNIQUE_ID", unique_id.hex() )
    
    unique_id = bytes.fromhex(store.get("BMTRAIN_UNIQUE_ID").decode())
    config['comm'] = nccl.commInitRank(unique_id, world_size, rank)
    config['zero_comm'] = config['comm']

def is_initialized() -> bool:
    return config["initialized"]

