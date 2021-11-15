import torch
import random
import torch.distributed as dist
import os
from .utils import print_dict
from .global_var import config
from . import nccl

def init_distributed(
        seed : int = 0, 
    ):
    torch.backends.cudnn.enabled = False
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    master = os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"]

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    config["local_rank"] = local_rank
    config["local_size"] = local_size
    config["rank"] = rank
    config["world_size"] = world_size
    config["calc_stream"] = torch.cuda.current_stream()
    config["load_stream"] = torch.cuda.Stream()
    config["load_event"] = torch.cuda.Event()

    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    store = dist.distributed_c10d._get_default_store()
    if rank == 0:
        unique_id : bytes = nccl.getUniqueId()
        store.set("BMPRETRAIN_UNIQUE_ID", unique_id.hex() )
    dist.barrier()
    
    unique_id = bytes.fromhex(store.get("BMPRETRAIN_UNIQUE_ID").decode())
    config['comm'] = nccl.commInitRank(unique_id, world_size, rank)
    
    print_dict("Initialization", {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "local_size": local_size,
        "master" : master,
        "device": torch.cuda.current_device(),
    })
