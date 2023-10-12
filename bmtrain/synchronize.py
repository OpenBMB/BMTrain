import torch
from . import distributed, nccl
from .global_var import config
import warnings

def synchronize():
    """
    Synchronize all the workers across all nodes. (both CPU and GPU are synchronized)
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")

    with torch.cuda.stream(config['barrier_stream']):
        barrier = torch.cuda.FloatTensor([1])
        nccl.allReduce(barrier.storage(), barrier.storage(), 'sum', config['comm'])
    config['barrier_stream'].synchronize()
