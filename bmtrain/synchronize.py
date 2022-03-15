import torch
from . import nccl
from .global_var import config

def synchronize():
    """
    Synchronize all the workers across all nodes. (both CPU and GPU are synchronized)
    """
    with torch.cuda.stream(config['barrier_stream']):
        barrier = torch.cuda.FloatTensor([1])
        nccl.allReduce(barrier.storage(), barrier.storage(), 'sum', config['comm'])
    config['barrier_stream'].synchronize()

def wait_loader():
    # wait lastest loader event, and set a new one
    config['load_event'].synchronize()
    config['calc_stream'].record_event(config['load_event'])


def sum_loss(loss : torch.Tensor):
    """
    Sum the loss across all workers.

    This is a helper function to reduce the loss across all workers.
    """
    ret = torch.empty_like(loss)
    nccl.allReduce(
        loss.storage(),
        ret.storage(),
        'avg',
        config['comm']
    )
    return ret

def gather_result(result: torch.Tensor):
    if not result.is_cuda:
        result = result.cuda()
    ret = torch.empty((result.shape[0]*config['world_size'], *list(result.shape[1:])), device=result.device, dtype=result.dtype)
    nccl.allGather(
        result.storage(),
        ret.storage(),
        config['comm']
    )
    return ret