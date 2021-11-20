import torch
from . import nccl
from .global_var import config

def synchronize():
    with torch.cuda.stream(config['barrier_stream']):
        barrier = torch.cuda.FloatTensor([1])
        nccl.allReduce(barrier.storage(), barrier.storage(), 'sum', config['comm'])
    config['barrier_stream'].synchronize()

def wait_loader():
    # wait lastest loader event, and set a new one
    config['load_event'].synchronize()
    config['load_stream'].record_event(config['load_event'])

def wait_reducer():
    # wait lastest reducer event, and set a new one
    config['reduce_event'].synchronize()
    config['reduce_stream'].record_event(config['reduce_event'])


def sum_loss(loss : torch.Tensor):
    ret = torch.empty_like(loss)
    nccl.allReduce(
        loss.storage(),
        ret.storage(),
        'avg',
        config['comm']
    )
    return ret