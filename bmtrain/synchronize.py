import warnings
from typing import Optional

import torch
import torch.distributed as dist

from . import distributed
from .comm import Communicator
from .global_var import config


def synchronize(comm=None):
    """
    Synchronize all the workers across all nodes. (both CPU and GPU are
    synchronized)
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")

    group = (comm or config["comm"]).group
    with torch.cuda.stream(config["barrier_stream"]):
        barrier = torch.ones(1, device="cuda")
        dist.all_reduce(barrier, group=group)
    config["barrier_stream"].synchronize()


def wait_loader():
    """
    Calc_stream (normally current stream) wait latest loader event, and set
    a new one.
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")

    config["load_event"].synchronize()
    config["calc_stream"].record_event(config["load_event"])


def sum_loss(loss: torch.Tensor, comm: Optional[Communicator] = None):
    """
    Sum the loss across all workers.

    This is a helper function to reduce the loss across all workers.
    """
    if comm is None:
        comm = config["comm"]
    warnings.warn(
        "bmtrain.sum_loss is deprecated and will be removed in later version. "
        "Use bmtrain.distributed.all_reduce instead.",
        DeprecationWarning,
    )

    return distributed.all_reduce(loss, "avg", comm)


def gather_result(result: torch.Tensor):
    """
    Gather result across all workers.
    """
    warnings.warn(
        "bmtrain.gather_result is deprecated and will be removed in later "
        "version. Use bmtrain.distributed.all_gather instead.",
        DeprecationWarning,
    )
    # Clone sliced or non-contiguous tensors so data_ptr is at offset 0 for NCCL.
    if result.storage_offset() != 0 or not result.is_contiguous():
        result = result.clone()

    output_cuda = True
    if not result.is_cuda:
        result = result.cuda()
        output_cuda = False
    ret = torch.empty(
        (result.shape[0] * config["world_size"], *list(result.shape[1:])),
        device=result.device,
        dtype=result.dtype,
    )
    config["comm"].all_gather(result, ret)
    if output_cuda:
        return ret
    else:
        return ret.cpu()
