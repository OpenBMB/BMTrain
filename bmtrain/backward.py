from typing import Optional
import torch
from .global_var import config
from .utils import print_rank
from .lr_scheduler.warmup import WarmupLRScheduler


def optim_step(optim : torch.optim.Optimizer, lr_scheduler : Optional[WarmupLRScheduler] = None):
    """
    This is a helper function to call optimizer.step() and lr_scheduler.step() and synchronize streams.

    Args:
        optim (torch.optim.Optimizer): A pytorch optimizer, e.g. torch.optim.Adam, torch.optim.SGD or bmtrain.optim.AdamOffloadOptimizer
        lr_scheduler (Optional[WarmupLRScheduler]): A warmup lr scheduler, e.g. bmt.lr_scheduler.Noam
    
    This function can also handle gradient overflow by reducing the loss scale when it occurs.

    """
    
    current_stream =  torch.cuda.current_stream()
    # some reduce ops of distributed parameter were launched on load stream
    current_stream.wait_stream(config['load_stream'])

    optim.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    config['load_stream'].wait_stream(current_stream)