from typing import Optional
import torch
from .global_var import config
from .utils import print_rank
from .lr_scheduler.warmup import WarmupLRScheduler


def optim_step(optim : torch.optim.Optimizer, lr_scheduler : Optional[WarmupLRScheduler] = None):
    """
    Backward with loss scale.
    Synchronize streams before optimizer steps.

    This is a helper function to call optimizer.step() and lr_scheduler.step() and synchronize streams.

    Args:
        optim (torch.optim.Optimizer): A pytorch optimizer, e.g. torch.optim.Adam, torch.optim.SGD or bmtrain.optim.AdamOffloadOptimizer
        lr_scheduler (Optional[WarmupLRScheduler]): A warmup lr scheduler, e.g. bmt.lr_scheduler.Noam
    
    This function can also handle gradient overflow by reducing the loss scale when it occurs.

    """
    
    has_scale = hasattr(optim, 'scale') and optim.scale > 1
    current_stream =  torch.cuda.current_stream()
    # some reduce ops of distributed parameter were launched on load stream
    current_stream.wait_stream(config['load_stream'])

    if has_scale:
        try:
            optim.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
        except OverflowError:
            print_rank("Gradient overflow, change scale from %lf to %lf" % (optim.scale, optim.scale / config["loss_scale_factor"]))
            optim.justify_scale(optim.scale / config["loss_scale_factor"])
            optim.zero_grad()

        if optim.steps_since_last_scale >= config["loss_scale_steps"]:
            optim.justify_scale(optim.scale * config["loss_scale_factor"])
    else:
        optim.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

    config['load_stream'].wait_stream(current_stream)