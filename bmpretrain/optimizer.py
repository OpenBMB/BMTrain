import torch
from .global_var import config

def optimizer_step(optim : torch.optim.Optimizer):
    """
    Synchronize streams before optimizer steps.
    """
    current_stream =  torch.cuda.current_stream()

    # some reduce ops of distributed parameter were launched on load stream
    current_stream.wait_stream(config['load_stream'])
    current_stream.wait_stream(config['reduce_stream'])

    optim.step()

    config['load_stream'].wait_stream(current_stream)