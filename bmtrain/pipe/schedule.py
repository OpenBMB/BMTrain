import sys
from bmtrain.global_var import  config
import bmtrain as bmt
from .comm import PipeCommander
import torch
import logging
from typing import Iterable

        
def backward_func(inp, backward_step, output, grad_output, optim_manager=None):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    if not isinstance(inp, list) :
        inp = [inp]
    for x in inp:
        if x is not None and (torch.is_tensor(x) and x.requires_grad):
            x.retain_grad()
    if not isinstance(output, Iterable):
        output = [output]
    if not isinstance(grad_output, Iterable):
        grad_output = [grad_output]
    backward_step(output, grad_output)
    input_grad = [None]
    if inp is not None:
        input_grad = []
        for x in inp:
            if x is None or (not torch.is_tensor(x)) or (not x.requires_grad):
                input_grad.append(None)
            else:
                input_grad.append(x.grad)

    return input_grad 

def forward_func(model, forward_step, inp, data, micro_idx, is_last_micro=False):
    output = forward_step(model, inp, data)
    if not isinstance(output, list) and not isinstance(output, tuple):
        output = [output]
    return output 

def get_logger(rank, level, print_to_screen=False):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('pipeline')
    logger.setLevel(level)
    if print_to_screen:
        if rank == 0:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    fh = logging.FileHandler(f'pipe_{rank}.log',mode="w")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def pipeline_forward_backward(model, data_iterator, forward_step, backward_step, micro_batch_size, num_micros, debug_log=False):
    """Forward and backward the pipeline model.

    Args:
        models (TransformerBlocklist): The list of models.
        data_iterator (iterator): The iterator of the dataset.
        forward_step(function): Describe how to forward the model and how to get loss
        micro_batch_size (int): The micro batch size.

    Returns:
        torch.Tensor: The loss of the model.
    """

    # forwrad unpack
    loss = None
    if 'logger' not in config:
        if debug_log:
            config['logger'] = get_logger(bmt.config['pipe_rank'], level="INFO", print_to_screen=True)
        else:
            config['logger'] = logging.getLogger("dummy")
            config['logger'].addHandler(logging.NullHandler())
        
    micro_batch_size = micro_batch_size
    num_micro_batches = num_micros
    global_batch_size = micro_batch_size * num_micro_batches
    assert (num_micro_batches) % config["pipe_size"] == 0, "The number of micro batches must be divisible by the pipeline size"
    config["micros"] = num_micro_batches
    topo = config["topology"]
    logger = config['logger']
    logger.info("topo: {}".format(topo))
    logger.info("num_micro_batches: {}".format(num_micro_batches))
    logger.info("micro_batch_size: {}".format(micro_batch_size))
    logger.info("global_batch_size: {}".format(global_batch_size))
    # construct Pipe Commander
    forward_only = False
    logger.info("forward_only: {}".format(forward_only))
    if forward_only:
        num_warmup = num_micro_batches
    else:
        num_warmup = topo.pipe_size - topo.pipe_rank - 1
    interleaving_size = 1
    commander = PipeCommander(topo,model=model, data_iter=data_iterator, num_micros=num_micro_batches,\
                                num_warmup=num_warmup, forward_only=False, \
                                interleaving_size=interleaving_size \
                                )
    inps = []
    outputs = []
    logger.info("num_warmup: {}".format(num_warmup))
    for micro in range(num_warmup):
        inp, data = commander.recv_prev(need_data=True)
        logger.info("{} recv micro {}th from prev neighbour".format(bmt.config["topology"].pipe_rank, micro))
        output = forward_func(model, forward_step, inp, data, micro)
        logger.info("{} micro forward".format(micro))
        # send activations
        commander.send_next(output)
        logger.info("{} send micro {}th to next neighbour".format(bmt.config["topology"].pipe_rank, micro))
        if not forward_only:
            inps.append(inp)
            outputs.append(output)
    remain_batch = num_micro_batches - num_warmup
    logger.info("remain_batch: {}".format(remain_batch))
    if remain_batch > 0:
        inp, data = commander.recv_prev(need_data=True)
    logger.info("recv micro from prev neighbour")
    for micro in range(num_micro_batches - num_warmup):
        is_last_micro = micro == num_micro_batches - num_warmup - 1
        output = forward_func(model, forward_step, inp, data, micro + num_warmup, is_last_micro)
        if commander.is_last_stage():
            loss = output[0]    
        logger.info("{} micro forward".format(micro+num_warmup))
        grad_output = commander.send_forward_recv_backward(output)

        inps.append(inp)
        outputs.append(output)

        logger.info("{} send micro hidden state {}th to next neighbour and recv micro grad {} from next neighbour".format(bmt.config["topology"].pipe_rank, micro + num_warmup, micro))

        inp = inps.pop(0)
        output = outputs.pop(0)

        inp_grad = backward_func(inp, backward_step, output, grad_output)
        logger.info("{} micro backward".format(micro+num_warmup))
        if micro == remain_batch - 1:
            inp = None
            commander.send_prev(inp_grad)
            logger.info("{} send micro grad {}th to prev neighbour".format(bmt.config["topology"].pipe_rank, micro + num_warmup))
        else:
            logger.info("send backward and recv forward")
            inp, data = commander.send_backward_recv_forward(inp_grad, need_data=True)
    if not forward_only:
        logger.info("cooling stage")
        for i in range(num_warmup):
            logger.info("{} recv micro grad {}th from next neighbour".format(bmt.config["topology"].pipe_rank, num_micro_batches - num_warmup + i))
            inp = inps.pop(0)
            output = outputs.pop(0)
            grad_output = commander.recv_next()
            logger.info("{} micro backward".format(num_micro_batches - num_warmup + i))
            input_grad = backward_func(
                inp, backward_step, output , grad_output,
            )
            logger.info("{} send micro grad {}th to prev neighbour".format(bmt.config["topology"].pipe_rank, i))

            commander.send_prev(input_grad)
    blocklist = model.get_blocklist()
    # blocklist.reduce_tied_module()
    
    bmt.synchronize()

    
