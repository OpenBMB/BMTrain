import sys
from bmtrain.global_var import  config
from bmtrain.loss import FusedCrossEntropy
import bmtrain as bmt
from .debug import get_logger
from .comm import PipeCommander
import torch
import logging
from typing import Iterable
def backward_step(inp, output, grad_output):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    if not isinstance(inp, list) :
        inp = [inp]
    for x in inp:
        if x is not None and x.requires_grad:
            x.retain_grad()
    if not isinstance(output, list):
        output = [output]
    if not isinstance(grad_output, list):
        grad_output = [grad_output]
    #TODO scale the grad
    # if output_tensor_grad[0] is None and config.grad_scale_func is not None:
    #     output_tensor[0] = config.grad_scale_func(output_tensor[0])
    torch.autograd.backward(output[0], grad_tensors=grad_output[0])

    input_grad = [None]
    if inp is not None:
        input_grad = []
        for x in inp:
            if x is None or not x.requires_grad:
                input_grad.append(None)
            else:
                input_grad.append(x.grad)

    return input_grad 

def forward_func(model, inp, micro_idx):
    if config["topology"].pipe_rank == config["topology"].pipe_size - 1:
        loss = model(*inp)
        config['logger'].info("loss: {}".format(loss.item()))
         
        return loss
    else:
        hidden_state = model(*inp)
        config['logger'].info("inp shape: {}".format(hidden_state[0].shape))
        if not isinstance(hidden_state, Iterable):
            hidden_state = [hidden_state]
        return hidden_state

def pipeline_forward_backward(model, data_iterator, global_batch_size, interleaving_size=1):
    """Forward and backward the pipeline model.

    Args:
        models (TransformerBlocklist): The list of models.
        data_iterator (iterator): The iterator of the dataset.
        micro_batch_size (int): The micro batch size.

    Returns:
        torch.Tensor: The loss of the model.
    """

    # forwrad unpack
    micro_batch_size = 2
    assert global_batch_size % micro_batch_size == 0, "The global batch size must be divisible by the micro batch size"
    num_micro_batches = global_batch_size // micro_batch_size
    assert (num_micro_batches) % config["pipe_size"] == 0, "The number of micro batches must be divisible by the pipeline size"
    config["micros"] = num_micro_batches
    topo = config["topology"]
    logger = get_logger(config['rank'], logging.DEBUG)
    config['logger'] = logger
    logger.info("topo: {}".format(topo))
    logger.info("num_micro_batches: {}".format(num_micro_batches))
    logger.info("micro_batch_size: {}".format(micro_batch_size))
    logger.info("global_batch_size: {}".format(global_batch_size))
    logger.info("interleaving_size: {}".format(interleaving_size))
    # construct Pipe Commander
    forward_only = False
    logger.info("forward_only: {}".format(forward_only))
    if forward_only:
        num_warmup = num_micro_batches
    else:
        num_warmup = topo.pipe_size - topo.pipe_rank - 1
    def generator(data_iterator):
        yield model.preprocess_func(next(data_iterator))
    commander = PipeCommander(topo,input_generator=generator(data_iterator), num_micros=num_micro_batches,\
                                num_warmup=num_warmup, forward_only=False, \
                                interleaving_size=interleaving_size, \
                                )
    # if commander.is_first_stage() or commander.is_last_stage():
        # module = model.head_layer() if commander.is_first_stage() else model.tail_layer()
        # commander.param_reduce(module)
    inps = []
    outputs = []
    logger.info("num_warmup: {}".format(num_warmup))
    for micro in range(num_warmup):
        inp = commander.recv_prev(need_data=True)
        logger.info("{} recv micro {}th from prev neighbour".format(config['rank'], micro))
        output = forward_func(model, inp, micro)
        logger.info("{} micro forward".format(micro))
        # send activations
        commander.send_next(output)
        logger.info("{} send micro {}th to next neighbour".format(config['rank'], micro))
        if not forward_only:
            inps.append(inp)
            outputs.append(output)
    remain_batch = num_micro_batches - num_warmup
    logger.info("remain_batch: {}".format(remain_batch))
    if remain_batch > 0:
        inp = commander.recv_prev(need_data=True)

    for micro in range(num_micro_batches - num_warmup):
        output = forward_func(model, inp, micro + num_warmup)
        logger.info("{} micro forward".format(micro+num_warmup))
        grad_output = commander.send_forward_recv_backward(output)
        inps.append(inp)
        outputs.append(output)
        logger.info("{} send micro hidden state {}th to next neighbour and recv micro grad {} from next neighbour".format(config['rank'], micro + num_warmup, micro))
        logger.debug("inp shape: {}".format(inp[0].shape))
        if not commander.is_last_stage():
            logger.debug("output shape: {}".format(output[0].shape))
        if  grad_output[0] is not None :
            logger.debug("grad_output shape: {}".format(grad_output[0].shape))
        inp = inps.pop(0)
        output = outputs.pop(0)
        for x in inp:
            logger.info("inp requires_grad: {}".format(x.requires_grad))
        inp_grad = backward_step(inp, output, grad_output)
        logger.info("{} micro backward".format(micro+num_warmup))
        if micro == remain_batch - 1:
            inp = None
            commander.send_prev(inp_grad)
            logger.info("{} send micro grad {}th to prev neighbour".format(config['rank'], micro + num_warmup))
        else:
            if inp_grad[0] is not None:
                logger.debug("inp_grad shape: {}".format(inp_grad[0].shape))
            inp = commander.send_backward_recv_forward(inp_grad, need_data=True)
            logger.debug("inp type: {}".format(type(inp)))
            logger.debug("inp shape: {}".format(inp[0].shape))
            logger.info("{} send micro grad {}th to prev neighbour and recv micro hidden state {} from prev neighbour".format(config['rank'], micro, micro + num_warmup + 1))


    if not forward_only:
        logger.info("cooling stage")
        for i in range(num_warmup):
            logger.info("{} recv micro grad {}th from next neighbour".format(config['rank'], num_micro_batches - num_warmup + i))
            inp = inps.pop(0)
            output = outputs.pop(0)

            grad_output = commander.recv_next()
            logger.info("{} micro backward".format(num_micro_batches - num_warmup + i))
            input_grad = backward_step(
                inp, output , grad_output, 
            )
            logger.info("{} send micro grad {}th to prev neighbour".format(config['rank'], i))

            commander.send_prev(input_grad)
    bmt.synchronize()

    