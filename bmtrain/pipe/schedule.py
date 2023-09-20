import sys
from bmtrain.global_var import  config
import bmtrain as bmt
from .comm import PipeCommander
import torch
from typing import Iterable

        
def backward_step(inp, output, grad_output, optim_manager=None):
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
    if not isinstance(output, Iterable):
        output = [output]
    if not isinstance(grad_output, Iterable):
        grad_output = [grad_output]
    #TODO scale the grad
    # if output_tensor_grad[0] is None and config.grad_scale_func is not None:
    #     output_tensor[0] = config.grad_scale_func(output_tensor[0])
    if optim_manager is not None and config["topology"].is_last_rank():
        output = optim_manager.scale_loss(output[0])
    else:
        output = output[0]
    torch.autograd.backward(output, grad_tensors=grad_output[0])
    current_stream = torch.cuda.current_stream()
    current_stream.wait_stream(config['load_stream'])
    input_grad = [None]
    if inp is not None:
        input_grad = []
        for x in inp:
            if x is None or not x.requires_grad:
                input_grad.append(None)
            else:
                input_grad.append(x.grad)

    return input_grad 

def forward_func(model, inp, micro_idx, is_last_micro=False):
    if config["topology"].pipe_rank == config["topology"].pipe_size - 1:
        loss = model(*inp)
         
        return [loss]
    else:
        hidden_state = model(*inp)
        config['logger'].info("inp shape: {}".format(hidden_state[0].shape))
        if not isinstance(hidden_state, Iterable):
            hidden_state = [hidden_state]
        return hidden_state

def pipeline_forward_backward(model, data_iterator, global_batch_size, optim_manager, interleaving_size=1):
    """Forward and backward the pipeline model.

    Args:
        models (TransformerBlocklist): The list of models.
        data_iterator (iterator): The iterator of the dataset.
        micro_batch_size (int): The micro batch size.

    Returns:
        torch.Tensor: The loss of the model.
    """

    # forwrad unpack
    loss = None
    optim_manager.zero_grad()
    micro_batch_size = 2
    assert global_batch_size % micro_batch_size == 0, "The global batch size must be divisible by the micro batch size"
    num_micro_batches = global_batch_size // micro_batch_size
    assert (num_micro_batches) % config["pipe_size"] == 0, "The number of micro batches must be divisible by the pipeline size"
    config["micros"] = num_micro_batches
    topo = config["topology"]
    logger = config['logger']
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
        while True:
            try:
                inp = next(data_iterator)
                yield model.preprocess_func(inp)
            except StopIteration:
                break

    commander = PipeCommander(topo,input_generator=generator(data_iterator), num_micros=num_micro_batches,\
                                num_warmup=num_warmup, forward_only=False, \
                                interleaving_size=interleaving_size, \
                                )
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
        is_last_micro = micro == num_micro_batches - num_warmup - 1
        output = forward_func(model, inp, micro + num_warmup, is_last_micro)
        if commander.is_last_stage():
            loss = output[0]    
        logger.info("{} micro forward".format(micro+num_warmup))
        grad_output = commander.send_forward_recv_backward(output)

        inps.append(inp)
        outputs.append(output)

        logger.info("{} send micro hidden state {}th to next neighbour and recv micro grad {} from next neighbour".format(config['rank'], micro + num_warmup, micro))

        inp = inps.pop(0)
        output = outputs.pop(0)

        for x in inp:
            logger.info("inp requires_grad: {}".format(x.requires_grad))
        inp_grad = backward_step(inp, output, grad_output, optim_manager)
        logger.info("{} micro backward".format(micro+num_warmup))
        if micro == remain_batch - 1:
            inp = None
            commander.send_prev(inp_grad)
            logger.info("{} send micro grad {}th to prev neighbour".format(config['rank'], micro + num_warmup))
        else:
            logger.info("send backward and recv forward")
            inp = commander.send_backward_recv_forward(inp_grad, need_data=True)
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
    model.transformers.reduce_tied_module()
    optim_manager.step()
    bmt.synchronize()
    return loss

    