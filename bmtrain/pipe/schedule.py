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
        if x is not None and (torch.is_tensor(x) and x.requires_grad):
            x.retain_grad()
    if not isinstance(output, Iterable):
        output = [output]
    if not isinstance(grad_output, Iterable):
        grad_output = [grad_output]
    #TODO scale the grad
    # if output_tensor_grad[0] is None and config.grad_scale_func is not None:
    #     output_tensor[0] = config.grad_scale_func(output_tensor[0])
    if optim_manager is not None and config["topology"].is_last_rank():
        if not torch.is_tensor(output[0]) and isinstance(output[0], Iterable):
            output = optim_manager.scale_loss(output[0][0])
        elif torch.is_tensor(output[0]):
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
            if x is None or (not torch.is_tensor(x)) or (not x.requires_grad):
                input_grad.append(None)
            else:
                input_grad.append(x.grad)

    return input_grad 

def forward_func(model, inp, micro_idx, is_last_micro=False):
    if config["topology"].pipe_rank == config["topology"].pipe_size - 1:
        loss = model(*inp)
         
        return [loss]
    else:
        config['logger'].info("inp shape: {}".format(inp[0].shape))
        hidden_state = model(*inp)
        config['logger'].info("inp shape: {}".format(hidden_state[0].shape))
        if torch.is_tensor(hidden_state) or (not isinstance(hidden_state, Iterable)):
            hidden_state = [hidden_state]
        return hidden_state

def pipeline_forward_backward(model, data_iterator, micro_batch_size, num_micros, optim_manager, clip_grad=1.0):
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
        inp = commander.recv_prev(need_data=True)
        logger.info("{} recv micro {}th from prev neighbour".format(bmt.config["topology"].pipe_rank, micro))
        output = forward_func(model, inp, micro)
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
        inp = commander.recv_prev(need_data=True)
    logger.info("recv micro from prev neighbour")
    loss_items = []
    for micro in range(num_micro_batches - num_warmup):
        is_last_micro = micro == num_micro_batches - num_warmup - 1
        output = forward_func(model, inp, micro + num_warmup, is_last_micro)
        if commander.is_last_stage():
            loss = output[0]    
            loss_items.append(loss)
        logger.info("{} micro forward".format(micro+num_warmup))
        grad_output = commander.send_forward_recv_backward(output)

        inps.append(inp)
        outputs.append(output)

        logger.info("{} send micro hidden state {}th to next neighbour and recv micro grad {} from next neighbour".format(bmt.config["topology"].pipe_rank, micro + num_warmup, micro))

        inp = inps.pop(0)
        output = outputs.pop(0)

        inp_grad = backward_step(inp, output, grad_output, optim_manager)
        logger.info("{} micro backward".format(micro+num_warmup))
        if micro == remain_batch - 1:
            inp = None
            commander.send_prev(inp_grad)
            logger.info("{} send micro grad {}th to prev neighbour".format(bmt.config["topology"].pipe_rank, micro + num_warmup))
        else:
            logger.info("send backward and recv forward")
            inp = commander.send_backward_recv_forward(inp_grad, need_data=True)
    if not forward_only:
        logger.info("cooling stage")
        for i in range(num_warmup):
            logger.info("{} recv micro grad {}th from next neighbour".format(bmt.config["topology"].pipe_rank, num_micro_batches - num_warmup + i))
            inp = inps.pop(0)
            output = outputs.pop(0)
            grad_output = commander.recv_next()
            logger.info("{} micro backward".format(num_micro_batches - num_warmup + i))
            input_grad = backward_step(
                inp, output , grad_output,
            )
            logger.info("{} send micro grad {}th to prev neighbour".format(bmt.config["topology"].pipe_rank, i))

            commander.send_prev(input_grad)
    blocklist = model.get_blocklist()
    # blocklist.reduce_tied_module()
    grad_norm = optim_manager.clip_grad_norm(optim_manager.optimizers[0].param_groups, clip_grad, norm_type=2)
    optim_manager.step()
    
    bmt.synchronize()
    return loss_items, grad_norm

    