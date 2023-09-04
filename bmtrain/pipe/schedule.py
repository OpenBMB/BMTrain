import sys
from bmtrain.global_var import  config
from bmtrain.loss import FusedCrossEntropy
import bmtrain as bmt
from comm import PipeCommander
import torch
from debug import get_logger

def backward_step(input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    if not isinstance(input_tensor, list) :
        input_tensor = [input_tensor]
    for x in input_tensor:
        if x is not None and x.requires_grad:
            x.retain_grad()
    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]
    #TODO scale the grad
    # if output_tensor_grad[0] is None and config.grad_scale_func is not None:
    #     output_tensor[0] = config.grad_scale_func(output_tensor[0])
    torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None or not x.requires_grad:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    return input_tensor_grad

def forward_func(model, inp):
    if not isinstance(inp, list):
        inp = [inp]
    if config["topology"].pipe_rank == config["topology"].pipe_size - 1:
        inp = model(*inp)
        config['logger'].debug("inp shape: {}".format(inp[0].shape))
        loss = inp.mean()
        config['logger'].debug("loss shape: {}".format(loss.shape))
        return loss
    else:
        hidden_state = model(*inp)
        if not isinstance(hidden_state, list):
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
    inp, *args = next(data_iterator)
    optimizer = bmt.optim.AdamOptimizer(model.parameters(), lr=0.001)
    micro_batch_size = inp.shape[0]
    assert global_batch_size % micro_batch_size == 0, "The global batch size must be divisible by the micro batch size"
    num_micro_batches = global_batch_size // micro_batch_size
    assert (num_micro_batches) % config["pipe_size"] == 0, "The number of micro batches must be divisible by the pipeline size"
    topo = config["topology"]
    logger = get_logger(config['rank'])
    config['logger'] = logger
    logger.debug("topo: {}".format(topo))
    logger.debug("num_micro_batches: {}".format(num_micro_batches))
    logger.debug("micro_batch_size: {}".format(micro_batch_size))
    logger.debug("global_batch_size: {}".format(global_batch_size))
    logger.debug("interleaving_size: {}".format(interleaving_size))
    # construct Pipe Commander
    forward_only = False
    logger.debug("forward_only: {}".format(forward_only))
    if forward_only:
        num_warmup = num_micro_batches
    else:
        num_warmup = topo.pipe_size - topo.pipe_rank - 1

    commander = PipeCommander(topo, num_micros=num_micro_batches,\
                                num_warmup=num_warmup, forward_only=False, \
                                interleaving_size=interleaving_size, \
                                data_iterator=data_iterator)
    inps = []
    outputs = []
    logger.debug("num_warmup: {}".format(num_warmup))
    for micro in range(num_warmup):
        inp = commander.recv_prev(need_data=True)
        logger.debug("{} recv micro {}th from prev neighbour".format(config['rank'], micro))
        output = forward_func(model, inp)
        logger.debug("{} micro forward".format(micro))
        # send activations
        commander.send_next(output)
        logger.debug("{} send micro {}th to next neighbour".format(config['rank'], micro))
        if not forward_only:
            inps.append(inp)
            outputs.append(output)
    remain_batch = num_micro_batches - num_warmup
    logger.debug("remain_batch: {}".format(remain_batch))
    if remain_batch > 0:
        inp = commander.recv_prev(need_data=True)

    for micro in range(num_micro_batches - num_warmup):
        output = forward_func(model, inp)
        logger.debug("{} micro forward".format(micro+num_warmup))
        grad_output = commander.send_forward_recv_backward(output)
        print(len(grad_output))
        logger.debug("{} send micro hidden state {}th to next neighbour and recv micro grad {} from next neighbour".format(config['rank'], micro + num_warmup, micro))
        logger.debug("inp shape: {}".format(inp[0].shape))
        if not commander.is_last_stage():
            logger.debug("output shape: {}".format(output[0].shape))
        if  grad_output[0] is not None :
            logger.debug("grad_output shape: {}".format(grad_output[0].shape))
        inp_grad = backward_step(inp, output, grad_output)
        logger.debug("{} micro backward".format(micro+num_warmup))
        if micro == remain_batch - 1:
            input_tensor = None
            commander.send_prev(inp_grad)
            logger.debug("{} send micro grad {}th to prev neighbour".format(config['rank'], micro + num_warmup))
        else:
            logger.debug("{} send micro grad {}th to prev neighbour and recv micro hidden state {} from recv neighbour".format(config['rank'], micro, micro + num_warmup + 1))
            logger.debug("inp_grad shape: {}".format(inp_grad[0].shape))
            input_tensor = commander.send_backward_recv_forward(inp_grad)
        inps.append(input_tensor)
        outputs.append(output)


    if not forward_only:
        logger.debug("cooling stage")
        for i in range(num_warmup):
            logger.debug("{} recv micro grad {}th from next neighbour".format(config['rank'], num_micro_batches - num_warmup + i))
            # if i == num_warmup - 1:
                # grad sync
                # if config.grad_sync_func is None or rank == 0:
                #     enable_grad_sync()

            input_tensor = inps.pop(0)
            output_tensor = outputs.pop(0)

            output_tensor_grad = commander.recv_next()
            logger.debug("{} micro backward".format(num_micro_batches - num_warmup + i))
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, 
            )
            logger.debug("{} send micro grad {}th to prev neighbour".format(config['rank'], i))

            commander.send_prev(input_tensor_grad)
    optimizer.step()

    