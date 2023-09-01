from ..global_var import  config
from .comm import PipeCommander
import torch

def backward_step(input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
    for x in input_tensor:
        if x is not None:
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
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    return input_tensor_grad

def pipeline_forward_backward(models, inputs, data_iterator, global_batch_size, interleaving_size=1):
    """Forward and backward the pipeline model.

    Args:
        models (TransformerBlocklist): The list of models.
        data_iterator (iterator): The iterator of the dataset.
        micro_batch_size (int): The micro batch size.

    Returns:
        torch.Tensor: The loss of the model.
    """

    # forwrad unpack
    inp, *args = data_iterator
    micro_batch_size = inp.shape[0]
    assert global_batch_size % micro_batch_size == 0, "The global batch size must be divisible by the micro batch size"
    num_micro_batches = global_batch_size // micro_batch_size
    assert (num_micro_batches) % config["pipe_size"] == 0, "The number of micro batches must be divisible by the pipeline size"
    topo = config["topology"]
    # construct Pipe Commander
    forward_only = torch.is_grad_enabled()
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

    for micro in range(num_warmup):
        inp = commander.recv_peer(need_data=True)
        output = models(*inp)
        # send activations
        commander.send_peer(output)
        if not forward_only:
            inps.append(inp)
            outputs.append(output)
    remain_batch = num_micro_batches - num_warmup

    if remain_batch > 0:
        inp = commander.recv_peer(need_data=True)

    for micro in range(num_micro_batches - num_warmup):
        output = models(*inp)
        grad_output = commander.send_forward_recv_backward(output)
        inp_grad = backward_step(inp, output, grad_output)
        if micro == remain_batch - 1:
            input_tensor = None
            commander.send_prev(inp_grad)
        else:
            input_tensor = commander.send_backward_recv_forward(inp_grad)
        for i in range(num_warmup):

            # if i == num_warmup - 1:
                # grad sync
                # if config.grad_sync_func is None or rank == 0:
                #     enable_grad_sync()

            input_tensor = inp.pop(0)
            output_tensor = output.pop(0)

            output_tensor_grad = commander.recv_next()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, 
            )

            commander.send_prev(input_tensor_grad)     


    