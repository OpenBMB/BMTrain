import torch
import torch.nn.functional as F
from bmtrain.global_var import config 
from ..distributed import all_gather, all_reduce
from .. import nccl
import bmtrain as bmt
from enum import Enum

class ReduceType(Enum):
    ALL_REDUCE = 1
    REDUCE_SCATTER = 2

def preprocess_input(input, gather_input, split_input):
    if gather_input:
        input = all_gather(input, config['tp_comm'])
        input = input.flatten(0, 1)

    if split_input:
        all_input_list = input.chunk(config['tp_size'], dim=-1)
        input = all_input_list[config['topology'].tp_id]
    return input

def async_all_gather_linear_func(input, weight, bias, async_chunks=2):
    dim = input.dim()
    shape = list(input.shape)
    if dim > 2:
        input = input.flatten(0, 1)
    tp_size = config['tp_size']
    current_stream = torch.cuda.current_stream()

    rounds = async_chunks
    inputs = input.chunk(rounds, dim=0)
    config['tp_comm_stream'].wait_stream(current_stream)
    outputs = [None] * tp_size * rounds

    input = all_gather(inputs[0], config['tp_comm'])
    input = input.flatten(0, 1)
    out = F.linear(input, weight, bias)
    outs = out.chunk(tp_size, dim=0)
    for i in range(tp_size):
        outputs[i * rounds] = outs[i]

    #async all_gather and overalap with linear
    for i in range(rounds-1):
        with torch.cuda.stream(config['tp_comm_stream']):
            inputs[i+1].record_stream(config['tp_comm_stream'])
            input = all_gather(inputs[i+1], config['tp_comm'])
            input = input.flatten(0, 1)

        current_stream.wait_stream(config['tp_comm_stream'])
        out = F.linear(input, weight, bias)
        outs = out.chunk(tp_size, dim=0)
        for j in range(tp_size):
            outputs[(i + 1) + j * rounds] = outs[j]

    out = torch.cat(outputs, dim=0)
    if dim > 2:
        out_shape = list(out.shape)
        shape[0] = out_shape[0] // shape[1]
        shape[2:] = out_shape[1:]
        out = out.view(shape)
    return out

class OpParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, gather_input=False, gather_output=False, split_input=False, reduce_output_type=None, async_gather_chunks=2):
        if reduce_output_type is not None:
            reduce_output_type = ReduceType(reduce_output_type)

        ctx.save_for_backward(input, weight, bias)
        ctx.gather_output = gather_output
        ctx.split_input = split_input
        ctx.gather_input = gather_input
        ctx.reduce_output_type = reduce_output_type

        if gather_input and config['tp_size'] > 1 and async_gather_chunks > 1:
            out = async_all_gather_linear_func(input, weight, bias, async_gather_chunks)
        else:
            all_input = preprocess_input(input, ctx.gather_input, ctx.split_input)
            out = F.linear(all_input, weight, bias)

        if gather_output:
            all_output_list = all_gather(out, config['tp_comm'])
            all_output_list = all_output_list.chunk(config['tp_size'], dim=0)        
            out = torch.cat(all_output_list, dim=all_output_list[0].dim()-1).flatten(0,1)

        if reduce_output_type is None:
            return out

        if reduce_output_type == ReduceType.ALL_REDUCE:
            nccl.allReduce(out.storage(), out.storage(), "sum", config['tp_comm'])
            return out 

        elif reduce_output_type == ReduceType.REDUCE_SCATTER:
            shape = list(out.shape)
            shape[0] = shape[0] // config['tp_size']
            reduce_out = torch.empty(shape, dtype=out.dtype, device=out.device)
            nccl.reduceScatter(out.storage(), reduce_out.storage(), "sum", config['tp_comm'])
            return reduce_out
        else:
            assert False, "no support reduce type{}".format(reduce_output_type)
            
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        gather_output = ctx.gather_output

        if ctx.reduce_output_type == ReduceType.REDUCE_SCATTER:
            grad_output = all_gather(grad_output, config['tp_comm'])
            grad_output = grad_output.flatten(0, 1)

        if gather_output:
            tp_size = config['tp_size']
            tp_id = config['topology'].tp_id
            grad_output_list = grad_output.chunk(tp_size, dim=-1)
            grad_output = grad_output_list[tp_id]

        grad_input = grad_weight = grad_bias = None

        current_stream = torch.cuda.current_stream()
        if input.requires_grad or weight.requires_grad:
            if ctx.gather_input:
                # async the all_gather
                with torch.cuda.stream(config['tp_comm_stream']):
                    input.record_stream(config['tp_comm_stream'])
                    config['tp_comm_stream'].wait_stream(current_stream)
                    all_input = preprocess_input(input, ctx.gather_input, ctx.split_input)
                    #use event to solve two streams waiting for each other
                    gather_event = config['tp_comm_stream'].record_event()
            else:
                all_input = preprocess_input(input, ctx.gather_input, ctx.split_input)

        if input.requires_grad:
            grad_all_input = grad_output.matmul(weight)
            grad_input = torch.zeros_like(input)
            if ctx.gather_input:
                # async the reduce_scatter
                with torch.cuda.stream(config['tp_comm_stream']):
                    config['tp_comm_stream'].wait_stream(current_stream)
                    grad_input.record_stream(config['tp_comm_stream'])
                    grad_all_input.record_stream(config['tp_comm_stream'])
                    nccl.reduceScatter(grad_all_input.storage(), grad_input.storage(), "sum", config['tp_comm'])
            else:
                grad_input = grad_all_input

            if ctx.split_input:
                with torch.cuda.stream(config['tp_comm_stream']):
                    config['tp_comm_stream'].wait_stream(current_stream)
                    grad_input.record_stream(config['tp_comm_stream'])
                    grad_input = all_gather(grad_input, config['tp_comm'])

        # wait all_gather 
        if ctx.gather_input:
            current_stream.wait_event(gather_event)
        if weight.requires_grad:
            dim = grad_output.dim()
            grad_weight = grad_output.reshape(-1,
                grad_output.shape[-1]).t().matmul(all_input.reshape(-1, all_input.shape[-1]))
         
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(config['tp_comm_stream'])
        return grad_input, grad_weight, grad_bias, None, None, None, None, None
