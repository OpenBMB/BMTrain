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
        all_input_list = input.chunk(config['tp_size'], dim=1)
        input = all_input_list[config['topology'].tp_id]
    return input

class OpParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, gather_input=False, gather_output=False, split_input=False, reduce_output_type=None):
        if reduce_output_type is not None:
            reduce_output_type = ReduceType(reduce_output_type)

        ctx.save_for_backward(input, weight, bias)
        ctx.gather_output = gather_output
        ctx.split_input = split_input
        ctx.gather_input = gather_input
        ctx.reduce_output_type = reduce_output_type

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
            grad_output_list = grad_output.chunk(tp_size, dim=1)
            grad_output = grad_output_list[tp_id]

        grad_input = grad_weight = grad_bias = None

        if input.requires_grad or weight.requires_grad:
            all_input = preprocess_input(input, ctx.gather_input, ctx.split_input)

        if input.requires_grad:
            current_stream = torch.cuda.current_stream()
            grad_all_input = grad_output.matmul(weight)
            grad_input = torch.zeros_like(input)
            if ctx.gather_input:
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

        if weight.requires_grad:
            dim = grad_output.dim()
            grad_weight = grad_output.reshape(-1,
                grad_output.shape[-1]).t().matmul(all_input.reshape(-1, all_input.shape[-1]))
         
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(config['tp_comm_stream'])
        return grad_input, grad_weight, grad_bias, None, None, None, None
