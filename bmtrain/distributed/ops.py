import torch

from ..comm import Communicator, groupcall  # re-export for backward compat
from ..global_var import config
from .dtype import DTYPE_LIST

__all__ = [
    "send_activations",
    "recv_activations",
    "OpBroadcast",
    "broadcast",
    "OpAllGather",
    "all_gather",
    "OpReduceScatter",
    "reduce_scatter",
    "OpAllReduce",
    "all_reduce",
    "groupcall",
]


# ---- helpers for sending activations along the pipeline ---------------------

def send_activations(hidden_state, next_rank, comm):
    send_meta(hidden_state, next_rank, comm)
    comm.send(hidden_state.contiguous().view(-1), next_rank)


def recv_activations(prev_rank, comm):
    dtype, shape = recv_meta(prev_rank, comm)
    hidden_state = torch.empty(shape, dtype=dtype, device="cuda")
    comm.recv(hidden_state.view(-1), prev_rank)
    return hidden_state


def send_meta(x, next_rank, comm):
    meta_data = torch.tensor(data=[0] * 50, device="cuda", dtype=torch.int)
    meta_data[0] = len(x.size())
    meta_data[1] = DTYPE_LIST.index(x.dtype)
    meta_data[2 : len(x.size()) + 2] = torch.tensor(
        x.size(), device="cuda", dtype=torch.int
    )
    meta_data = meta_data.contiguous()
    comm.send(meta_data, next_rank)


def recv_meta(prev_rank, comm):
    meta_data = torch.tensor(data=[0] * 50, device="cuda", dtype=torch.int)
    comm.recv(meta_data, prev_rank)
    n_dims = meta_data[0].item()
    dtype = DTYPE_LIST[meta_data[1].item()]
    shape = meta_data[2 : n_dims + 2].tolist()
    return dtype, shape


# ---- Broadcast --------------------------------------------------------------

class OpBroadcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, root, comm=None):
        if comm is None:
            comm = config["comm"]
        ctx.comm = comm
        outputs = torch.empty_like(src, dtype=src.dtype, device=src.device)
        comm.broadcast(src.contiguous().view(-1), outputs.view(-1), root)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        res = all_reduce(grad_output, "sum", ctx.comm)
        return res, None, None


def broadcast(src, root, comm=None):
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")
    return OpBroadcast.apply(src, root, comm)


# ---- AllGather --------------------------------------------------------------

class OpAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, comm: "Communicator" = None):
        if comm is None:
            comm = config["comm"]
        world_size = comm.world_size
        if not input.is_contiguous():
            input = input.contiguous()
        # Clone if storage_offset != 0 so data_ptr points to the start of the tensor data.
        if input.storage_offset() != 0:
            input = input.clone()
        output = torch.empty(
            (world_size,) + input.size(), dtype=input.dtype, device=input.device
        )
        ctx.comm = comm
        comm.all_gather(input.view(-1), output.view(-1))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output[ctx.comm.rank], None


def all_gather(x: torch.Tensor, comm=None):
    """Gathers the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (...).

    Returns:
        torch.Tensor: The gathered tensor of shape (world_size, ...).
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")
    assert x.is_cuda
    return OpAllGather.apply(x, comm)


# ---- ReduceScatter ----------------------------------------------------------

class OpReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, op: str, comm: "Communicator" = None):
        if comm is None:
            comm = config["comm"]
        ctx.comm = comm
        rank = comm.rank
        assert (
            input.shape[0] % comm.world_size == 0
        ), "The dimension 0 must be divisible by the number of communication processes"
        if not input.is_contiguous():
            input = input.contiguous()
        if input.storage_offset() != 0:
            input = input.clone()
        output_shape = (input.shape[0] // comm.world_size, *input.shape[1:])
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        comm.reduce_scatter(input.view(-1), output.view(-1), op)
        ctx.op = op
        if op in ["sum", "avg"]:
            pass
        elif op in ["max", "min"]:
            ctx.save_for_backward(
                output != input[rank * input.shape[0] : (rank + 1) * input.shape[0]]
            )
        else:
            ctx.save_for_backward(
                output / input[rank * input.shape[0] : (rank + 1) * input.shape[0]]
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            grad_output = OpAllGather.apply(grad_output, ctx.comm).flatten(0, 1)
        if ctx.op in ["max", "min", "prod"]:
            raise NotImplementedError("max min operation now do not support backward")
        else:
            if ctx.op == "avg":
                grad_output /= ctx.comm.world_size
            return grad_output, None, None


def reduce_scatter(x: torch.Tensor, op: str = "sum", comm=None):
    """Reduces the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (world_size, ...).
        op (str): The reduction operation, one of "sum", "avg", "max", "min", "prod".
            Default: "sum".

    Returns:
        torch.Tensor: The reduced tensor of shape (...).
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")
    assert x.is_cuda
    return OpReduceScatter.apply(x, op, comm)


# ---- AllReduce --------------------------------------------------------------

class OpAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, op: str, comm: "Communicator" = None):
        if comm is None:
            comm = config["comm"]
        ctx.comm = comm
        if not input.is_contiguous():
            input = input.contiguous()
        if input.storage_offset() != 0:
            input = input.clone()
        output = torch.empty(input.size(), dtype=input.dtype, device=input.device)

        comm.all_reduce(input.view(-1), output.view(-1), op)
        ctx.op = op

        if op in ["sum", "avg"]:
            pass
        elif op in ["max", "min"]:
            ctx.save_for_backward(input != output)
        else:
            ctx.save_for_backward(output / input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.op == "sum":
            return grad_output, None, None
        elif ctx.op == "avg":
            return grad_output / ctx.comm.world_size, None, None
        elif ctx.op in ["max", "min"]:
            return torch.masked_fill(grad_output, ctx.saved_tensors[0], 0), None, None
        else:
            return grad_output * ctx.saved_tensors[0], None, None


def all_reduce(x: torch.Tensor, op: str = "sum", comm=None):
    """Reduces the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (...).
        op (str): The reduction operation, one of "sum", "avg", "max", "min", "prod".
            Default: "sum".

    Returns:
        torch.Tensor: The reduced tensor of shape (...).
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")
    assert x.is_cuda
    return OpAllReduce.apply(x, op, comm)
