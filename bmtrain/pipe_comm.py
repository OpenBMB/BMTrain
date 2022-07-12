import torch
from . import nccl
from .global_var import config
from .store import DTYPE_LIST
def send_meta(hidden_state, next_rank, comm):
    meta = [len(hidden_state.size()), DTYPE_LIST.index(hidden_state.dtype)] + list(hidden_state.size())
    meta_data = torch.tensor(data=meta, device=hidden_state.device, dtype=torch.long)
    nccl.send(meta_data.storage(), next_rank, comm)
def send_activations(hidden_state, next_rank, comm):
    send_meta(hidden_state, next_rank, comm)
    nccl.send(hidden_state.storage(), next_rank, comm)
def recv_meta(prev_rank, comm):
    meta_data = torch.tensor(data=[0]*50, device="cuda", dtype=torch.long)
    nccl.recv(meta_data.storage(), prev_rank, comm)
    n_dims = meta_data[0].item()
    dtype = DTYPE_LIST[meta_data[1].item()]
    shape = meta_data[2:n_dims+2].tolist()
    return dtype,shape
def recv_activations(prev_rank, comm):
    dtype, shape = recv_meta(prev_rank, comm)
    hidden_state = torch.empty(shape, dtype=dtype, device="cuda")
    nccl.recv(hidden_state.storage(), prev_rank, comm)
    return hidden_state
def gather_input(hidden_state, comm):
    shape = list(hidden_state.size())
    shape[0] = hidden_state.size()[0]*nccl.commCount(comm)
    inputs = torch.empty(shape, dtype=hidden_state.dtype, device=hidden_state.device)
    nccl.allGather(hidden_state.storage(), inputs.storage(), comm)
    return inputs
def broadcast(src, root, comm, inplace=False):
    if inplace:
        outputs = src
    else:
        outputs = torch.empty_like(src, dtype = src.dtype, device = src.device)
    nccl.broadcast(src.storage(), outputs.storage(), root, comm)
    return outputs
def forward_pass():
    pass
def backward_pass():
    pass
def send_grad():
    pass
def receive_grad():
    pass