import contextlib
import pickle

import torch

from ..comm import groupcall as _comm_groupcall
from ..global_var import config
from .dtype import DTYPE_LIST


_p2p_stream = {}
_p2p_events = {}


@contextlib.contextmanager
def groupcall():
    """``ncclGroupStart/End``-equivalent context.

    Implemented on top of ``torch.distributed`` coalescing manager.
    """
    with _comm_groupcall():
        yield


class handler:
    def __init__(self, event):
        self.event = event

    def wait(self):
        torch.cuda.current_stream().wait_event(self.event)


# ---- object send/recv -------------------------------------------------------

def send_object(obj, peer_rank, comm):
    data_bytes: bytes = pickle.dumps(obj)
    data_length: int = len(data_bytes)

    gpu_data_length = torch.tensor([data_length], device="cuda", dtype=torch.long)
    comm.send(gpu_data_length, peer_rank)

    byte_tensor = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).cuda()
    comm.send(byte_tensor, peer_rank)


def recv_object(peer_rank, comm):
    data_length = torch.tensor([0], device="cuda", dtype=torch.long)
    comm.recv(data_length, peer_rank)
    byte_tensor = torch.empty(int(data_length.item()), dtype=torch.uint8, device="cuda")
    comm.recv(byte_tensor, peer_rank)
    buf = byte_tensor.cpu().numpy().tobytes()
    return pickle.loads(buf)


def record_stream_helper(tensor_list, stream):
    for t in tensor_list:
        t.record_stream(stream)


# ---- multi-tensor send/recv -------------------------------------------------

def send_tensors(tensor_list, peer_rank, comm):
    h = _send_tensors(tensor_list, peer_rank, comm)
    h.wait()


def isend_tensor(tensor_list, peer_rank, comm):
    return _send_tensors(tensor_list, peer_rank, comm)


def _send_tensors(tensor_list, peer_rank, comm):
    p2p_key = f"send {peer_rank}"
    if p2p_key not in _p2p_stream:
        _p2p_stream[p2p_key] = torch.cuda.Stream()
    if p2p_key not in _p2p_events:
        _p2p_events[p2p_key] = torch.cuda.Event()
    stream = _p2p_stream[p2p_key]
    event = _p2p_events[p2p_key]
    event.record(torch.cuda.current_stream())
    stream.wait_event(event)
    with torch.cuda.stream(stream):
        length = torch.tensor(
            data=[len([h for h in tensor_list])], device="cuda", dtype=torch.int
        )
        flags = torch.tensor(
            data=[0 for _ in range(len(tensor_list))],
            device="cuda",
            dtype=torch.int,
        )
        for i in range(len(tensor_list)):
            if tensor_list[i] is None:
                flag = -1
            elif torch.is_tensor(tensor_list[i]):
                flag = 0
            else:
                flag = 1
            flags[i] = flag
        comm.send(length, peer_rank)
        comm.send(flags.contiguous(), peer_rank)
        for i in range(len(tensor_list)):
            if flags[i] == 0:
                tensor_list[i].record_stream(stream)
                send_tensor(tensor_list[i], peer_rank, comm)
            elif flags[i] == 1:
                send_object(tensor_list[i], peer_rank, comm)
        event.record(stream)
    return handler(event)


def recv_tensors(peer_rank, comm):
    tensors, handle = _recv_tensors(peer_rank, comm)
    handle.wait()
    return tensors


def irecv_tensors(peer_rank, comm):
    tensors, handle = _recv_tensors(peer_rank, comm)
    return tensors, handle


def _recv_tensors(peer_rank, comm):
    p2p_key = f"recv {peer_rank}"
    if p2p_key not in _p2p_stream:
        _p2p_stream[p2p_key] = torch.cuda.Stream()
    if p2p_key not in _p2p_events:
        _p2p_events[p2p_key] = torch.cuda.Event()
    stream = _p2p_stream[p2p_key]
    event = _p2p_events[p2p_key]
    with torch.cuda.stream(stream):
        length = torch.tensor(data=[0], device="cuda", dtype=torch.int)
        tensor_list = []
        comm.recv(length, peer_rank)
        flags = torch.tensor(
            data=[0 for _ in range(int(length.item()))],
            device="cuda",
            dtype=torch.int,
        )
        comm.recv(flags, peer_rank)
        for i in range(length[0].item()):
            flag = flags[i].item()
            if flag == -1:
                tensor_list.append(None)
            elif flag == 0:
                tensor_list.append(recv_tensor(peer_rank, comm))
            elif flag == 1:
                tensor_list.append(recv_object(peer_rank, comm))
    event.record(stream)
    record_stream_helper(
        [tensor_list[i] for i in range(length[0].item()) if flags[i].item() != -1],
        torch.cuda.current_stream(),
    )
    return tensor_list, handler(event)


# ---- single-tensor send/recv ------------------------------------------------

def send_tensor(hidden_state, peer_rank, comm):
    hidden_state = hidden_state.contiguous()
    send_meta(hidden_state, peer_rank, comm)
    comm.send(hidden_state.view(-1), peer_rank)


def send_tensor_inplace(hidden_state, peer_rank, comm):
    hidden_state = hidden_state.contiguous()
    comm.send(hidden_state.view(-1), peer_rank)


def recv_tensor_inplace(hidden_state, peer_rank, comm):
    hidden_state = hidden_state.contiguous()
    comm.recv(hidden_state.view(-1), peer_rank)
    return hidden_state


def recv_tensor(peer_rank, comm):
    dtype, shape = recv_meta(peer_rank, comm)
    hidden_state = torch.empty(shape, dtype=dtype, device="cuda")
    comm.recv(hidden_state.view(-1), peer_rank)
    return hidden_state


# ---- meta send/recv (small fixed-size descriptor) ---------------------------

def send_meta(x, peer_rank, comm):
    meta_data = torch.tensor(data=[0] * 50, device="cuda", dtype=torch.int)
    meta_data[0] = len(x.size())
    meta_data[1] = DTYPE_LIST.index(x.dtype)
    meta_data[2 : len(x.size()) + 2] = torch.tensor(
        x.size(), device="cuda", dtype=torch.int
    )
    meta_data = meta_data.contiguous()
    comm.send(meta_data, peer_rank)


def recv_meta(peer_rank, comm):
    meta_data = torch.tensor(data=[0] * 50, device="cuda", dtype=torch.int)
    comm.recv(meta_data, peer_rank)
    n_dims = meta_data[0].item()
    dtype = DTYPE_LIST[meta_data[1].item()]
    shape = meta_data[2 : n_dims + 2].tolist()
    return dtype, shape
