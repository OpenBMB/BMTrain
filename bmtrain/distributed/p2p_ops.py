import torch
from bmtrain import config
from ..nccl import reduceScatter as ncclReduceScatter
from ..nccl import send as ncclSend
from ..nccl import recv as ncclRecv
from ..nccl import groupStart,groupEnd
from .dtype import DTYPE_LIST
import pickle
import contextlib

_p2p_stream = {}
_p2p_events = {}

@contextlib.contextmanager
def groupcall():
    groupStart()
    yield
    groupEnd()
class handler:
    def __init__(self, event):
        self.event= event

    def wait(self):
        torch.cuda.current_stream().wait_event(self.event)

def send_object(obj, peer_rank, comm):
    data_bytes: bytes = pickle.dumps(obj)
    data_length: int = len(data_bytes)

    gpu_data_length = torch.tensor([data_length], device="cuda", dtype=torch.long)
    ncclSend(gpu_data_length.storage(), peer_rank, comm)
    byte_storage = torch.ByteStorage.from_buffer(data_bytes).cuda()
    ncclSend(byte_storage, peer_rank, comm)

def recv_object(peer_rank, comm):
    data_length = torch.tensor([0], device="cuda", dtype=torch.long)
    ncclRecv(data_length.storage(), peer_rank, comm)
    data_bytes_stor = torch.cuda.ByteStorage(data_length.item())
    ncclRecv(data_bytes_stor, peer_rank, comm)
    tensor = torch.ByteTensor(data_bytes_stor.cpu())
    data = pickle.loads(tensor.numpy().tobytes())
    return data

def record_stream_helper(tensor_list, stream):
    for t in tensor_list:
        t.record_stream(stream)

def send_tensors(tensor_list, peer_rank, comm):
    handler = _send_tensors(tensor_list, peer_rank, comm)
    handler.wait()

def isend_tensor(tensor_list, peer_rank, comm):
    return _send_tensors(tensor_list, peer_rank, comm)

def _send_tensors(tensor_list, peer_rank, comm):
    p2p_key = f"send {peer_rank}"
    if p2p_key not in _p2p_stream:
        _p2p_stream[p2p_key] = torch.cuda.Stream()
    if p2p_key not in _p2p_events: 
        _p2p_events[p2p_key] = torch.cuda.Event()
    stream = _p2p_stream[peer_rank]
    end_event = _p2p_events[p2p_key]
    with torch.cuda.stream(stream):
        length = torch.tensor(data=[len([h for h in tensor_list ])], device="cuda", dtype=torch.int)
        ncclSend(length.storage(), peer_rank, comm)
        flags = torch.tensor(data=[0 for _ in range(len(tensor_list))], device="cuda",dtype=torch.int)
        for i in range(len(tensor_list)):
            if tensor_list[i] is None:
                flag = -1
            elif torch.is_tensor(tensor_list[i]):
                flag = 0
            else:
                flag = 1
            flags[i] = flag
        ncclSend(flags.contiguous().storage(), peer_rank, comm)
        for i in range(len(tensor_list)):
            if flags[i] == 0:
                tensor_list[i].record_stream(stream)
                send_tensor(tensor_list[i], peer_rank, comm)
            elif flags[i] == 1:
                send_object(tensor_list[i], peer_rank, comm)
        end_event.record(stream)
    return handler(end_event)

def recv_tensors(peer_rank, comm):
    tensors,handle = _recv_tensors(peer_rank, comm)
    handle.wait()
    return tensors

def irecv_tensors(peer_rank, comm):
    tensors, handle = _recv_tensors(peer_rank, comm)
    return tensors, handle

def _recv_tensors(peer_rank, comm):
    p2p_key = f"recv {peer_rank}"
    if p2p_key not in _p2p_stream:
        _p2p_stream[peer_rank] = torch.cuda.Stream()
    if p2p_key not in _p2p_events:
        _p2p_events[p2p_key] = torch.cuda.Event()
    stream = _p2p_stream[peer_rank]
    end_event = _p2p_events[p2p_key]
    with torch.cuda.stream(stream):
        length = torch.tensor(data=[0], device="cuda", dtype=torch.int)
        tensor_list = []
        ncclRecv(length.storage(), peer_rank, comm)
        flags = torch.tensor(data=[0 for _ in range(length)], device="cuda",dtype=torch.int)
        ncclRecv(flags.storage(), peer_rank, comm)
        for i in range(length[0].item()):
            flag = flags[i].item()
            if flag == -1:
                tensor_list.append(None)
            elif flag == 0:
                recv = recv_tensor(peer_rank, comm)
                tensor_list.append(recv)
            elif flag == 1:
                recv = recv_object(peer_rank, comm)
                tensor_list.append(recv)
        end_event.record(stream)
    record_stream_helper([tensor_list[i] for i in range(length[0]).item() if flags[i].item() == 0], torch.cuda.current_stream())
    return tensor_list, handler(end_event)

def send_tensor(hidden_state, peer_rank, comm):
    hidden_state = hidden_state.contiguous()
    send_meta(hidden_state, peer_rank, comm)
    ncclSend(hidden_state.storage(), peer_rank, comm)

def send_tensor_inplace(hidden_state, peer_rank, comm):
    hidden_state = hidden_state.contiguous()
    ncclSend(hidden_state.storage(), peer_rank, comm)

def recv_tensor_inplace(hidden_state, peer_rank, comm):
    hidden_state = hidden_state.contiguous()
    ncclRecv(hidden_state.storage(), peer_rank, comm)
    return hidden_state

def recv_tensor(peer_rank, comm):
    dtype, shape = recv_meta(peer_rank, comm)
    hidden_state = torch.empty(shape, dtype=dtype, device="cuda")
    ncclRecv(hidden_state.storage(), peer_rank, comm)
    return hidden_state

def send_meta(x, peer_rank, comm):
    meta_data = torch.tensor(data=[0]*50, device="cuda", dtype=torch.int)
    meta_data[0] = len(x.size())
    meta_data[1] = DTYPE_LIST.index(x.dtype)
    meta_data[2:len(x.size())+2] = torch.tensor(x.size(), device="cuda", dtype=torch.int)
    meta_data = meta_data.contiguous()
    ncclSend(meta_data.storage(), peer_rank, comm)

def recv_meta(peer_rank, comm):
    meta_data = torch.tensor(data=[0]*50, device="cuda", dtype=torch.int)
    ncclRecv(meta_data.storage(), peer_rank, comm)
    n_dims = meta_data[0].item()
    dtype = DTYPE_LIST[meta_data[1].item()]
    shape = meta_data[2:n_dims+2].tolist()

    return dtype,shape