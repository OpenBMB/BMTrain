
from typing_extensions import Literal
import torch
from . import _C as C
from .enums import *

class NCCLCommunicator:
    def __init__(self, ptr) -> None:
        self.__ptr = ptr
    
    @property
    def ptr(self):
        if self.__ptr == -1:
            raise RuntimeError("NCCL Communicator is already destroyed")
        return self.__ptr
    
    def _destroy_ptr(self):
        self.__ptr = -1

# utils

def dtype2nccl(dtype : torch.dtype) -> int:
    MAP = {
        torch.int8: ncclInt8,
        torch.uint8 : ncclUint8,
        torch.int32 : ncclInt32,
        torch.int : ncclInt32,
        torch.int64 : ncclInt64,
        torch.float16 : ncclFloat16,
        torch.half : ncclHalf,
        torch.float32 : ncclFloat32,
        torch.float : ncclFloat,
        torch.float64 : ncclFloat64,
        torch.double : ncclDouble
    }
    if dtype not in MAP:
        raise TypeError("Unsupport dtype %s" % dtype)
    return MAP[dtype]

def op2nccl(
    op : Literal["sum", "prod", "max", "min", "avg"]
):
    if op == "sum":
        return ncclSum
    if op == "prod":
        return ncclProd
    if op == "max":
        return ncclMax
    if op == "min":
        return ncclMin
    if op == "avg":
        return ncclAvg
    raise ValueError("Unknown gather op %s")

# wrappers

def getUniqueId() -> bytes:
    return C.ncclGetUniqueId()

def commInitRank(unique_id : bytes, world_size : int, rank : int) -> NCCLCommunicator:
    assert rank >= 0 and rank < world_size, "rank must be between 0 and world_size-1"
    return NCCLCommunicator(C.ncclCommInitRank(unique_id, world_size, rank))

def commDestroy(comm : NCCLCommunicator):
    C.ncclCommDestroy(comm.ptr)
    comm._destroy_ptr()

### collective

def allReduce(
        src : torch.storage._StorageBase,
        dst : torch.storage._StorageBase,
        op : Literal["sum", "prod", "max", "min", "avg"],
        comm : NCCLCommunicator
    ):
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.is_cuda and dst.is_cuda

    sendbuff = src.data_ptr()
    recvbuff = dst.data_ptr()
    count = src.size()
    datatype = dtype2nccl(src.dtype)
    operator = op2nccl(op)

    assert src.size() == dst.size(), "Buffer size not aligned"
    C.ncclAllReduce(
        sendbuff,
        recvbuff,
        count,
        datatype,
        operator,
        comm.ptr,
        torch.cuda.current_stream().cuda_stream
    )

def broadcast(
        src : torch.storage._StorageBase,
        dst : torch.storage._StorageBase,
        root : int,
        comm : NCCLCommunicator
    ):
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.is_cuda and dst.is_cuda

    sendbuff = src.data_ptr()
    recvbuff = dst.data_ptr()
    count = src.size()
    datatype = dtype2nccl(src.dtype)

    assert dst.size() == src.size(), "Buffer size not aligned"
    C.ncclBroadcast(
        sendbuff, 
        recvbuff, 
        count, 
        datatype, 
        root, 
        comm.ptr, 
        torch.cuda.current_stream().cuda_stream
    )

def reduce(
        src : torch.storage._StorageBase,
        dst : torch.storage._StorageBase,
        op : Literal["sum", "prod", "max", "min", "avg"],
        root : int,
        comm : NCCLCommunicator
    ):
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.is_cuda and dst.is_cuda

    sendbuff = src.data_ptr()
    recvbuff = dst.data_ptr()
    count = src.size()
    datatype = dtype2nccl(src.dtype)
    operator = op2nccl(op)

    assert dst.size() == src.size(), "Buffer size not aligned"
    C.ncclReduce(sendbuff, recvbuff, count, datatype, operator, root, comm.ptr, torch.cuda.current_stream().cuda_stream)

def allGather(
        src : torch.storage._StorageBase,
        dst : torch.storage._StorageBase,
        comm : NCCLCommunicator
    ):
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.is_cuda and dst.is_cuda

    sendbuff = src.data_ptr()
    recvbuff = dst.data_ptr()
    sendcount = src.size()
    datatype = dtype2nccl(src.dtype)

    assert dst.size() % sendcount == 0, "Buffer size not aligned"
    C.ncclAllGather(
        sendbuff, 
        recvbuff, 
        sendcount, 
        datatype, 
        comm.ptr, 
        torch.cuda.current_stream().cuda_stream
    )


def reduceScatter(
        src : torch.storage._StorageBase,
        dst : torch.storage._StorageBase,
        op : Literal["sum", "prod", "max", "min", "avg"],
        comm : NCCLCommunicator
    ):
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.is_cuda and dst.is_cuda

    sendbuff = src.data_ptr()
    recvbuff = dst.data_ptr()
    recvcount = dst.size()
    datatype = dtype2nccl(src.dtype)
    operator = op2nccl(op)

    assert src.size() % recvcount == 0, "Buffer size not aligned"
    C.ncclReduceScatter(
        sendbuff,
        recvbuff,
        recvcount,
        datatype,
        operator,
        comm.ptr,
        torch.cuda.current_stream().cuda_stream
    )

def groupStart():
    C.ncclGroupStart()

def groupEnd():
    C.ncclGroupEnd()