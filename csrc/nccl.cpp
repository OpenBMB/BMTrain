
#include <torch/extension.h>
#include "include/nccl.h"
#include <ATen/cuda/CUDAContext.h>

void checkNCCLStatus(ncclResult_t result) {
    if (result == ncclSuccess) return;
    throw std::logic_error(
        std::string("NCCL Error: ") +
        ncclGetErrorString(result)
    );
}

py::bytes pyNCCLGetUniqueID() {
    ncclUniqueId uniqueID;
    checkNCCLStatus(ncclGetUniqueId(&uniqueID));
    return py::bytes(uniqueID.internal, NCCL_UNIQUE_ID_BYTES);
}

std::uintptr_t pyNCCLCommInitRank(py::bytes byteUniqueID, int world_size, int rank) {
    ncclUniqueId uniqueID;
    std::memcpy(uniqueID.internal, std::string(byteUniqueID).c_str(), NCCL_UNIQUE_ID_BYTES);
    ncclComm_t comm;
    checkNCCLStatus(ncclCommInitRank(&comm, world_size, uniqueID, rank));
    return reinterpret_cast<std::uintptr_t>(comm);
}

void pyNCCLCommDestroy(std::uintptr_t ptrcomm) {
    ncclComm_t comm = reinterpret_cast<ncclComm_t>(ptrcomm);
    checkNCCLStatus(ncclCommDestroy(comm));
}

void pyNCCLAllGather(
    std::uintptr_t sendbuff,
    std::uintptr_t recvbuff,
    size_t sendcount,
    int datatype,
    std::uintptr_t comm,
    std::uintptr_t stream
) {
    checkNCCLStatus(ncclAllGather(
        reinterpret_cast<void*>(sendbuff),
        reinterpret_cast<void*>(recvbuff),
        sendcount, 
        static_cast<ncclDataType_t>(datatype),
        reinterpret_cast<ncclComm_t>(comm), 
        reinterpret_cast<cudaStream_t>(stream)
    ));
}

void pyNCCLAllReduce(
    std::uintptr_t sendbuff, 
    std::uintptr_t recvbuff, 
    size_t count, 
    int data_type,
    int op,
    std::uintptr_t comm,
    std::uintptr_t stream
) {
    checkNCCLStatus(ncclAllReduce(
        reinterpret_cast<void*>(sendbuff),
        reinterpret_cast<void*>(recvbuff),
        count, 
        static_cast<ncclDataType_t>(data_type),
        static_cast<ncclRedOp_t>(op), 
        reinterpret_cast<ncclComm_t>(comm), 
        reinterpret_cast<cudaStream_t>(stream)
    ));
}

void pyNCCLBroadcast(
    std::uintptr_t sendbuff,
    std::uintptr_t recvbuff,
    size_t count,
    int datatype,
    int root,
    std::uintptr_t comm,
    std::uintptr_t stream
) {
    checkNCCLStatus(ncclBroadcast(
        reinterpret_cast<void*>(sendbuff),
        reinterpret_cast<void*>(recvbuff),
        count, 
        static_cast<ncclDataType_t>(datatype),
        root,
        reinterpret_cast<ncclComm_t>(comm), 
        reinterpret_cast<cudaStream_t>(stream)
    ));
}

void pyNCCLReduce(
    std::uintptr_t sendbuff,
    std::uintptr_t recvbuff,
    size_t count,
    int datatype,
    int op,
    int root,
    std::uintptr_t comm,
    std::uintptr_t stream
) {
    checkNCCLStatus(ncclReduce(
        reinterpret_cast<void*>(sendbuff),
        reinterpret_cast<void*>(recvbuff),
        count, 
        static_cast<ncclDataType_t>(datatype),
        static_cast<ncclRedOp_t>(op),
        root,
        reinterpret_cast<ncclComm_t>(comm), 
        reinterpret_cast<cudaStream_t>(stream)
    ));
}

void pyNCCLReduceScatter(
    std::uintptr_t sendbuff,
    std::uintptr_t recvbuff,
    size_t recvcount,
    int datatype,
    int op,
    std::uintptr_t comm,
    std::uintptr_t stream
) {
    checkNCCLStatus(ncclReduceScatter(
        reinterpret_cast<void*>(sendbuff),
        reinterpret_cast<void*>(recvbuff),
        recvcount,
        static_cast<ncclDataType_t>(datatype),
        static_cast<ncclRedOp_t>(op),
        reinterpret_cast<ncclComm_t>(comm), 
        reinterpret_cast<cudaStream_t>(stream)
    ));
}

void pyNCCLGroupStart() {
    checkNCCLStatus(ncclGroupStart());
}

void pyNCCLGroupEnd() {
    checkNCCLStatus(ncclGroupEnd());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ncclGetUniqueId", &pyNCCLGetUniqueID, "nccl get unique ID");
    m.def("ncclCommInitRank", &pyNCCLCommInitRank, "nccl init rank");
    m.def("ncclCommDestroy", &pyNCCLCommDestroy, "nccl delete rank");
    m.def("ncclAllGather", &pyNCCLAllGather, "nccl all gather");
    m.def("ncclAllReduce", &pyNCCLAllReduce, "nccl all reduce");
    m.def("ncclBroadcast", &pyNCCLBroadcast, "nccl broadcast");
    m.def("ncclReduce", &pyNCCLReduce, "nccl reduce");
    m.def("ncclReduceScatter", &pyNCCLReduceScatter, "nccl reduce scatter");
    m.def("ncclGroupStart", &pyNCCLGroupStart, "nccl group start");
    m.def("ncclGroupEnd", &pyNCCLGroupEnd, "nccl group end");
}
