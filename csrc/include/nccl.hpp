#include <cstdint>
#include <string>
#include <pybind11/pybind11.h>
#include "watch_dog.hpp"

namespace py = pybind11;
#include <nccl.h>

WatchDog* watchDog = nullptr;

void checkNCCLStatus(ncclResult_t result) {
#if NCCL_VERSION_CODE >= 21403
    if (result == ncclSuccess || result == ncclInProgress)
#else
    if (result == ncclSuccess)
#endif
        return;
    throw std::logic_error(
        std::string("NCCL Error: ") + ncclGetErrorString(result));
}

py::bytes pyNCCLGetUniqueID() {
    ncclUniqueId uniqueID;
    checkNCCLStatus(ncclGetUniqueId(&uniqueID));
    return py::bytes(uniqueID.internal, NCCL_UNIQUE_ID_BYTES);
}

std::uintptr_t pyNCCLCommInitRank(
    py::bytes byteUniqueID,
    int world_size,
    int rank,
    int timeout,
    bool non_blocking) {
    ncclUniqueId uniqueID;
    std::memcpy(uniqueID.internal, std::string(byteUniqueID).c_str(), NCCL_UNIQUE_ID_BYTES);
    ncclComm_t comm;

#if NCCL_VERSION_CODE >= 21403
    if (!non_blocking) {
        checkNCCLStatus(ncclCommInitRank(&comm, world_size, uniqueID, rank));
    } else {
        std::chrono::milliseconds timeout_(timeout);
        std::chrono::time_point<std::chrono::steady_clock> startTime_(std::chrono::steady_clock::now());
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = 0;

        checkNCCLStatus(ncclCommInitRankConfig(&comm, world_size, uniqueID, rank, &config));
        ncclResult_t state;
        do {
            checkNCCLStatus(ncclCommGetAsyncError(comm, &state));
            if (std::chrono::steady_clock::now() - startTime_ > timeout_) {
                ncclCommAbort(comm);
                throw std::runtime_error("NCCL timeout");
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(1));
        } while (state == ncclInProgress);
        if (state != ncclSuccess) {
            ncclCommAbort(comm);
            throw std::logic_error(
                std::string("NCCL Error: ") + ncclGetErrorString(state));
        }
    }
#else
    checkNCCLStatus(ncclCommInitRank(&comm, world_size, uniqueID, rank));
#endif
    if (!watchDog)
        watchDog = new WatchDog(timeout);
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
    std::uintptr_t stream) {
    CUDAEvent event(reinterpret_cast<cudaStream_t>(stream), "NCCLAllGather");
    event.recordStart();
    checkNCCLStatus(ncclAllGather(
        reinterpret_cast<void*>(sendbuff),
        reinterpret_cast<void*>(recvbuff),
        sendcount,
        static_cast<ncclDataType_t>(datatype),
        reinterpret_cast<ncclComm_t>(comm),
        reinterpret_cast<cudaStream_t>(stream)));
    event.recordEnd();

    watchDog->watch(event);
}

void pyNCCLAllReduce(
    std::uintptr_t sendbuff,
    std::uintptr_t recvbuff,
    size_t count,
    int data_type,
    int op,
    std::uintptr_t comm,
    std::uintptr_t stream) {
    CUDAEvent event(reinterpret_cast<cudaStream_t>(stream), "NCCLAllReduce");
    event.recordStart();
    checkNCCLStatus(ncclAllReduce(
        reinterpret_cast<void*>(sendbuff),
        reinterpret_cast<void*>(recvbuff),
        count,
        static_cast<ncclDataType_t>(data_type),
        static_cast<ncclRedOp_t>(op),
        reinterpret_cast<ncclComm_t>(comm),
        reinterpret_cast<cudaStream_t>(stream)));
    event.recordEnd();
    watchDog->watch(event);
}

void pyNCCLBroadcast(
    std::uintptr_t sendbuff,
    std::uintptr_t recvbuff,
    size_t count,
    int datatype,
    int root,
    std::uintptr_t comm,
    std::uintptr_t stream) {
    CUDAEvent event(reinterpret_cast<cudaStream_t>(stream), "NCCLBroadcast");
    event.recordStart();
    checkNCCLStatus(ncclBroadcast(
        reinterpret_cast<void*>(sendbuff),
        reinterpret_cast<void*>(recvbuff),
        count,
        static_cast<ncclDataType_t>(datatype),
        root,
        reinterpret_cast<ncclComm_t>(comm),
        reinterpret_cast<cudaStream_t>(stream)));
    event.recordEnd();
    watchDog->watch(event);
}

void pyNCCLReduce(
    std::uintptr_t sendbuff,
    std::uintptr_t recvbuff,
    size_t count,
    int datatype,
    int op,
    int root,
    std::uintptr_t comm,
    std::uintptr_t stream) {
    CUDAEvent event(reinterpret_cast<cudaStream_t>(stream), "NCCLReduce");
    event.recordStart();
    checkNCCLStatus(ncclReduce(
        reinterpret_cast<void*>(sendbuff),
        reinterpret_cast<void*>(recvbuff),
        count,
        static_cast<ncclDataType_t>(datatype),
        static_cast<ncclRedOp_t>(op),
        root,
        reinterpret_cast<ncclComm_t>(comm),
        reinterpret_cast<cudaStream_t>(stream)));
    event.recordEnd();
    watchDog->watch(event);
}

void pyNCCLReduceScatter(
    std::uintptr_t sendbuff,
    std::uintptr_t recvbuff,
    size_t recvcount,
    int datatype,
    int op,
    std::uintptr_t comm,
    std::uintptr_t stream) {
    CUDAEvent event(reinterpret_cast<cudaStream_t>(stream), "NCCLReduceScatter");
    event.recordStart();
    checkNCCLStatus(ncclReduceScatter(
        reinterpret_cast<void*>(sendbuff),
        reinterpret_cast<void*>(recvbuff),
        recvcount,
        static_cast<ncclDataType_t>(datatype),
        static_cast<ncclRedOp_t>(op),
        reinterpret_cast<ncclComm_t>(comm),
        reinterpret_cast<cudaStream_t>(stream)));
    event.recordEnd();
    watchDog->watch(event);
}

void pyNCCLSend(
    std::uintptr_t sendbuff,
    size_t sendcount,
    int data_type,
    int peer,
    std::uintptr_t comm,
    std::uintptr_t stream) {
    CUDAEvent event(reinterpret_cast<cudaStream_t>(stream), "NCCLSend");
    event.recordStart();
    checkNCCLStatus(ncclSend(
        reinterpret_cast<void*>(sendbuff),
        sendcount,
        static_cast<ncclDataType_t>(data_type),
        peer,
        reinterpret_cast<ncclComm_t>(comm),
        reinterpret_cast<cudaStream_t>(stream)));
    event.recordEnd();
    watchDog->watch(event);
}

void pyNCCLRecv(
    std::uintptr_t recvbuff,
    size_t recvcount,
    int data_type,
    int peer,
    std::uintptr_t comm,
    std::uintptr_t stream) {
    CUDAEvent event(reinterpret_cast<cudaStream_t>(stream), "NCCLRecv");
    event.recordStart();
    checkNCCLStatus(ncclRecv(
        reinterpret_cast<void*>(recvbuff),
        recvcount,
        static_cast<ncclDataType_t>(data_type),
        peer,
        reinterpret_cast<ncclComm_t>(comm),
        reinterpret_cast<cudaStream_t>(stream)));
    event.recordEnd();
    watchDog->watch(event);
}
void pyNCCLGroupStart() {
    checkNCCLStatus(ncclGroupStart());
}

void pyNCCLGroupEnd() {
    checkNCCLStatus(ncclGroupEnd());
}

int pyNCCLCommCount(
    std::uintptr_t comm) {
    int res;
    checkNCCLStatus(ncclCommCount(reinterpret_cast<ncclComm_t>(comm), &res));
    return res;
}

int pyNCCLCommUserRank(
    std::uintptr_t comm) {
    int rank;
    checkNCCLStatus(ncclCommUserRank(reinterpret_cast<ncclComm_t>(comm), &rank));
    return rank;
}
