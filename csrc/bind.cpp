#include "include/bind.hpp"

PYBIND11_MODULE(C, m) {
    m.def("has_nan_inf_launcher",&has_nan_inf_launcher,"has nan inf");
    m.def("adam_launcher", &adam_launcher, "adam function cpu");
    m.def("adam_bf16_launcher", &adam_bf16_launcher, "adam function cpu");
    m.def("adam_cpu_launcher", &adam_cpu_launcher, "adam function cpu");
    m.def("cross_entropy_forward_launcher", &cross_entropy_forward_launcher, "cross entropy forward");
    m.def("cross_entropy_backward_launcher", &cross_entropy_backward_launcher, "cross entropy backward");
    m.def("cross_entropy_forward_inplace_launcher", &cross_entropy_forward_inplace_launcher, "cross entropy forward inplace");
    m.def("cross_entropy_backward_inplace_launcher", &cross_entropy_backward_inplace_launcher, "cross entropy backward inplace");
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
    m.def("ncclSend",&pyNCCLSend,"nccl send");
    m.def("ncclRecv",&pyNCCLRecv,"nccl recv");
    m.def("ncclCommCount",&pyNCCLCommCount,"nccl comm count");
    m.def("ncclCommUserRank",&pyNCCLCommUserRank,"nccl comm user rank");
}
