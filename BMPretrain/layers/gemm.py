import torch
from ..functions.gemm import GEMMFloat, GEMMInt8

def bmm(A : torch.Tensor, aT : bool, B : torch.Tensor, bT : bool, int8 : bool =False) -> torch.Tensor:
    assert A.ndim == 3
    assert B.ndim == 3
    if int8:
        return GEMMInt8.apply(A, aT, B, bT)
    else:
        return GEMMFloat.apply(A, aT, B, bT)