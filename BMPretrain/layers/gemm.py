import torch
import BMPretrain._c as C

class GEMMInt8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, aT, B, bT):
        scale_A = C.calc_scale(A, aT)
        scale_B = C.calc_scale(B, not bT)

        quantized_A = C.round(A, aT, scale_A)
        quantized_B = C.round(B, not bT, scale_B)
        ret = C.bmm(quantized_A, aT, quantized_B, bT)
        ret = C.scale(ret, scale_A, scale_B)

        # save backward
        bw_scale_A = C.calc_scale(A, not aT)
        bw_scale_B = C.calc_scale(B, bT)
        bw_quantized_A = C.round(A, not aT, bw_scale_A)
        bw_quantized_B = C.round(B, bT, bw_scale_B)
        ctx.save_for_backward(
            bw_scale_A, bw_quantized_A,
            bw_scale_B, bw_quantized_B
        )
        ctx.aT = aT
        ctx.bT = bT
        return ret

    @staticmethod
    def backward(ctx, grad_f : torch.Tensor):
        if not grad_f.is_contiguous():
            grad_f = grad_f.contiguous()

        bw_scale_A, bw_quantized_A, bw_scale_B, bw_quantized_B = ctx.saved_tensors
        aT, bT = ctx.aT, ctx.bT
        
        grad_scale_r = C.calc_scale(grad_f, False)
        grad_quantized_r = C.round(grad_f, False, grad_scale_r)

        if aT:
            grad_A = C.scale(C.bmm(bw_quantized_B, bT, grad_quantized_r, True), bw_scale_B, grad_scale_r)
        else:
            grad_A = C.scale(C.bmm(grad_quantized_r, False, bw_quantized_B, not bT), grad_scale_r, bw_scale_B)


        grad_scale_c = C.calc_scale(grad_f, True)
        grad_quantized_c = C.round(grad_f, True, grad_scale_c)

        if bT:
            grad_B = C.scale(C.bmm(grad_quantized_c, True, bw_quantized_A, aT), grad_scale_c, bw_scale_A)
        else:
            grad_B = C.scale(C.bmm(bw_quantized_A, not aT, grad_quantized_c, False), bw_scale_A, grad_scale_c)
        
        if bw_scale_A.size(0) == 1 and grad_A.size(0) > 1:
            grad_A = grad_A.sum(dim=0, keepdim=True)
        if bw_scale_B.size(0) == 1 and grad_B.size(0) > 1:
            grad_B = grad_B.sum(dim=0, keepdim=True)
    
        return grad_A, None, grad_B, None 
        


class GEMMFloat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, aT, B, bT):
        ctx.save_for_backward(A, B)
        ctx.aT = aT
        ctx.bT = bT

        return C.bmm(A, aT, B, bT)

    @staticmethod
    def backward(ctx, grad_f):
        aT = ctx.aT
        bT = ctx.bT
        A, B = ctx.saved_tensors
        if aT:
            grad_A = C.bmm(B, bT, grad_f, True)
        else:
            grad_A = C.bmm(grad_f, False, B, not bT)
        
        if bT:
            grad_B = C.bmm(grad_f, True, A, aT)
        else:
            grad_B = C.bmm(A, not aT, grad_f, False)
        
        if A.size(0) == 1 and grad_A.size(0) > 1:
            grad_A = grad_A.sum(dim=0, keepdim=True)
        if B.size(0) == 1 and grad_B.size(0) > 1:
            grad_B = grad_B.sum(dim=0, keepdim=True)

        return grad_A, None, grad_B, None 

def bmm(A : torch.Tensor, aT : bool, B : torch.Tensor, bT : bool, int8 : bool =False) -> torch.Tensor:
    assert A.ndim == 3
    assert B.ndim == 3
    if int8:
        return GEMMInt8.apply(A, aT, B, bT)
    else:
        return GEMMFloat.apply(A, aT, B, bT)