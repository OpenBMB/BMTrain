import torch
import torch.nn.functional as F
import BMPretrain._c as C

cublas_handle = C.i8_create_handle()

@torch.jit.script
def round_to_int8(x, scale):
    return torch.round(x / scale).to(torch.int8)

@torch.jit.script
def calc_scale(x):
    return x.abs().max(dim=-1, keepdim=True)[0] / 127

class ScaleQuantizedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x : torch.Tensor, scale_x : torch.Tensor, weight : torch.Tensor, scale_weight : torch.Tensor):
        quantized_x = round_to_int8(x, scale_x)
        quantized_weight = round_to_int8(weight, scale_weight)
        out = C.i8_forward(cublas_handle, quantized_x, quantized_weight[None])
        ret = C.i8_scale_2d(out, scale_x[:, :, 0], scale_weight[None, :, 0])

        del quantized_weight
        del quantized_x
        del out
        
        xT = x.transpose(-1, -2)
        weightT = weight.transpose(-1, -2)
        bw_scale_x = calc_scale(xT)
        bw_scale_weight = calc_scale( weightT )
        bw_quantized_xT = round_to_int8(xT, bw_scale_x).contiguous()
        bw_quantized_wT = round_to_int8( weightT, bw_scale_weight ).contiguous()
        ctx.save_for_backward(bw_quantized_xT, bw_scale_x, bw_quantized_wT, bw_scale_weight)
        return ret
    
    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        bw_quantized_xT, bw_scale_xT, bw_quantized_wT, bw_scale_wT = ctx.saved_tensors
        g = grad_outputs
        scale_g = calc_scale(g)
        quantized_g = round_to_int8(g, scale_g)
        grad_x = C.i8_scale_2d(
            C.i8_forward(cublas_handle, quantized_g, bw_quantized_wT[None]),
            scale_g[:, :, 0],
            bw_scale_wT[None, :, 0]
        )
        del quantized_g
        del scale_g

        gT = g.transpose(-1, -2).contiguous()
        scale_gT = calc_scale(gT)
        quantized_gT = round_to_int8(gT, scale_gT)
        grad_w = C.i8_scale_2d(
            C.i8_forward(cublas_handle, quantized_gT, bw_quantized_xT),
            scale_gT[:, :, 0],
            bw_scale_xT[:, :, 0],
        )

        return grad_x, None, grad_w.sum(dim=0), None

def simple_quantized_linear(x : torch.Tensor, weight : torch.Tensor) -> torch.Tensor:
    scale_x = calc_scale(x)
    scale_weight = calc_scale(weight)
    return ScaleQuantizedLinear.apply(x, scale_x, weight, scale_weight)
