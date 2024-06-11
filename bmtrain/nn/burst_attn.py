import bmtrain as bmt
import torch
import math
from .burst_utils import (
    inter_normal_attn,
    inter_normal_attn_backward,
    inter_flash_attn_triton,
    inter_flash_attn_backward_triton,
    inter_flash_cuda_fwd,
    inter_flash_cuda_bwd,
)
from .burst_utils import triton_scale_out, record_stream, Ring


class OpBurstAttn(torch.autograd.Function):
    """
    for Normal Attention:
        q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    for Flash:
        q, k, v: [B, S, N, H] (batch_size, num_heads, sub_seqlen, head_dim)

    """

    @staticmethod
    def forward(
        ctx, q, k, v, softmax_scale=None, flash=None, causal=False, optimize_bwd_comm=False, return_softmax=False, bias=None
    ):
        assert flash in [None, "cuda", "triton"], "flash must be None, 'cuda', or 'triton'"
        assert bias is None or flash != "cuda", "Flash Attn cuda impl does not support bias"

        m_i = None
        acc_o = None
        lse_i = None
        ctx.optimize_bwd_comm = optimize_bwd_comm or flash != "cuda"
        ctx.causal = causal 
        ctx.has_bias = bias is not None
        if softmax_scale is None:
            ctx.softmax_scale = 1 / math.sqrt(q.shape[-1])
        else:
            ctx.softmax_scale = softmax_scale
        ctx.flash = None if flash not in ["cuda", "triton"] else flash
        if ctx.flash:
            forward_func = (
                inter_flash_attn_triton
                if ctx.flash == "triton"
                else inter_flash_cuda_fwd
            )
        else:
            forward_func = inter_normal_attn
        sp_count = bmt.config["sp_size"]
        burst_comm = Ring(
            bmt.config["sp_comm"], bmt.config["sp_rank"]
        )
        ctx.burst_comm = burst_comm 
        sp_rank = bmt.config["sp_rank"]

        for r in range(1, sp_count + 1):
            if causal and r > sp_rank + 1:
                continue
            causal_arg = causal if r == 1 else False
            bufs = burst_comm.ring_send_recv(k, v)
            burst_comm.commit()
            if ctx.flash:
                if ctx.flash == "triton":
                    acc_o, m_i, lse_i = forward_func(
                        q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, bias, causal=causal_arg
                    )
                else:
                    acc_o, lse_i = forward_func(
                        q, k, v, acc_o, lse_i, ctx.softmax_scale, causal=causal_arg
                    )
            else:
                acc_o, m_i, lse_i = forward_func(
                    q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, bias, causal=causal_arg
                )
            k, v = record_stream(*bufs)
            burst_comm.wait()

        if ctx.flash == "triton":
            acc_o = triton_scale_out(acc_o, m_i, lse_i)
        elif not ctx.flash:
            o_scale = torch.exp(m_i - lse_i)
            acc_o = acc_o * o_scale
        acc_o = acc_o.to(dtype=q.dtype)
        if flash == "cuda":
            lse_i = lse_i.transpose(1, 2)
        lse_i = lse_i.contiguous()
        save_tensor = (q, k, v, lse_i, acc_o) if bias is None else (q, k, v, lse_i, acc_o, bias)
        ctx.save_for_backward(*save_tensor)
        return acc_o if not return_softmax else (acc_o, lse_i)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            q, k, v, lse_i, o_i, bias = ctx.saved_tensors
        else:
            q, k, v, lse_i, o_i = ctx.saved_tensors 
            bias = None
        d_q = torch.zeros_like(q)
        d_k = torch.zeros_like(k)
        d_v = torch.zeros_like(v)
        if not ctx.optimize_bwd_comm:
            delta = o_i.contiguous()
        else:
            delta = (o_i * grad_output)
            if ctx.flash:
                delta = delta.to(torch.float32)
            delta = delta.sum(-1, keepdim=not ctx.flash)
            if ctx.flash:
                delta = delta.transpose(1, 2).contiguous()
            
        if ctx.flash:
            backward_func = (
                inter_flash_attn_backward_triton
                if ctx.flash == "triton"
                else inter_flash_cuda_bwd
            )
        else:
            backward_func = inter_normal_attn_backward

        burst_comm = ctx.burst_comm
        #i = bmt.config["sp_rank"]
        sp_count = bmt.config["sp_size"]
        dq = torch.zeros_like(d_q)
        for r in range(1, sp_count + 1):
            #j = (i + sp_count - r) % sp_count
            if ctx.causal and r > bmt.config['sp_rank']+1: 
                continue
            if r != sp_count:
                bufs = burst_comm.ring_send_recv(delta, grad_output, q, lse_i)
            if r != 1:
                dq_buf = burst_comm.ring_send_recv(d_q)
            burst_comm.commit()
            if ctx.flash == "cuda":
                backward_func(
                    grad_output,
                    q,
                    k,
                    v,
                    delta,
                    lse_i,
                    dq,
                    d_k,
                    d_v,
                    ctx.softmax_scale,
                    causal=ctx.causal and r == 1,
                )
            else:
                backward_func(
                    grad_output,
                    q,
                    k,
                    v,
                    delta,
                    lse_i,
                    dq,
                    d_k,
                    d_v,
                    ctx.softmax_scale,
                    bias, 
                    causal=ctx.causal and r == 1,
                )
            burst_comm.wait()
            if r != sp_count:
                delta, grad_output, q, lse_i = record_stream(*bufs)
                torch.cuda.current_stream().wait_stream(bmt.config["sp_stream"])
            if r != 1:
                (d_q,) = record_stream(*dq_buf)
                d_q += dq
            else:
                d_q = dq.clone().detach()

        (d_q,) = burst_comm.ring_send_recv(d_q)
        burst_comm.commit()
        burst_comm.wait()

        return d_q, d_k, d_v, None, None, None, None
