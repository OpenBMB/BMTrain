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
        ctx, q, k, v, softmax_scale=None, flash=None, optimize_bwd_comm=False, return_softmax=False
    ):
        m_i = None
        acc_o = None
        lse_i = None
        ctx.optimize_bwd_comm = optimize_bwd_comm or flash is None
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

        for r in range(1, sp_count + 1):
            bufs = burst_comm.ring_send_recv(k, v)
            burst_comm.commit()
            if ctx.flash:
                if ctx.flash == "triton":
                    acc_o, m_i, lse_i = forward_func(
                        q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, None
                    )
                else:
                    acc_o, lse_i = forward_func(
                        q, k, v, acc_o, lse_i, ctx.softmax_scale
                    )
            else:
                acc_o, m_i, lse_i = forward_func(
                    q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, None
                )
            k, v = record_stream(*bufs)
            burst_comm.wait()

        if ctx.flash == "triton":
            acc_o = triton_scale_out(acc_o, m_i, lse_i)
        elif not ctx.flash:
            o_scale = torch.exp(m_i - lse_i)
            acc_o = acc_o * o_scale
        acc_o = acc_o.to(dtype=q.dtype)
        if flash is not None:
            lse_i = lse_i.squeeze(dim=-1).transpose(1, 2).contiguous()
        ctx.save_for_backward(q, k, v, lse_i.contiguous(), acc_o)
        return acc_o if not return_softmax else (acc_o, lse_i)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse_i, o_i = ctx.saved_tensors
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

            if r != sp_count:
                bufs = burst_comm.ring_send_recv(delta, grad_output, q, lse_i)
            if r != 1:
                dq_buf = burst_comm.ring_send_recv(d_q)
            burst_comm.commit()
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
                None,
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
