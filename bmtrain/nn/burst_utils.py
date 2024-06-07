import bmtrain as bmt
import torch
from flash_attn.flash_attn_interface import (
    _flash_attn_forward as _flash_attn_forward_cuda,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_backward as _flash_attn_backward_cuda,
)
import inspect

class ops_wrapper:
    def __init__(self, op, tensor, *args, **kwargs):
        self.op = op
        self.tensor = tensor
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.op(self.tensor.storage(), *self.args, **self.kwargs)

class Ring:
    def __init__(self, comm, rank):
        self.comm = comm
        self.rank = rank
        self.reqs = []
        self.ops = []

    def ring_send_recv(self, *tensor_list):
        comm = self.comm
        rank = self.rank
        count = bmt.nccl.commCount(comm)
        next_rank = (rank + 1) % count
        prev_rank = (rank - 1 + count) % count
        output = []
        i = 0
        for tensor in tensor_list:
            i += 1
            res = torch.zeros_like(tensor)
            send_op = ops_wrapper(bmt.nccl.send, tensor, next_rank, comm)
            recv_op = ops_wrapper(bmt.nccl.recv, res, prev_rank, comm)
            self.ops.append(send_op)
            self.ops.append(recv_op)
            output.append(res)
        return output

    def commit(self):
        torch.cuda.synchronize()
        with torch.cuda.stream(bmt.config["sp_stream"]):
            for op in self.ops:
                op.tensor.record_stream(bmt.config["sp_stream"])
            bmt.nccl.groupStart()
            for op in self.ops:
                op()
            bmt.nccl.groupEnd()
        reqs = None
        self.reqs = reqs

    def wait(self):
        torch.cuda.current_stream().wait_stream(bmt.config["sp_stream"])
        self.reqs = []
        self.ops = []

@torch.jit.script
def triton_scale_out(acc_o, m_i, lse_i):
    o_scale = torch.exp(m_i - lse_i)
    o_scale = o_scale.unsqueeze(-1).transpose(1, 2)
    acc_o = acc_o * o_scale
    return acc_o


@torch.jit.script
def cuda_scale_out_lse_helper(
    o,
    lse,
    o_i,
    lse_i,
):
    o_i = o_i.to(torch.float32)
    lse_i = lse_i.transpose(-2, -1).unsqueeze(dim=-1).contiguous()
    new_lse = lse + torch.log(1 + torch.exp(lse_i - lse))
    o = torch.exp(lse - new_lse) * o + torch.exp(lse_i - new_lse) * o_i

    lse = new_lse
    return o, lse


def record_stream(*tensorlist):
    for t in tensorlist:
        t.record_stream(torch.cuda.current_stream())
    return tensorlist


def inter_normal_attn(q, k, v, m_i, lse_i, acc_o, softmax_scale=1.0, mask_bias=None):
    m_i = m_i.to(q.dtype) if m_i is not None else None
    qk = q @ k.transpose(-2, -1) * softmax_scale
    if mask_bias is not None:
        qk = torch.masked_fill(
            qk,
            not mask_bias,
            torch.scalar_tensor(float("-10000"), device=qk.device, dtype=qk.dtype),
        )

    m_ij = torch.max(qk, dim=-1, keepdim=True)[0]
    if m_i is not None:
        m_ij = torch.maximum(m_ij, m_i)
    p = torch.exp(qk - m_ij)
    if mask_bias is not None:
        p = torch.masked_fill(
            p,
            not mask_bias,
            torch.scalar_tensor(float("0"), device=qk.device, dtype=qk.dtype),
        )
    l_ij = torch.sum(p, dim=-1, keepdim=True)
    if acc_o is not None:
        acc_o_scale = torch.exp(m_i - m_ij)
        pv = (p @ v).to(dtype=torch.float32)
        acc_o = pv + acc_o_scale * acc_o
    else:
        acc_o = (p @ v).to(dtype=torch.float32)

    if lse_i is None:
        lse_i = torch.log(l_ij + 1e-5) + m_ij
    else:
        lse_i = torch.log(torch.exp(lse_i - m_ij) + l_ij + 1e-5) + m_ij
    return acc_o, m_ij, lse_i


def inter_normal_attn_backward(
    do, q, k, v, delta, lse, d_q, d_k, d_v, softmax_scale, mask_bias
):
    # ensure q,k,v with shape [b,n,s,d]
    qk = q @ k.transpose(-2, -1) * softmax_scale
    if mask_bias is not None:
        qk = torch.masked_fill(
            qk,
            not mask_bias,
            torch.scalar_tensor(float("-10000"), device=qk.device, dtype=qk.dtype),
        )
    p = torch.exp(qk - lse)
    if mask_bias is not None:
        p = torch.masked_fill(
            p,
            not mask_bias,
            torch.scalar_tensor(float("0"), device=qk.device, dtype=qk.dtype),
        )
    d_v += p.transpose(-2, -1) @ do
    d_p = do @ v.transpose(-2, -1)
    softmax_scale = softmax_scale
    d_s = p * (d_p - delta) * softmax_scale
    d_q[:] = d_s @ k
    d_k += d_s.transpose(-2, -1) @ q


def inter_flash_attn_triton(
    q, k, v, m_i, lse_i, acc_o, softmax_scale=1.0, mask_bias=None
):
    from .burst_lao import _flash_attn_forward
    b, s, n, d = q.shape
    if m_i is None:
        m_i = (
            -torch.ones((b, n, s), dtype=torch.float32, device="cuda") * torch.inf
        ).contiguous()
    if lse_i is None:
        lse_i = (
            -torch.ones((b, n, s), dtype=torch.float32, device="cuda") * torch.inf
        ).contiguous()
    if acc_o is None:
        acc_o = torch.zeros((b, s, n, d), dtype=torch.float32, device="cuda")
    acc_o, lse_i, m_ij, softamx_scale = _flash_attn_forward(
        q,
        k,
        v,
        m_i,
        lse_i,
        acc_o.to(dtype=torch.float32),
        causal=False,
        bias=mask_bias,
        softmax_scale=softmax_scale,
    )
    return acc_o, m_ij, lse_i


def inter_flash_attn_backward_triton(
    do, q, k, v, delta, lse, dq, dk, dv, softmax_scale, mask_bias
):
    from .burst_lao import _flash_attn_backward
    # dq_ = torch.empty_like(q)
    dk_ = torch.empty_like(q)
    dv_ = torch.empty_like(q)
    _flash_attn_backward(
        do,
        q,
        k,
        v,
        delta,
        lse,
        dq,
        dk_,
        dv_,
        softmax_scale=softmax_scale,
        bias=mask_bias,
    )
    # dq += dq_
    dk += dk_
    dv += dv_


def inter_flash_cuda_fwd(q, k, v, o, lse, softmax_scale=1.0):
    o_i, _, _, _, _, lse_i, _, _ = _flash_attn_forward_cuda(
        q,
        k,
        v,
        0.0,
        softmax_scale,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_softmax=False,
    )
    if o is None:
        o = o_i.to(torch.float32)
        lse = lse_i.transpose(-2, -1).unsqueeze(dim=-1).contiguous()
    else:
        o, lse = cuda_scale_out_lse_helper(o, lse, o_i, lse_i)
    return o, lse


def inter_flash_cuda_bwd(do, q, k, v, o, lse, dq, dk, dv, softmax_scale, mask_bias):
    dk_ = torch.empty_like(q)
    dv_ = torch.empty_like(q)
    if len(o.shape) == 3:
        # use sum(o_i * gradoutput) as delta and pass a empty out to flash backward
        # this feature requires a build of this PR: https://github.com/Dao-AILab/flash-attention/pull/905
        delta = o
        o = q
    elif len(o.shape) == 4:
        delta = None
    if delta is not None:
        assert inspect.signature(_flash_attn_backward_cuda).parameters.get(
            "softmax_d"
        ), "optimize_bwd_comm is not supported for this version of flash-attention, \
            you have to compile flash-attention with this PR: \
            https://github.com/Dao-AILab/flash-attention/pull/905"
        _flash_attn_backward_cuda(
            do,
            q,
            k,
            v,
            o,
            lse,
            dq,
            dk_,
            dv_,
            0.0,
            softmax_scale,
            False,
            (-1, -1),
            None,
            False,
            None,
            delta,
        )
    else:
        _flash_attn_backward_cuda(
            do,
            q,
            k,
            v,
            o,
            lse,
            dq,
            dk_,
            dv_,
            0.0,
            softmax_scale,
            False,
            (-1, -1),
            None,
            False,
            None,
        )
    # dq += dq_
    dk += dk_
    dv += dv_
