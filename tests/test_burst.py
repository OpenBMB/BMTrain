import torch
import bmtrain as bmt
from flash_attn.flash_attn_interface import flash_attn_func as flash_cuda
import numpy as np

OpBurstAttn = bmt.nn.OpBurstAttn
def ref_attn(q, k, v):
    scale = q.shape[-1] ** -0.5
    s = q @ k.transpose(-2, -1) * scale
    s = torch.softmax(s, dim=-1)
    p = s @ v
    return p


def flash(q, k, v):
    return flash_cuda(q, k, v, causal=False, softmax_scale=None)


def burst(q, k, v):
    res_burst = OpBurstAttn.apply(q, k, v, None, None, False)
    return res_burst

def test_func(q, k, v, func, grad_output):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    grad_output = grad_output.contiguous()
    o = func(q, k, v)
    gq, gk, gv = torch.autograd.grad(o, (q, k, v), grad_output)
    return o, (gq, gk, gv)

def test_burst():
    dtype = torch.float16
    bmt.init_distributed(sp_size=4)
    flash = None
    seq_dim = 2 if not flash else 1
    def get_chunk(t, dim):
        return t.chunk(bmt.config['sp_size'], dim=dim)[bmt.config['sp_rank']].contiguous()

    b, s, n, d = 2, 4096, 16, 32
    if bmt.config["sp_rank"] == 0:
        qkv = torch.randn(b, n*3, s, d, dtype=dtype).cuda()
        grad_output = torch.randn(b, n, s, d, dtype=dtype).cuda()
        torch.save(qkv, "./qkv.pt")
        torch.save(grad_output, "./grad.pt")
    bmt.synchronize()
    qkv = torch.load("qkv.pt", map_location="cuda")
    grad_output = torch.load("grad.pt", map_location="cuda")
    qkv1 = [t.clone().detach().requires_grad_() for t in qkv.chunk(3, dim=1)]

    o_ref, g_ref = test_func(qkv1[0], qkv1[1], qkv1[2], ref_attn, grad_output)
    for i in range(3):
        if flash is not None:
            qkv1[i] = qkv1[i].transpose(1, 2)
        qkv1[i] = qkv1[i].chunk(bmt.world_size(), dim=seq_dim)[bmt.rank()]
        qkv1[i] = qkv1[i].clone().detach().requires_grad_()
    if flash is not None:
        grad_output = grad_output.transpose(1, 2)

    grad_output = (
        grad_output.chunk(bmt.world_size(), dim=seq_dim)[bmt.rank()]
        .clone()
        .detach()
        .contiguous()
    )
    o1, grad_qkv1 = test_func(qkv1[0], qkv1[1], qkv1[2], burst, grad_output)
    if flash:
        o1 = o1.transpose(1, 2).contiguous()
        grad_qkv1 = [g.transpose(1, 2).contiguous() for g in grad_qkv1]
    o_ref = get_chunk(o_ref, dim=2)
    g_ref = [get_chunk(g, dim=2) for g in g_ref]
    np.testing.assert_allclose(
        o1.detach().cpu().numpy(),
        o_ref.detach().cpu().numpy(),
        atol=1e-2,
        rtol=0,
    )
    for i in range(3):
        falsh_g_rank = g_ref[i].detach().cpu().numpy()
        burst_g_rank = grad_qkv1[i].detach().cpu().numpy()
        np.testing.assert_allclose(falsh_g_rank, burst_g_rank, atol=1e-2, rtol=0)
        bmt.print_rank(f"passed {i}")

if __name__ == "__main__":
    test_burst()
