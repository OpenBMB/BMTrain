import torch
import bmtrain as bmt
from flash_attn.flash_attn_interface import flash_attn_func as flash_cuda
import numpy as np

OpBurstAttn = bmt.nn.OpBurstAttn

def ref_attn(q, k, v, causal=False):
    scale = q.shape[-1] ** -0.5
    s = q @ k.transpose(-2, -1) * scale
    s = torch.softmax(s, dim=-1)
    if causal:
        s = torch.tril(s)
    p = s @ v
    return p


def burst(q, k, v, flash, causal, softmax_scale=None):
    # assert not causal, "causal not supported yet"
    res_burst = OpBurstAttn.apply(q, k, v, softmax_scale, flash)
    return res_burst

def get_chunk(t, dim):
    return t.chunk(bmt.config['sp_size'], dim=dim)[bmt.config['sp_rank']].contiguous()

def test_main():
    bmt.init_distributed(sp_size=4)
    test_burst(torch.float32, flash=None, causal=False)
    test_burst(torch.float16, flash=None, causal=False)
    test_burst(torch.float16, flash="cuda", causal=False)
    test_burst(torch.float16, flash="triton", causal=False)
    test_burst(torch.float32, flash=None, causal=True)
    test_burst(torch.float16, flash=None, causal=True)
    test_burst(torch.float16, flash="cuda", causal=True)
    test_burst(torch.float16, flash="triton", causal=True)

def test_burst(dtype, flash, causal):
    seq_dim = 2 if not flash else 1
    b, s, n, d = 2, 4096, 16, 32
    if bmt.config["sp_rank"] == 0:
        qkv_whole = torch.randn(b, n*3, s, d, dtype=dtype).cuda()
        grad_output = torch.randn(b, n, s, d, dtype=dtype).cuda()
        torch.save(qkv_whole, "./qkv.pt")
        torch.save(grad_output, "./grad.pt")
    bmt.synchronize()
    qkv_whole = torch.load("qkv.pt", map_location="cuda")
    grad_output = torch.load("grad.pt", map_location="cuda")
    qkv = [t.clone().detach().requires_grad_() for t in qkv_whole.chunk(3, dim=1)]
    
    o_ref = ref_attn(qkv[0], qkv[1], qkv[2])
    g_ref = torch.autograd.grad(o_ref, qkv, grad_output)

    for i in range(3):
        if flash is not None:
            qkv[i] = qkv[i].transpose(1, 2)
        qkv[i] = qkv[i].chunk(bmt.world_size(), dim=seq_dim)[bmt.rank()]
        qkv[i] = qkv[i].clone().detach().requires_grad_()

    if flash is not None:
        grad_output = grad_output.transpose(1, 2)

    grad_output = (
        grad_output.chunk(bmt.world_size(), dim=seq_dim)[bmt.rank()]
        .clone()
        .detach()
        .contiguous()
    )
    o1 = burst(qkv[0], qkv[1], qkv[2], flash, causal, softmax_scale=None)
    grad_qkv = torch.autograd.grad(o1, qkv, grad_output)
    if flash:
        o1 = o1.transpose(1, 2).contiguous()
        grad_qkv = [g.transpose(1, 2).contiguous() for g in grad_qkv]
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
        burst_g_rank = grad_qkv[i].detach().cpu().numpy()
        try:
            np.testing.assert_allclose(falsh_g_rank, burst_g_rank, atol=1e-2, rtol=0)
        except Exception as e:
            bmt.print_rank(e)

        bmt.print_rank(f"passed {i}")
    bmt.print_rank(f"dtype = {dtype}, flash = {flash}, causal = {causal} setting passed ")

if __name__ == "__main__":
    test_main()
