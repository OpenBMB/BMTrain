import torch
import bmtrain as bmt
from flash_attn.flash_attn_interface import flash_attn_func as flash_cuda
import numpy as np

OpBurstAttn = bmt.nn.OpBurstAttn

def flash(q, k, v):
    return flash_cuda(q, k, v, causal=False, softmax_scale=None, return_attn_probs=True)


def burst(q, k, v):
    res_burst = OpBurstAttn.apply(q, k, v, None, None, False, True)
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
    def get_chunk(t, dim):
        return t.chunk(bmt.config['sp_size'], dim=dim)[bmt.config['sp_rank']].contiguous()

    b, s, n, d = 2, 4096, 16, 32
    if bmt.config["sp_rank"] == 0:
        qkv = torch.randn(b, n*3, s, d, dtype=dtype).cuda()
        torch.save(qkv, "./qkv.pt")
    bmt.synchronize()
    qkv = torch.load("qkv.pt", map_location="cuda")
    qkv1 = [t.clone().detach().requires_grad_().transpose(1, 2).contiguous() for t in qkv.chunk(3, dim=1)]
    qkv_burst_normal = [get_chunk(t, dim=2).clone().detach().requires_grad_() for t in qkv.chunk(3, dim=1)]
    output, lse, _ = flash(qkv1[0], qkv1[1], qkv1[2])
    output_burst, lse_burst = burst(qkv_burst_normal[0], qkv_burst_normal[1], qkv_burst_normal[2])
    def test_allclose(t1, t2, atol, rtol):
        t1 = t1.detach().cpu().numpy()
        t2 = t2.detach().cpu().numpy()
        assert np.testing.assert_allclose(t1, t2, atol=atol, rtol=rtol)
    try:
        output = get_chunk(output.transpose(1, 2), 2).contiguous()
        lse = get_chunk(lse, 2).unsqueeze(dim=-1)
        print(torch.allclose(output, output_burst))
        print(torch.allclose(lse, lse_burst))
        raise Exception
    except Exception:
        if bmt.rank() == 0:
            from IPython import embed;embed()
        bmt.synchronize()
        
        
     
    
if __name__ == "__main__":
    test_burst()

