from typing import Optional
import os
import torch
import bmtrain as bmt
from bmtrain.global_var import config
from layers.attention import all2all_tensor
import torch
import bmtrain as bmt

def print_rank(msg):
    if bmt.rank() == 0:
        print(msg)

def check_helper(v1, v2, debug=False):
    if debug:
        print_rank(torch.max(torch.abs(v1 - v2)))
        print_rank(torch.mean(torch.abs(v1 - v2)))
    torch.testing.assert_close(v1, v2, rtol=1e-3, atol=1e-2)


def check_helper_list(l1, l2, end=False):
    if bmt.rank() == 0:
        for i in range(len(l1)):
            check_helper(l1[i], l2[i])
    if end:
        exit()


def check_is_nan(tensor):
    if torch.isnan(tensor).any():
        print("nan detected")
        exit()



def test(q, k, v, func, grad_output):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    grad_output = grad_output.contiguous()
    o = func(q, k, v)
    gq, gk, gv = torch.autograd.grad(o, (q, k, v), grad_output)
    return o, (gq, gk, gv)

def test_msg(test_func, msg, *args, **kwargs):
    try:
        test_func(*args, **kwargs)
        bmt.print_rank(msg, " Success")
    except:
        bmt.print_rank(msg, " Failed")
        exit()

def get_chunk(t, dim):
    return t.chunk(bmt.config["sp_size"], dim=dim)[bmt.config['sp_rank']].contiguous()

def ref_attn(q, k, v):
    scale = q.shape[-1] ** -0.5
    s = q @ k.transpose(-2, -1) * scale
    s = torch.softmax(s, dim=-1)
    p = s @ v
    return p

def all2all_attn(q, k, v):
    q = all2all_tensor(q, 2, 1)
    k = all2all_tensor(k, 2, 1)
    v = all2all_tensor(v, 2, 1)
    o = ref_attn(q, k, v)
    o = all2all_tensor(o, 1, 2)
    return o

def test_all2all():
    bmt.init_distributed(sp_size=2)
    b, n, s, d = 2, 16, 1024, 32
    if bmt.rank() == 0:
        qkv = torch.randn(b, n*3, s, d, dtype=torch.float16).cuda()
        grad_output = torch.randn(b, n, s, d, dtype=torch.float16).cuda()
        torch.save(qkv, "qkv.pt")
        torch.save(grad_output, "grad.pt")
    bmt.synchronize()
    qkv = torch.load("qkv.pt", map_location="cuda")
    grad_output = torch.load("grad.pt", map_location="cuda")
    qkv1 = [t.clone().detach().requires_grad_() for t in qkv.chunk(3, dim=1)]
    if bmt.rank() == 0:
        os.remove("qkv.pt")
        os.remove("grad.pt")

    o_ref, g_ref = test(qkv1[0], qkv1[1], qkv1[2], ref_attn, grad_output)
    for i in range(3):
        qkv1[i] = qkv1[i].chunk(bmt.world_size(), dim=2)[bmt.rank()]
    grad_output = (
        grad_output
        .chunk(bmt.world_size(), dim=2)[bmt.rank()]
        .clone()
        .detach()
        .contiguous()
    )
    o1, grad_qkv1 = test(qkv1[0], qkv1[1], qkv1[2], all2all_attn, grad_output)
    o1 = o1.contiguous()
    grad_qkv1 = [g.contiguous() for g in grad_qkv1]
    o_ref = get_chunk(o_ref, dim=2)
    g_ref = [get_chunk(g, dim=2) for g in g_ref]
    test_msg(check_helper, "Output Correctness Check", o_ref, o1)
    test_msg(check_helper, "Value Correctness Check", g_ref[2], grad_qkv1[2])
    test_msg(check_helper, "Key Correctness Check", g_ref[1], grad_qkv1[1])
    test_msg(check_helper, "Query Correctness Check", g_ref[0], grad_qkv1[0])

if __name__ == "__main__":

    test_all2all()

