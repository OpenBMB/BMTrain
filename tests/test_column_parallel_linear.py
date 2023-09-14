import torch
import bmtrain as bmt
from bmtrain.global_var import config
import numpy as np

def run_bmt(x, gather_output, ckp_path, tp_size=2):
    linear = bmt.nn.ColumnParallelLinear(8,8, gather_output=gather_output)
    linear = bmt.Block(linear)
    bmt.init_parameters(linear)
    y = linear(x[config['topology'].tp_id])
    y.sum().backward()
    bmt.save(linear, ckp_path)
    bmt.synchronize()
    return y, linear._parameters['weight'].grad, linear._parameters['bias'].grad

def run_torch(x, ckp_path):
    linear = torch.nn.Linear(8, 8)
    linear_dict = torch.load(ckp_path)
    linear.load_state_dict(linear_dict)
    linear = linear.cuda()
    linear.weight.requires_grad_()
    y = linear(x)
    y.sum().backward()
    return y, linear.weight.grad, linear.bias.grad

def run(gather_output, ckp_path):
    tp_size = bmt.config['tp_size']
    torch.cuda.manual_seed(100)
    x = torch.randn(tp_size, 8,8, device='cuda').requires_grad_()
    y1, weight_grad1, bias_grad1 = run_bmt(x, gather_output, ckp_path)
    y2, weight_grad2, bias_grad2 = run_torch(x, ckp_path)
    tp_rank = config['topology'].tp_id
    if gather_output:
        assert np.allclose(y1.detach().cpu().numpy(), y2.flatten(0,1).detach().cpu().numpy())
    else:
        torch_out_list = torch.split(y2, y2.size()[-1] // tp_size, dim=y2.dim()-1)
        assert np.allclose(y1.detach().cpu().numpy(), torch_out_list[tp_rank].flatten(0,1).detach().cpu().numpy())

    weight_grad_list = weight_grad2.chunk(tp_size, dim=0)
    assert np.allclose(weight_grad1.reshape(weight_grad_list[tp_rank].shape).cpu().numpy(), weight_grad_list[tp_rank].cpu().numpy())

    bias_grad_list = bias_grad2.chunk(tp_size, dim=0)
    assert np.allclose(bias_grad1.reshape(bias_grad_list[tp_rank].shape).cpu().numpy(), bias_grad_list[tp_rank].cpu().numpy())

def test_gather_output():
    run(True, 'linear.ckp')

def test_no_gather_output():
    run(False, 'linear_no_gather.ckp')

if __name__ == "__main__":
    bmt.init_distributed(tp_size=2)
    test_gather_output()
    test_no_gather_output()

