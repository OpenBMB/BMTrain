import torch
import torch.nn.functional as F
import bmtrain as bmt
from bmtrain.global_var import config 
from . import TransformerEncoder 


gb = 1024.0 * 1024.0 * 1024.0

class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward(x, weight, bias)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        if x.requires_grad:
            grad_x = grad_output.matmul(weight)
        if weight.requires_grad:
            dim = grad_output.dim()
            grad_weight = grad_output.reshape(-1,
                grad_output.shape[-1]).t().matmul(x.reshape(-1, x.shape[-1]))
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_x, grad_weight, grad_bias


class LinearFunctionForZeroStage3(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    #@autocast_custom_fwd
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):

        ctx.save_for_backward(input, weight, bias)

        if input.dim() == 2 and bias is not None:
            # fused op is marginally faster
            ret = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
            ret = output

        return ret

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    #@autocast_custom_bwd
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        #print(f"backward shaped grad_output {grad_output.shape}, input {input.shape}, weight {weight.shape} and bias {bias.shape if bias is not None else None}")
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            #print(f"Computing grad input weight {weight.shape} grad_output {grad_output.shape}")
            grad_input = grad_output.matmul(weight)
            #print(f"Computed grad input {grad_input.shape}")
        if ctx.needs_input_grad[1]:
            #print("Computing grad weight")
            dim = grad_output.dim()
            if dim > 2:
                grad_weight = grad_output.reshape(-1,
                                                  grad_output.shape[-1]).t().matmul(input.reshape(-1, input.shape[-1]))
            else:
                grad_weight = grad_output.t().matmul(input)
            #print(f"Computed grad weight grad_weight {grad_weight.shape}")
        if bias is not None and ctx.needs_input_grad[2]:
            #print("Computing grad bias")
            grad_bias = grad_output.sum(0)
            #print("Done computing grad bias")
            #print("needs bias")
        #print(f"backward shaped grad_input {grad_input.shape}, grad_weight {grad_weight.shape}, grad_bias {grad_bias.shape if grad_bias is not None else None}")
        return grad_input, grad_weight, grad_bias


class Linear(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = False, dtype = torch.float16) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=dtype, device="cuda"), init_method=torch.nn.init.xavier_normal_)
        if bias:
            self.bias = bmt.DistributedParameter(torch.empty((1, out_features), dtype=dtype, device="cuda"), init_method=torch.nn.init.zeros_)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        #return CustomLinear.apply(input, self.weight, self.bias)
        return LinearFunctionForZeroStage3.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Feedforward(bmt.DistributedModule):
    def __init__(self, dim_model : int, dim_ff : int, bias : bool = False, dtype = torch.float16) -> None:
        super().__init__()

        self.w_in = Linear(dim_model, dim_ff, bias = bias, dtype=dtype)
        self.w_out = Linear(dim_ff, dim_model, bias = bias, dtype=dtype)
        self.gate = Linear(dim_model, dim_ff, bias = bias, dtype=dtype)

        self.relu = torch.nn.ReLU()
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        gate_out = self.relu(self.gate(input))
        return self.w_out(self.w_in(input) * gate_out)

bmt.init_distributed(zero_level=2)

linears = []
for i in range(10):
    linears.append(bmt.CheckpointBlock(TransformerEncoder(8192, 20480), use_checkpoint=False))

linears = bmt.TransformerBlockList(linears)

device = torch.device('cuda')
bmt.synchronize()
if config['rank'] == 0:
	print('before forward', torch.cuda.memory_allocated(device) / gb)

x = torch.randn(4096, 8192, dtype=torch.float16, device=device).requires_grad_()
bmt.synchronize()
if config['rank'] == 0:
	print('init input', torch.cuda.memory_allocated(device) / gb)

y = linears(x)
bmt.synchronize()
if config['rank'] == 0:
	print('after forward', torch.cuda.memory_allocated(device) / gb)

y.sum().backward()
bmt.synchronize()
if config['rank'] == 0:
	print('after backward', torch.cuda.memory_allocated(device) / gb)
