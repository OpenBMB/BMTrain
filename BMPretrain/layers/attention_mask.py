import torch
import BMPretrain._c as C


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        last_dim = x.size(-1)
        out = C.softmax_forward(x.view(-1, last_dim)).view(x.size())
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.saved_tensors[0]
        return (grad_output - (grad_output * out).sum(dim=-1, keepdim=True)) * out

def attention_bias_mask_probs(attention_logits, bias, mask):
    masked_attn_logits = torch.where(
        mask,
        attention_logits + bias,
        torch.scalar_tensor(float('-inf'), device=attention_logits.device, dtype=attention_logits.dtype)
    )
    
    return Softmax.apply(masked_attn_logits)
