from .. import nccl
from ..global_var import config
import torch

def clip_grad_norm(param_groups, max_norm, scale, norm_type=2, eps=1e-6):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a single Tensor that will have gradients normalized.
        max_norm (float or int): max norm of the gradients.
        norm_type (float or int): type of the used p-norm. Can be 'inf' for infinity norm.
        eps (float): epsilon used to avoid zero division.

    Returns:
        Total norm of the parameters (viewed as a single vector).

    Examples:
        >>> optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), scale=128)
        >>> # ...
        >>> # backward_step()
        >>> bmt.optim.clip_grad_norm(optimizer.param_groups, max_norm=1.0, scale=optimizer.scale, norm_type=2)

    """
    scale = scale / config['world_size']
    parameters = [p for group in param_groups for p in group['params'] if p.grad is not None]

    if norm_type == 'inf':
        total_norm_cuda = max(p.grad.data.abs().max() for p in parameters).detach()
        nccl.allReduce(total_norm_cuda.storage(), total_norm_cuda.storage(), "max", config["comm"])
        total_norm = total_norm_cuda
    else:
        norm_type = float(norm_type)
        total_norm_cuda = torch.cuda.FloatTensor([0])
        for index, p in enumerate(parameters):
            param_norm = p.grad.data.float().norm(norm_type)
            total_norm_cuda += param_norm ** norm_type
        nccl.allReduce(total_norm_cuda.storage(), total_norm_cuda.storage(), "sum", config["comm"])
        total_norm = total_norm_cuda[0] ** (1. / norm_type)
    # total_norm = total_norm / scale
    # clip_coef = float(max_norm) / (total_norm + eps)
    clip_coef = float(max_norm * scale) / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm / scale