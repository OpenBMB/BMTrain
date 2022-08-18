from typing import Optional, Union, List, Dict, Tuple
import torch

from . import _cuda as G
from ..utils import print_rank
from ..lr_scheduler.warmup import WarmupLRScheduler
from .. import nccl
from ..global_var import config

def check_overflow(param_groups):
    # check overflow
    has_inf_or_nan = torch.zeros(1, dtype=torch.uint8, device="cuda")[0]
    for group in param_groups:
        for p in group['params']:
            if p.grad is not None and p.dtype == torch.half: # TODO support other types
                G.f_has_inf_nan(p.grad, has_inf_or_nan)

    if "comm" in config:
        nccl.allReduce(has_inf_or_nan.storage(), has_inf_or_nan.storage(), "max", config["comm"])

    if has_inf_or_nan > 0:
        raise OverflowError("Gradient overflow")

def grad_rescale(param_groups, scale):
    for group in param_groups:
        for p in group['params']:
            if p.grad is not None and p.requires_grad:
                p.grad /= scale

class LossScaler:
    """Loss scaler for mix-precision training

    Args:
        loss_scale (float): The initial loss scale.
        loss_scale_factor (float): The loss scale factor.
        loss_scale_steps (int): The loss scale steps.
    """
    def __init__(self,
        loss_scale : float = 1,
        loss_scale_factor : float = 2,
        loss_scale_steps : int = 1024,
    ):
        self.loss_scale = loss_scale
        self.steps_since_last_scale = 0
        self.loss_scale_factor = loss_scale_factor if loss_scale_factor > 1 else 1 / loss_scale_factor
        self.loss_scale_steps = loss_scale_steps

        self.optimizers = []
        self.lr_schedulers = []

    def add_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[WarmupLRScheduler],
    ):
        """Add optimizer and (optional) its corresponding lr_scheduler into loss_scaler.
        All optimizers in the same loss_scaler share the same loss scale.

        Args:
            optim (torch.optim.Optimizer): A pytorch optimizer, e.g. torch.optim.Adam, torch.optim.SGD or bmtrain.optim.AdamOffloadOptimizer
            lr_scheduler (Optional[WarmupLRScheduler]): A warmup lr scheduler, e.g. bmt.lr_scheduler.Noam
        """
        self.optimizers.append(optimizer)
        if lr_scheduler is not None:
            self.lr_schedulers.append(lr_scheduler)

    def __call__(self, loss : torch.Tensor) -> torch.Tensor:
        """
        Backward with loss scale.

        Args:
            loss (torch.Tensor): loss
        """
        return loss * (self.loss_scale / config['world_size'])

    def zero_grad(self):
        """
        This is a helper function to call optimizer.zero_grad()
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        """
        Backward with loss scale.
        Synchronize streams before optimizer steps.

        This is a helper function to call optimizer.step() and lr_scheduler.step() and synchronize streams.

        This function can also handle gradient overflow by reducing the loss scale when it occurs.
        """
        current_stream =  torch.cuda.current_stream()
        # some reduce ops of distributed parameter were launched on load stream
        current_stream.wait_stream(config['load_stream'])

        if self.loss_scale > 1:
            has_overflow = False
            for optimizer in self.optimizers:
                try:
                    check_overflow(optimizer.param_groups)
                except OverflowError:
                    has_overflow = True
                    break
            if has_overflow:
                print_rank("Gradient overflow, change scale from %lf to %lf" % (self.loss_scale, self.loss_scale / self.loss_scale_factor))
                self._justify_scale(self.loss_scale / self.loss_scale_factor)
                self.zero_grad()
                return
                
        for optimizer, lr_scheduler in zip(self.optimizers, self.lr_schedulers):
            if hasattr(optimizer, "_bmtrain_optimizer") and optimizer._bmtrain_optimizer:
                optimizer.step(scale=self.loss_scale)
            else:
                grad_rescale(optimizer.param_groups, self.loss_scale)
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        self.steps_since_last_scale += 1

        if self.steps_since_last_scale >= self.loss_scale_steps:
            self._justify_scale(self.loss_scale * self.loss_scale_factor)

        config['load_stream'].wait_stream(current_stream)

    def clip_grad_norm(self, param_groups, max_norm, norm_type=2, eps=1e-6):
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
            >>> optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters())
            >>> loss_scaler = bmt.optim.LossScaler()
            >>> loss_scaler.add_optimizer(optimizer)
            >>> # ...
            >>> # backward_step()
            >>> loss_scaler.clip_grad_norm(optimizer.param_groups, max_norm=1.0, norm_type=2)

        """
        scale = self.loss_scale / config['world_size']
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

    @torch.no_grad()
    def _justify_scale(self, scale):
        for optimizer in self.optimizers:
            if hasattr(optimizer, "_on_justify_scale"):
                optimizer._on_justify_scale(self.loss_scale, scale)
        self.loss_scale = scale
        self.steps_since_last_scale = 0

