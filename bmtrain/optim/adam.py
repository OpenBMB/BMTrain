import torch
from ..global_var import config
from . import _function as F
# torch.optim._functional was removed in some PyTorch versions; fall back to manual Adam if unavailable.
try:
    import torch.optim._functional as _optim_functional
    _has_torch_adam = True
except ImportError:
    _has_torch_adam = False
from .. import C
from .. import nccl
import inspect
from copy import deepcopy
from itertools import chain
from collections import defaultdict


def _functional_adam_state_step(step) -> torch.Tensor:
    """Scalar step tensor for torch.optim._functional.adam.

    PyTorch 2.x _multi_tensor_adam groups state tensors with params; step is only
    exempted when it is CPU float32/float64. torch.tensor(int) yields int64 and
    triggers RuntimeError in _group_tensors_by_device_and_dtype.
    """
    if isinstance(step, torch.Tensor):
        return step.detach().to(dtype=torch.float64, device="cpu").reshape(())
    return torch.tensor(float(step), dtype=torch.float64, device="cpu")


def _torch_functional_adam_other_kwargs():
    """Build extra kwargs needed by torch.optim._functional.adam across versions."""
    other_kwargs = {}
    if not _has_torch_adam:
        return other_kwargs
    sig = inspect.signature(_optim_functional.adam).parameters
    if "maximize" in sig:
        other_kwargs["maximize"] = False
    # PyTorch 2.4+ defaults foreach=True on CUDA; int64 step used to break grouping
    if "foreach" in sig:
        other_kwargs["foreach"] = False
    if "fused" in sig:
        other_kwargs["fused"] = False
    return other_kwargs


class AdamOptimizer(torch.optim.Optimizer):
    """
    Adam optimizer support fp16 and bf16.
    """

    _bmtrain_optimizer = True

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        hold_steps=0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self._hold_steps = hold_steps

    def _on_justify_scale(self, old_scale, new_scale):
        delta = new_scale / old_scale
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state:
                    state = self.state[p]
                    if len(state) > 0:
                        if p.dtype == torch.float16:
                            state["exp_avg"] *= delta
                            state["exp_avg_sq"] *= delta

    @torch.no_grad()
    def step(self, closure=None, scale=1):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # update parameters
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and p.requires_grad:
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    if p.dtype not in [torch.float32, torch.half, torch.bfloat16]:
                        raise RuntimeError(
                            "Adam only supports fp32, fp16 and bf16 gradients"
                        )

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        if p.dtype == torch.float16:
                            state["exp_avg"] = torch.zeros(
                                p.size(), dtype=torch.float16, device=p.device
                            )  # on device
                        else:
                            state["exp_avg"] = torch.zeros(
                                p.size(), dtype=torch.float32, device=p.device
                            )  # on device
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros(
                            p.size(), dtype=torch.float32, device=p.device
                        )  # on device

                        if p.dtype != torch.float32:
                            state["_param_fp32"] = torch.empty(
                                p.size(), dtype=torch.float32, device=p.device
                            )  # on device
                            state["_param_fp32"].copy_(p)

                    # update the steps for each param group update
                    if ("maximize" in group) and (group["maximize"] is True):
                        grad = -p.grad
                    else:
                        grad = p.grad

                    if p.dtype == torch.float32:
                        if _has_torch_adam:
                            other_kwargs = _torch_functional_adam_other_kwargs()
                            _optim_functional.adam(
                                [p],
                                [grad / scale],
                                [state["exp_avg"]],
                                [state["exp_avg_sq"]],
                                [],
                                [_functional_adam_state_step(state["step"])],
                                amsgrad=False,
                                beta1=group["betas"][0],
                                beta2=group["betas"][1],
                                lr=0.0 if state["step"] < self._hold_steps else group["lr"],
                                weight_decay=group["weight_decay"],
                                eps=group["eps"],
                                **other_kwargs
                            )
                        else:
                            # Fallback: manual Adam when torch.optim._functional is unavailable.
                            lr = 0.0 if state["step"] < self._hold_steps else group["lr"]
                            beta1, beta2 = group["betas"]
                            eps = group["eps"]
                            weight_decay = group["weight_decay"]
                            g = grad / scale
                            if weight_decay != 0:
                                g = g.add(p, alpha=weight_decay)
                            state["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
                            state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                            bias_correction1 = 1 - beta1 ** (state["step"] + 1)
                            bias_correction2 = 1 - beta2 ** (state["step"] + 1)
                            step_size = lr / bias_correction1
                            denom = (state["exp_avg_sq"].sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                            p.addcdiv_(state["exp_avg"], denom, value=-step_size)
                        state["step"] += 1
                    else:
                        f = F.adam_fp16 if p.dtype == torch.float16 else F.adam_bf16
                        state["step"] += 1
                        f(
                            state["_param_fp32"],  # fp32
                            p,  # fp16
                            grad,  # fp16
                            state["exp_avg"],  # fp16: m
                            state["exp_avg_sq"],  # fp32: v
                            group["betas"][0],
                            group["betas"][1],
                            group["eps"],
                            0.0 if state["step"] < self._hold_steps else group["lr"],
                            scale,
                            group["weight_decay"],
                            state["step"],
                        )

        return loss

    def get_avg_delta():

        raise NotImplementedError(
            "get delta info is not supported in Adam optimizer , try bmt.optim.AdamOffloadOptimizer"
        )

    def get_var_delta():

        raise NotImplementedError(
            "get delta info is not supported in Adam optimizer , try bmt.optim.AdamOffloadOptimizer"
        )

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of " "parameter groups"
            )
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable((g["params"] for g in saved_groups)),
                chain.from_iterable((g["params"] for g in groups)),
            )
        }

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]

                if param.dtype != torch.float32 and "_param_fp32" not in v:
                    v["_param_fp32"] = torch.empty(
                        param.size(), dtype=torch.float32, device=param.device
                    )
                    v["_param_fp32"].copy_(param)

                for name, dtype in [
                    (
                        "exp_avg",
                        (
                            torch.float16
                            if param.dtype == torch.float16
                            else torch.float32
                        ),
                    ),
                    ("exp_avg_sq", torch.float32),
                    ("_param_fp32", torch.float32),
                ]:
                    if name in v:
                        v[name] = v[name].to(param.device).to(dtype)

                state[param] = v
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

    # TODO zero_grad(set_to_none=True) makes optimizer crashed, maybe the reason of grad accu
    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none=set_to_none)
