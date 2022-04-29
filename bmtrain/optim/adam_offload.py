import torch
from ..global_var import config
from . import _cpu as C
from . import _cuda as G
from .. import nccl
import torch.optim._functional as F

class AdamOffloadOptimizer(torch.optim.Optimizer):
    """
    Adam optimizer
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, scale=1, hold_steps=0):
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

        self._scale = scale
        self._steps_since_last_scale = 0
        self._hold_steps = hold_steps

    @property
    def scale(self):
        return self._scale

    @property
    def steps_since_last_scale(self):
        return self._steps_since_last_scale

    @torch.no_grad()
    def justify_scale(self, scale):
        self._scale = scale
        self._steps_since_last_scale = 0

    @torch.no_grad()
    def step(self, closure=None):
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

        # check overflow
        has_inf_or_nan = torch.zeros(1, dtype=torch.uint8, device="cuda")[0]
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.dtype == torch.half:
                    G.f_has_inf_nan(p.grad, has_inf_or_nan)

        if "comm" in config:
            nccl.allReduce(has_inf_or_nan.storage(), has_inf_or_nan.storage(), "max", config["comm"])

        if has_inf_or_nan > 0:
            raise OverflowError("Gradient overflow")

        # parameters to be updated
        update_params = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.requires_grad:
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    if p.dtype not in [torch.float16, torch.float32]:
                        raise RuntimeError('Adam only supports fp32 or fp16 gradients')

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros(p.size(), dtype=torch.float32, device="cpu")         # on host
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros(p.size(), dtype=torch.float32, device="cpu")      # on host

                        if p.dtype == torch.half:
                            state['_param_fp32'] = torch.empty(p.size(), dtype=torch.float32, device="cpu")     # on host
                            state['_param_fp32'].copy_(p)

                            # placeholder
                            state["_param_fp16"] = torch.empty(p.size(), dtype=torch.float16, pin_memory=True)  # on host
                            state["_grad_fp16"] = torch.empty(p.size(), dtype=torch.float16, pin_memory=True)   # on host
                        else:
                            state['_param_fp32'] = torch.empty(p.size(), dtype=torch.float32, pin_memory=True)     # on host
                            state['_param_fp32'].copy_(p)

                            # placeholder
                            state["_grad_fp32"] = torch.empty(p.size(), dtype=torch.float32, pin_memory=True)   # on host

                        state["_load_event"] = torch.cuda.Event()

                    update_params.append((p, state, group['betas'][0], group['betas'][1], group['eps'], group['lr'], group['weight_decay']))

        # transfer parameters to host asynchronously
        for param, state, _, _, _, _, _ in update_params:
            if param.dtype == torch.half:
                state["_grad_fp16"].copy_(param.grad, non_blocking=True)
            else:
                state["_grad_fp32"].copy_(param.grad, non_blocking=True)
            torch.cuda.current_stream().record_event(state["_load_event"])

        for param, state, beta1, beta2, eps, lr, weight_decay in update_params:
            # wait for transfer to host
            state["_load_event"].synchronize()

            state["step"] += 1

            # update parameters
            if param.dtype == torch.half:
                C.f_adam_cpu(
                    state["_param_fp32"].view(-1),
                    state["_param_fp16"].view(-1),
                    state["_grad_fp16"].view(-1),
                    state["exp_avg"].view(-1),
                    state["exp_avg_sq"].view(-1),
                    beta1, beta2,
                    eps,  0.0 if state["step"] <= self._hold_steps else lr,
                    self._scale,
                    weight_decay,
                    state["step"]
                )
                # transfer parameters back to device asynchronously
                param.copy_(state["_param_fp16"], non_blocking=True)
            else:
                state["_grad_fp32"].mul_(1.0 / self._scale)
                F.adam(
                    [state["_param_fp32"]],
                    [state["_grad_fp32"]],
                    [state["exp_avg"]],
                    [state["exp_avg_sq"]],
                    [],
                    [state["step"]],
                    amsgrad=False,
                    beta1=beta1,
                    beta2=beta2,
                    lr=0.0 if state["step"] <= self._hold_steps else lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    maximize=False
                )
                # transfer parameters back to device asynchronously
                param.copy_(state["_param_fp32"], non_blocking=True)

        self._steps_since_last_scale += 1

        return loss

    def loss_scale(self, loss : torch.Tensor) -> torch.Tensor:
        """
        Backward with loss scale.
        """
        return loss * (self.scale / config['world_size'])
