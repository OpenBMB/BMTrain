import torch
from ..global_var import config
from . import _cuda as C
from .. import nccl
class AdamTransferManager:
    def __init__(self, avg_sq_host, param_host, device, stream):
        self._avg_sq_host = avg_sq_host
        self._param_host = param_host

        self.device = device
        self.stream = stream

        self.avg_sq = None
        self.param_fp32 = None
    
    def enter(self):
        curr_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.stream):
            curr_avg_sq = torch.empty( self._avg_sq_host.size(), dtype=torch.float32, device=self.device )
            curr_avg_sq.copy_( self._avg_sq_host, non_blocking=True )
            curr_param = torch.empty( self._param_host.size(), dtype=torch.float32, device=self.device )
            curr_param.copy_( self._param_host, non_blocking=True )
        
        # It's okay not to use record_stream() here

        self.avg_sq = curr_avg_sq
        self.param_fp32 = curr_param
        curr_stream.wait_stream(self.stream)
    
    def exit(self):
        curr_stream = torch.cuda.current_stream()
        self.stream.wait_stream(curr_stream)
        with torch.cuda.stream(self.stream):
            self._avg_sq_host.copy_(self.avg_sq, non_blocking=True)
            self._param_host.copy_(self.param_fp32, non_blocking=True)
        self.avg_sq = None
        self.param_fp32 = None

class AdamOptimizer(torch.optim.Optimizer):
    """
    Adam optimizer
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, scale=65536, hold_steps=0):
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

        self.load_stream = torch.cuda.Stream()
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
        delta = scale / self._scale
        self._scale = scale
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) > 0:
                    state['exp_avg'] /= delta
                    state['exp_avg_sq'] /= delta
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
                if p.grad is not None:
                    C.f_has_inf_nan(p.grad, has_inf_or_nan)
        
        if "comm" in config:
            nccl.allReduce(has_inf_or_nan.storage(), has_inf_or_nan.storage(), "max", config["comm"])

        if has_inf_or_nan > 0:
            raise OverflowError("Gradient overflow")
        
        self._steps_since_last_scale += 1

        # update parameters
        last_mgr = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros(p.size(), dtype=torch.half, device=p.device) # on device
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros(p.size(), dtype=torch.float32, pin_memory=True)   # on host

                        state['param_fp32'] = torch.empty(p.size(), dtype=torch.float32, pin_memory=True)   # on host
                        state['param_fp32'].copy_(p)

                    curr_mgr = AdamTransferManager( state['exp_avg_sq'], state['param_fp32'], p.device, self.load_stream )

                    curr_mgr.enter()
                    if last_mgr is not None:
                        last_mgr.exit()
                    last_mgr = curr_mgr

                    # update the steps for each param group update
                    state['step'] += 1
                    
                    C.f_adam(
                        curr_mgr.param_fp32,    # fp32
                        p,                      # fp16
                        p.grad,                 # fp16
                        state['exp_avg'],       # fp16: m
                        curr_mgr.avg_sq,        # fp32: v
                        group['betas'][0], group['betas'][1],
                        group['eps'],
                        0.0 if state["step"] <= self._hold_steps else group['lr'],
                        self._scale,
                        group['weight_decay'],
                        state['step']
                    )
        if last_mgr is not None:
            last_mgr.exit()
        
        return loss
    
    def loss_scale(self, loss : torch.Tensor) -> torch.Tensor:
        """
        Backward with loss scale.
        """
        return loss * (self.scale / config['world_size'])