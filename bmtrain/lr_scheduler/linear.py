from .warmup import WarmupLRScheduler


class Linear(WarmupLRScheduler):
    """
        After a warmup period during which learning rate increases linearly between 0 and the start_lr,
        The decay period performs :math:`\text{lr}=\text{start\_lr}\times \dfrac{\text{end\_iter}-\text{num\_iter}}{\text{end\_iter}-\text{warmup\_iter}}`
    """

    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter) -> float:
        return max(0.0, self.start_lr * (self.end_iter - num_iter) / (self.end_iter - self.warmup_iter))
