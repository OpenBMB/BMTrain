from .warmup import WarmupLRScheduler


class Linear(WarmupLRScheduler):

    def get_lr_warmpup(self, num_iter) -> float:
        return self.start_lr * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter) -> float:
        return max(0.0, self.start_lr * (self.end_iter - num_iter) / (self.end_iter - self.warmup_iter))
