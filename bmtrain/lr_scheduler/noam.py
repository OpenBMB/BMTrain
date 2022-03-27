import math
from .warmup import WarmupLRScheduler

class Noam(WarmupLRScheduler):

    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr / math.sqrt(self.warmup_iter) * num_iter / self.warmup_iter
    
    def get_lr_decay(self, num_iter) -> float:
        return self.start_lr / math.sqrt(num_iter)
