import math
from .warmup import WarmupLRSchduler

class Noam(WarmupLRSchduler):

    def get_lr_warmpup(self, num_iter) -> float:
        return self.start_lr / math.sqrt(self.warmup_iter) * num_iter / self.warmup_iter
    
    def get_lr_decay(self, num_iter) -> float:
        return self.start_lr / math.sqrt(num_iter)
