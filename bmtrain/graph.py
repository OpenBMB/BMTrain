from numpy.core.fromnumeric import size
from sklearn.metrics import zero_one_loss
from .global_var import config
class nodeGraph:
    def __init__(self,world_size,rank,zero2_size,pp_size,zero3_size) -> None:
        self.world_size = world_size
        self.rank = rank
        self.grid = [(i,j)for i in range(zero3_size) for j in range(zero2_size)]