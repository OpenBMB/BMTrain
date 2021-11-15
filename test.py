import torch
import torch.distributed as dist
import bmpretrain
import os

bmpretrain.init_distributed()

param = bmpretrain.DistributedParameter(torch.arange(13), requires_grad=False)

dist.barrier()
print(param)
print(param.gather())