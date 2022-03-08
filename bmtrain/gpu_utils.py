import torch

from .synchronize import synchronize
from .utils import print_rank

def see_memory(message, detail=False):
    synchronize()
    print_rank(message)
    if detail:
        print_rank("Model mem\n", torch.cuda.memory_summary())
    else:
        print_rank(f"""
        =======================================================================================
        memory_allocated {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB
        max_memory_allocated {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB
        =======================================================================================
        """)
    torch.cuda.reset_peak_memory_stats()