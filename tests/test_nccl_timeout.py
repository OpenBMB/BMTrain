from utils import *
import os
import bmtrain as bmt
from bmtrain import nccl
from bmtrain import C
import torch
import time

def test_main():
    unique_id : bytes = nccl.getUniqueId()
    exp = None
    comm = None
    try:
        comm = nccl.commInitRank(unique_id, 2, 0, 1000)
    except RuntimeError as e:
        exp = e
    assert isinstance(exp, RuntimeError)
    
    # can't test unless trapping signal.
    # """
    comm = nccl.commInitRank(nccl.getUniqueId(), 1, 0, 2000)
    start = time.time()
    C.cuda_spin(5000, 0); 
    torch.cuda.synchronize();
    print(time.time() - start)
    time.sleep(10)
    # """


if __name__ == "__main__":


    # os.environ["WORLD_SIZE"] = "2"
    # bmt.init_distributed()

    test_main()