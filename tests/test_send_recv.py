import bmtrain as bmt
import torch
from bmtrain.global_var import config
from bmtrain.pipe_comm import send_activations, recv_activations, gather_input
from bmtrain import nccl
from torch.distributed.distributed_c10d import recv
from time import sleep
def test_send_tensor():
    bmt.init_distributed()
    current_stream = torch.cuda.current_stream()
    groups = [0,2]
    rank = config['rank']
def test_gather_input():
    bmt.init_distributed(pipe_size=2)
    if config['topology'].get_group_id("pipe") == 0:
        a = torch.ones((2,1))
    else:
        a = torch.zeros((2,1))
    res = gather_input(a.cuda(),config['pipe_comm'])
    if config['topology'].get_group_rank("pipe") == 0:
        print(res)
        
def main():
    test_send_tensor()

if __name__ == '__main__':
    main()