import bmtrain as bmt
import torch
from bmtrain.global_var import config
from bmtrain.pipe_comm import send_activations, recv_activations, gather_input
def test_send_tensor():
    bmt.init_distributed()
    current_stream = torch.cuda.current_stream()
    if config['rank'] in [0,2]:
        a=torch.empty((2,4),dtype=torch.float16,device="cuda")
        a[0,0]=5
        send_activations(a,config['rank']+1)
    else:
        a = recv_activations(config['rank']-1)
        print(a.max())
def test_gather_input():
    bmt.init_distributed(pipe_size=4)
    res = gather_input(torch.zeros((3,2)),torch.zeros(()))
    if config['topology'].stage_id == 0:
        print(len(res))
        for i in res:
            print(i.shape)
def main():
    test_gather_input()

if __name__ == '__main__':
    main()