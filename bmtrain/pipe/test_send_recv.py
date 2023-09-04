from schedule import pipeline_forward_backward
import torch
import bmtrain as bmt
from comm import PipeCommander,groupcall
def generate(iters):
    for i in range(iters):
        yield (torch.randint(0,1024,size=(12,1024),device="cuda", dtype=torch.int32),)

bmt.init_distributed(pipe_size=4)

topo = bmt.config["topology"]
num_micro_batches = 48
num_warmup = 3
interleaving_size = 1
data_iterator = iter(generate(100))
commander = PipeCommander(topo, num_micros=num_micro_batches,\
                            num_warmup=num_warmup, forward_only=False, \
                            interleaving_size=interleaving_size, \
                            data_iterator=data_iterator)
# with groupcall():
commander.send_prev([torch.randn((12,1024,128),device="cuda", dtype=torch.float16).requires_grad_()])
recv = commander.recv_next()
if recv[0] is not None:
    print(recv[0].shape)