import torch
import torch.nn.functional as F
import bmtrain as bmt
from bmtrain.global_var import config 
from layers import TransformerEncoder


gb = 1024.0 * 1024.0 * 1024.0
device = torch.device('cuda')

def reserved(device):
    return torch.cuda.memory_reserved(device) / gb
def allocated(device):
    return torch.cuda.memory_allocated(device) / gb
def max_allocated(device):
    return torch.cuda.max_memory_allocated(device) / gb


bmt.init_distributed(zero_level=3)

linears = []
for i in range(10), :
    linears.append(bmt.CheckpointBlock(TransformerEncoder(
                    dim_model=8192, 
                    dim_head=128, 
                    num_heads=64,
                    dim_ff=20480,
                    bias=False,
                    dtype=torch.half
                    ), 
                use_checkpoint=False)
            )

linears = bmt.TransformerBlockList(linears)

bmt.synchronize()
if config['rank'] == 0:
    print('before forward',  allocated(device), reserved(device), max_allocated(device))
batch_size=1
seq_len=4096
x = torch.randn(batch_size, seq_len, 8192, dtype=torch.float16, device=device).requires_grad_()
bmt.synchronize()
if config['rank'] == 0:
    print('init input',  allocated(device), reserved(device), max_allocated(device))
enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
mask = torch.arange(seq_len).unsqueeze(0) <= torch.arange(seq_len).unsqueeze(1)
mask = mask.unsqueeze(0).unsqueeze(0).to(device)
y = linears(x,mask)
bmt.synchronize()
if config['rank'] == 0:
    print('after forward',  allocated(device), reserved(device), max_allocated(device))

y.sum().backward()
bmt.synchronize()
if config['rank'] == 0:
    print('after backward',  allocated(device), reserved(device), max_allocated(device))
