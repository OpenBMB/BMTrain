import torch
import torch.nn.functional as F
import bmtrain as bmt
from bmtrain.global_var import config 
from layers import TransformerEncoder


gb = 1024.0 * 1024.0 * 1024.0
def reserved(device):
    return torch.cuda.memory_reserved(device) / gb
def allocated(device):
    return torch.cuda.memory_allocated(device) / gb
def max_allocated(device):
    return torch.cuda.max_memory_allocated(device) / gb

bmt.init_distributed(zero_level=3)

linears = []
for i in range(10), :
    linears.append(TransformerEncoder(
                    dim_model=8192, 
                    dim_head=128, 
                    num_heads=64,
                    dim_ff=20480,
                    bias=False,
                    dtype=torch.half
                    ) 
            )

#linears = bmt.TransformerBlockList(linears)
linears = torch.nn.ModuleList(linears)

optimizer = bmt.optim.AdamOffloadOptimizer(linears.parameters(), weight_decay=1e-2)
lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)

optim_manager = bmt.optim.OptimManager(loss_scale=2**20)
optim_manager.add_optimizer(optimizer, lr_scheduler)

bmt.synchronize()

device = torch.device('cuda')
bmt.synchronize()
if config['rank'] == 0:
	print('before init input',  allocated(device), reserved(device))
batch_size=1
seq_len=4096

for i in range(4):
    x = torch.randn(batch_size, seq_len, 8192, dtype=torch.float16, device=device).requires_grad_()
    bmt.synchronize()
    if config['rank'] == 0:
        print('init input',  allocated(device), reserved(device))
    enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
    mask = torch.arange(seq_len).unsqueeze(0) <= torch.arange(seq_len).unsqueeze(1)
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)
#y = linears(x,mask)
    y = x
    for encoder in linears:
        y = encoder(y, mask)
    bmt.synchronize()
    if config['rank'] == 0:
        print('after forward',  allocated(device), reserved(device), max_allocated(device))

    y.sum().backward()
    bmt.synchronize()
    if config['rank'] == 0:
        print('after backward',  allocated(device), reserved(device), max_allocated(device))
    optim_manager.step()
    if config['rank'] == 0:
        print('after optimizer',  allocated(device), reserved(device))
#torch.cuda.empty_cache()
    optim_manager.zero_grad()
