import torch
import torch.nn.functional as F
import bmtrain as bmt
from bmtrain.global_var import config 
from . import Attention 


gb = 1024.0 * 1024.0 * 1024.0

bmt.init_distributed(zero_level=2)

linears = []
for i in range(10), :
    linears.append(bmt.CheckpointBlock(Attention(
                    dim_model=8192, 
                    dim_head=128, 
                    num_head=64
                    dropout_p=0.0,
                    use_flash_attn=True,
                    dtype=torch.half
                    ), 
                use_checkpoint=False)
            )

linears = bmt.TransformerBlockList(linears)

device = torch.device('cuda')
bmt.synchronize()
if config['rank'] == 0:
	print('before forward', torch.cuda.memory_allocated(device) / gb)

x = torch.randn(4096, 8192, dtype=torch.float16, device=device).requires_grad_()
bmt.synchronize()
if config['rank'] == 0:
	print('init input', torch.cuda.memory_allocated(device) / gb)

y = linears(x)
bmt.synchronize()
if config['rank'] == 0:
	print('after forward', torch.cuda.memory_allocated(device) / gb)

y.sum().backward()
bmt.synchronize()
if config['rank'] == 0:
	print('after backward', torch.cuda.memory_allocated(device) / gb)
