import torch
import torch.nn.functional as F
import bmtrain as bmt
from bmtrain.global_var import config 
from layers import Attention 


gb = 1024.0 * 1024.0 * 1024.0

bmt.init_distributed(zero_level=3)

linears = []
for i in range(10), :
    linears.append(bmt.CheckpointBlock(Attention(
                    dim_model=8192, 
                    dim_head=128, 
                    num_heads=64,
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
batch_size=1
seq_len=4096
x = torch.randn(batch_size, seq_len, 8192, dtype=torch.float16, device=device).requires_grad_()
bmt.synchronize()
if config['rank'] == 0:
	print('init input', torch.cuda.memory_allocated(device) / gb)
enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
mask = torch.arange(seq_len).unsqueeze(0) <= torch.arange(seq_len).unsqueeze(1)
mask = mask.unsqueeze(0).unsqueeze(0)
print(mask.shape)
y = linears(x,x,mask)
bmt.synchronize()
if config['rank'] == 0:
	print('after forward', torch.cuda.memory_allocated(device) / gb)

y.sum().backward()
bmt.synchronize()
if config['rank'] == 0:
	print('after backward', torch.cuda.memory_allocated(device) / gb)
