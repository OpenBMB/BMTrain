from schedule import pipeline_forward_backward
import torch
import bmtrain as bmt
def generate(iters):
    for i in range(iters):
        yield (torch.randint(0,1024,size=(12,1024),device="cuda", dtype=torch.int32),)

data_loader = iter(generate(100))
bmt.init_distributed(pipe_size=4)
# print(bmt.config['rank'])
models = [bmt.nn.PipeEmbedding(1024,128,dtype=torch.float16)]
for i in range(11):
    models.append(bmt.nn.Linear(128,128,dtype=torch.float16))
models = bmt.PipeDreamBlockList(models)

pipeline_forward_backward(models,  data_loader, 48)