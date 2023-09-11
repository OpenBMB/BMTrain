from schedule import pipeline_forward_backward
import torch
import bmtrain as bmt
import time
import sys
def generate(iters):
    torch.manual_seed(42)
    for i in range(iters):
        inp = (torch.randint(0,1024,size=(12,1024),device="cuda", dtype=torch.int32),)
        yield inp
data_loader = iter(generate(100*16))
iters=10
dtype=torch.half

def test_pipe():
    bmt.init_distributed(seed=42, pipe_size=4)
    models = [bmt.nn.PipeEmbedding(1024,128,dtype=dtype)]
    for i in range(11):
        models.append(bmt.nn.Linear(128,128,dtype=dtype))
    bmt.init_parameters(models)
    models = bmt.PipeDreamBlockList(models)
    optimizer = bmt.optim.AdamOptimizer(models.parameters(), lr=0.001)
    start = time.time()
    for i in range(iters):
        pipeline_forward_backward(models,  data_loader, 12*16)
        if bmt.config['topology'].pipe_rank == 0:
            print(models['0'].weight.grad)
        optimizer.step()
    t = time.time() - start

def test_dp():
    bmt.init_distributed(seed=42, pipe_size=1)
    models = [bmt.nn.PipeEmbedding(1024,128,dtype=dtype)]
    for i in range(11):
        models.append(bmt.nn.Linear(128,128,dtype=dtype))
    bmt.init_parameters(models)
    models = bmt.TransformerBlockList(models)
    optimizer = bmt.optim.AdamOptimizer(models.parameters(), lr=0.001)
    for it in range(iters):
        for i in range(16):
            inp = next(data_loader)
            loss_tmp = models(*inp)
            loss_tmp = loss_tmp.mean()
            loss_tmp.backward()

        print(models['0'].weight.grad)
        optimizer.step()
    
if __name__ == "__main__":
    if sys.argv[1] == "dp":
        print("dp")
        test_dp()
    else:
        test_pipe()