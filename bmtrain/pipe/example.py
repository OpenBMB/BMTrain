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

def test_pipe():
    bmt.init_distributed(seed=42, pipe_size=4)
    models = [bmt.nn.PipeEmbedding(1024,128,dtype=torch.float16)]
    for i in range(11):
        models.append(bmt.nn.Linear(128,128,dtype=torch.float16))
    # print(models[0].weight)
    bmt.init_parameters(models)
    models = bmt.PipeDreamBlockList(models)
    start = time.time()
    for i in range(1):
        pipeline_forward_backward(models,  data_loader, 12*16)
    if bmt.config['topology'].pipe_rank == 0:
        print(models['0'].weight.grad)
    t = time.time() - start
    print(t)

def test_dp():
    bmt.init_distributed(seed=42, pipe_size=1)
    models = [bmt.nn.PipeEmbedding(1024,128,dtype=torch.float16)]
    for i in range(11):
        models.append(bmt.nn.Linear(128,128,dtype=torch.float16))
    bmt.init_parameters(models)
    models = bmt.TransformerBlockList(models)
    for iter in range(1):
        loss = 0
        for i in range(16):
            loss_tmp = models(*next(data_loader))
            loss_tmp = loss_tmp.mean()
            print(loss_tmp.item())
            loss += loss_tmp
        print(loss)
        loss.backward()
    print(models['0'].weight.grad)
if __name__ == "__main__":
    if sys.argv[1] == "dp":
        print("dp")
        test_dp()
    else:
        test_pipe()