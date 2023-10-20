import bmtrain as bmt
import torch
from models import GPT, GPTPipe
import re
from collections import OrderedDict

def partition(pipe_rank,pipe_size,len_modules):
    part_lens = [0]+[(len_modules // pipe_size + (i < (len_modules % pipe_size))) for i in range(pipe_rank+1)]
    start = sum(part_lens[:pipe_rank+1])
    end = start + part_lens[pipe_rank+1]
    return start,end

def key_process(key, pipe_size , rank, start, end):
    res = re.search("\.(\d+)\.", key)
    if res is not None:
        layer_idx = int(res.group(1))
    else:
        layer_idx = None
    if layer_idx is None or (layer_idx >= start and layer_idx < end): 
        if rank == 0:
            if key in ["word_emb.weight","pos_emb.weight"]:
                return key
            else:
                if layer_idx is not None:
                    return re.sub(r"\d+", str(layer_idx), key)
        elif rank == pipe_size - 1:
            if key in ["word_emb.weight"] or key.startswith("layernorm"):
                return key
            else:
                if layer_idx is not None:
                    return re.sub(r"\d+", str(layer_idx - start), key)
        else:
            if layer_idx is not None:
                return re.sub(r"\d+", str(layer_idx - start), key)
            else:
                return None



def init_model():
    model = GPT(
        num_layers=8,
        vocab_size=10240, 
        dim_model=2560,
        dim_head=80,
        num_heads=32,
        dim_ff=8192,
        max_distance=1024,
        bias=True,
        dtype=torch.half
    )
    return model

def get_len_modules(state):
    max_len = 0
    for key in state:
        s = re.search("\.(\d+)\.", key)
        if s is not None:
            res = int(s.group(1))
            if res>max_len:
                max_len = res
    return max_len+1


if __name__ == "__main__":
    bmt.init_distributed()
    model = init_model()
    bmt.load(model, "ckpt-0.pt")
    pipe_size = 4
    state = model.state_dict()

    for rank in range(pipe_size):
        print(rank)
        dic = OrderedDict()
        len_modules = get_len_modules(state)
        s,e = partition(rank, pipe_size, len_modules)
        print(s," ",e)
        for i in state.keys():
            k = key_process(i, pipe_size, rank, s, e)
            if k is not None:
                dic[k] =  state[i]
        print(dic.keys())
        torch.save(dic, f"pipe_{rank}.ckpt")
        
            