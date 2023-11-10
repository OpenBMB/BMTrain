import bmtrain as bmt
import torch
import re
from collections import OrderedDict

def partition(pipe_rank, pipe_size, len_modules):
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
        if layer_idx is not None:
            return re.sub("\.(\d+)\.", "."+str(layer_idx - start)+".", key)
        else:
            return key

def get_len_modules(state):
    max_len = 0
    for key in state:
        s = re.search("\.(\d+)\.", key)
        if s is not None:
            res = int(s.group(1))
            if res>max_len:
                max_len = res
    return max_len+1

def get_state_dict_pipe(path):
    pipe_size = bmt.config["pipe_size"]
    pipe_rank = bmt.config["pipe_rank"]

    if bmt.rank() == 0:
        ds_state_dict = bmt.store.DistributedStateDictWrapper(torch.load(path))
    else:
        ds_state_dict = bmt.store.DistributedStateDictWrapper({})

    len_modules = get_len_modules(ds_state_dict)
    s,e = partition(pipe_rank, pipe_size, len_modules)
    state_dict = OrderedDict()

    for key in ds_state_dict:
        param = ds_state_dict[key].broadcast()
        k_p = key_process(key, pipe_size, pipe_rank, s, e)
        if k_p is not None:
            state_dict[k_p] = param
        else:
            del param
    return state_dict
            
def load_model_pipe(model, path, load_whole=True):
    """
    load_whole: Boolean, if True, load from the whole model file, else load model from the pipeline/tensor parallel model file
    """
    if load_whole:
        state_dict = get_state_dict_pipe(path)
        model.load_state_dict(state_dict, strict=False)
    else:
        pipe_rank = bmt.config["pipe_rank"]
        tp_rank = bmt.config["tp_rank"]
        ckpt_path = f"{path}_pp_{pipe_rank}_tp_{tp_rank}.pt"
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict)

def save_model_pipe(model, path):
    pipe_rank = bmt.config["pipe_rank"]
    tp_rank = bmt.config["tp_rank"]
    state_dict = model.state_dict()
    torch.save(state_dict, f"{path}_pp_{pipe_rank}_tp_{tp_rank}.pt")
