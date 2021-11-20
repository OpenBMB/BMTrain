import torch
from .global_var import config

def save(model : torch.nn.Module, file_name : str):
    torch.cuda.synchronize()
    state_dict = model.state_dict()
    if config["rank"] == 0:
        torch.save(state_dict, file_name)

def load(model : torch.nn.Module, file_name : str):
    model.load_state_dict( torch.load(file_name) )
    torch.cuda.synchronize()