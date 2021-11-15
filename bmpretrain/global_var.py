import torch
from typing_extensions import TypedDict
class ConfigMap(TypedDict):
    rank : int
    local_rank : int
    world_size : int
    local_size : int

    calc_stream : torch.cuda.Stream
    load_stream : torch.cuda.Stream
    load_event : torch.cuda.Event

    comm : 'NCCLCommunicator'

config = ConfigMap()