from .. import nccl
from .shape import SHAPES
from ..global_var import config
from ..utils import round_up, print_rank
from .utils import format_size
import torch

def all2all():
    current_stream = torch.cuda.current_stream()
    for shape in SHAPES:
        global_size = round_up(shape, config['world_size'] * 2)

        result_tensor = torch.empty(global_size // 2, dtype=torch.half, device="cuda")
        global_tensor = torch.empty(global_size // 2, dtype=torch.half, device="cuda")

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        current_stream.record_event(start_evt)
        nccl.all2all(global_tensor.storage(), result_tensor.storage(), config['comm'])
        current_stream.record_event(end_evt)
        current_stream.synchronize()

        time_usage = start_evt.elapsed_time(end_evt)
        bw = global_size / 1024 / 1024 / 1024 * 1000 / time_usage
        print_rank("All to All:\tsize {}\ttime: {:4.3f}\tbw: {:2.6f} GB/s".format(format_size(global_size), time_usage, bw))

def all2one():
    current_stream = torch.cuda.current_stream()
    for shape in SHAPES:
        global_size = round_up(shape, config['world_size'] * 2)

        result_tensor = torch.empty(global_size // 2, dtype=torch.half, device="cuda")
        global_tensor = torch.empty(global_size // 2, dtype=torch.half, device="cuda")

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        current_stream.record_event(start_evt)
        nccl.groupStart()
        for r in range(config['world_size']):
            nccl.all2one(global_tensor.storage(), result_tensor.storage(), r, config['comm'])
        nccl.groupEnd()
        current_stream.record_event(end_evt)
        current_stream.synchronize()

        time_usage = start_evt.elapsed_time(end_evt)
        bw = global_size / 1024 / 1024 / 1024 * 1000 / time_usage
        print_rank("All to one:\tsize {}\ttime: {:4.3f}\tbw: {:2.6f} GB/s".format(format_size(global_size), time_usage, bw))
