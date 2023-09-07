import torch
from bmtrain.distributed.ops import send_activations_list, recv_activations_list, send_activations, recv_activations, groupcall
from bmtrain.global_var import config
from collections.abc import Iterable
class PipeCommander:
    def __init__(self, topo, input_generator, num_micros, num_warmup, forward_only, interleaving_size) -> None:
        self.topo = topo
        self.input_generator = input_generator
        self.num_micros = num_micros
        self.num_warmup = num_warmup
        self.forward_only = forward_only
        self.interleaving_size = interleaving_size
    
    def get_data(self):
        assert config["topology"].pipe_rank == 0
        micro_batch = next(self.input_generator) 
        assert isinstance(micro_batch, Iterable)
        return list(micro_batch)

    def send_next(self, tensors):
        if not self.is_last_stage():
            if not isinstance(tensors, list):
                tensors = [tensors]
            send_activations_list(tensors, self.topo.pipe_rank + 1, config["pipe_comm"])
    
    def send_prev(self, tensors):
        if not self.is_first_stage():
            if not isinstance(tensors, list):
                tensors = [tensors]
            send_activations_list(tensors, self.topo.pipe_rank - 1, config["pipe_comm"])

    def recv_prev(self, need_data=False):
        if not self.is_first_stage():
            res = recv_activations_list(self.topo.pipe_rank - 1, config["pipe_comm"])
            for t in res:
                t.requires_grad_()
                
            return res
        else:
            if need_data:
                return self.get_data()
            else:
                return [None]
    
    def recv_next(self):
        if not self.is_last_stage():
            return recv_activations_list(self.topo.pipe_rank + 1, config["pipe_comm"])
        else:
            return [None]

    def allocate_tensor(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda")

    def is_first_stage(self):
        return self.topo.pipe_rank == 0

    def is_last_stage(self):
        return self.topo.pipe_rank == self.topo.pipe_size - 1
    
    def send_forward_recv_backward(self, forward_state):
        if not self.is_last_stage():
            if forward_state[0] is not None:
                self.send_next(forward_state)
            backward_grad = self.recv_next()
        else:
            backward_grad = [None]
        return backward_grad
    
    def send_backward_recv_forward(self, backward_grad, need_data=False):
        if not self.is_first_stage():
            if backward_grad[0] is not None:
                self.send_prev(backward_grad)
            forward_state = self.recv_prev()
        else:
            if need_data:
                forward_state = self.get_data()
            else:
                forward_state = [None]
        return forward_state


         