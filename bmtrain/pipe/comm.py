import torch
from bmtrain.distributed.ops import groupcall,all_reduce
from bmtrain.distributed.p2p_ops import send_tensors, recv_tensors
from bmtrain.global_var import config
from collections.abc import Iterable
from bmtrain.synchronize import synchronize
class PipeCommander:
    def __init__(self, topo, model, data_iter, num_micros, num_warmup, forward_only, interleaving_size) -> None:
        self.topo = topo
        self.comm = config['pipe_comm']
        self.input_generator = data_iter
        self.num_micros = num_micros
        self.num_warmup = num_warmup
        self.forward_only = forward_only
        self.interleaving_size = interleaving_size
        self.model = model

    def is_first_stage(self):
        if self.interleaving_size == 1:
            return self.topo.is_first_rank("pipe")
        else:
            raise ValueError("Now only supoort interleaving_size == 1")

    def is_last_stage(self):
        if self.interleaving_size == 1:
            return self.topo.is_last_rank("pipe")
        else:
            raise ValueError("Now only supoort interleaving_size == 1")


    def param_reduce(self, module):
        for name, param in module.named_parameters():
            p = all_reduce(param, "sum", config["pipe_tied_comm"])
            param.data = p

    def get_data(self):
        micro_batch = next(self.input_generator) 
        assert isinstance(micro_batch, Iterable)
        return micro_batch

    def send_next(self, tensors):
        if not self.is_last_stage():
            if not isinstance(tensors, Iterable):
                tensors = [tensors]
            elif not isinstance(tensors, list):
                tensors = list(tensors)
            send_tensors(tensors, self.topo.pipe_rank + 1, self.comm)

    def send_prev(self, tensors):
        if not self.is_first_stage():
            if not isinstance(tensors, Iterable):
                tensors = [tensors]
            elif not isinstance(tensors, list):
                tensors = list(tensors)
            send_tensors(tensors, self.topo.pipe_rank - 1, self.comm)

    def wait(self):
        torch.cuda.current_stream().wait_stream(config["pp_comm_stream"])
        
    def recv_prev(self, need_data=False):
        if not self.is_first_stage():
            res = recv_tensors(self.topo.pipe_rank - 1, self.comm)
            for idx,tensor in enumerate(res):
                if idx == 0:
                    tensor.requires_grad_()
            data = self.get_data()
            # return hidden state and data
            return res, data
        else:
            if need_data:
                # for first stage , only data
                return [None], self.get_data()
            else:
                # empty load for first stage
                return [None], [None]
    
    def recv_next(self):
        if not self.is_last_stage():
            res = recv_tensors(self.topo.pipe_rank + 1, self.comm)
            return res
        else:
            return [None]

    def allocate_tensor(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda")

    def is_even_rank(self):
        return self.topo.pipe_rank % 2 == 0

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
            forward_state, data = self.recv_prev()
            if backward_grad[0] is not None:
                self.send_prev(backward_grad)
        else:
            if need_data:
                forward_state = [None]
                data = self.get_data()
            else:
                forward_state = [None]
                data = [None]
        return forward_state, data


         
