import torch
from bmtrain.distributed.ops import send_activations_list, recv_activations_list, send_activations, recv_activations, groupcall,all_reduce
from bmtrain.global_var import config
from collections.abc import Iterable
from bmtrain.synchronize import synchronize
class PipeCommander:
    def __init__(self, topo, model, data_iter, num_micros, num_warmup, forward_only, interleaving_size) -> None:
        self.topo = topo
        self.comm = self.topo.get_comm("pipe")
        self.input_generator = self.generator(data_iter)
        self.num_micros = num_micros
        self.num_warmup = num_warmup
        self.forward_only = forward_only
        self.interleaving_size = interleaving_size
        self.model = model
        self.send_handle = {"next":[], "prev":[]}
        self.recv_handle = {"next":[], "prev":[]}
    def generator(self, data_iterator):
        while True:
            try:
                inp = next(data_iterator)
                yield self.model.preprocess_func(inp)
            except StopIteration:
                break

    def param_reduce(self, module):
        for name, param in module.named_parameters():
            p = all_reduce(param, "sum", config["pipe_tied_comm"])
            param.data = p

    def get_data(self):
        micro_batch = next(self.input_generator) 
        assert isinstance(micro_batch, Iterable)
        return list(micro_batch)

    def send_next(self, tensors):
        handle = []
        if not self.is_last_stage():
            if not isinstance(tensors, Iterable):
                tensors = [tensors]
            elif not isinstance(tensors, list):
                tensors = list(tensors)
            handle.append(send_activations_list(tensors, self.topo.pipe_rank + 1, self.comm, async_op=True))
        self.send_handle["next"] = handle

    def send_prev(self, tensors):
        if not self.is_first_stage():
            if not isinstance(tensors, Iterable):
                tensors = [tensors]
            elif not isinstance(tensors, list):
                tensors = list(tensors)
            self.send_handle["prev"].append(send_activations_list(tensors, self.topo.pipe_rank - 1, self.comm, async_op=True))

    def recv_prev(self, need_data=False):
        if not self.is_first_stage():
            res, h = recv_activations_list(self.topo.pipe_rank - 1, self.comm)
            self.recv_handle["prev"].append(h)
            synchronize(config["pp_zero_comm"])
            for idx,tensor in enumerate(res):
                if idx == 0:
                    tensor.requires_grad_()
            return res
        else:
            if need_data:
                return self.get_data()
            else:
                return [None]
    
    def recv_next(self):
        if not self.is_last_stage():
            res, h = recv_activations_list(self.topo.pipe_rank + 1, self.comm)
            self.recv_handle["next"].append(h)
            return res
        else:
            return [None]

    def allocate_tensor(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda")

    def is_first_stage(self):
        return self.topo.pipe_rank == 0

    def is_last_stage(self):
        return self.topo.pipe_rank == self.topo.pipe_size - 1

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
            forward_state = self.recv_prev()
            if backward_grad[0] is not None:
                self.send_prev(backward_grad)
        else:
            if need_data:
                forward_state = self.get_data()
            else:
                forward_state = [None]
        return forward_state


         