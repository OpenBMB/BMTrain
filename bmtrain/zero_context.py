import torch
from . import nccl
from .global_var import config
from .synchronize import wait_loader


class ZeroContext:
    """ZeroContext is a helper class to Gather parameters before module forward and reduce scatter
    gradients after module backward.

    Args:
        block (BLock): Input Block.
        ctx_dict (dict): block._layer_dict.

    """

    def __init__(self, block: "Block", ctx_dict: dict = None) -> None:
        self.block = block
        self.ctx_dict = ctx_dict
        self._param_buffer = {}
        self._grad_buffer = {}
        self._need_release = False

    def enter(self, flag=0, requires_grad=False):
        """
        Gather parameters before module forward and init grad buffer before backward.
        flags = 0: normal mode
        flags = 1: gather param and not release , then save in ctx_dict
        flags = 2: not gather param and use the param in ctx_dict
        """
        if self.block._ready:
            return
        self.block._ready = True
        self._need_release = True

        wait_loader()
        with torch.cuda.stream(config["load_stream"]):
            for kw, val in self.block._storage_info.items():
                assert self.block._storage_params[kw].is_cuda
                if val["world_size"] == 1:
                    continue
                assert kw not in self._grad_buffer
                assert kw not in self._param_buffer
                local_param = self.block._storage_params[kw]

                if flag != 2:
                    self._param_buffer[kw] = torch.empty(
                        val["partition_size"] * val["world_size"],
                        dtype=local_param.dtype,
                        device=local_param.device,
                    )

                if requires_grad and local_param.requires_grad:
                    self._grad_buffer[kw] = torch.zeros(
                        val["partition_size"] * val["world_size"],
                        dtype=local_param.dtype,
                        device=local_param.device,
                    )
            if flag != 2:
                nccl.groupStart()
                for kw, val in self.block._storage_info.items():
                    if val["world_size"] == 1:
                        continue
                    nccl.allGather(
                        self.block._storage_params[kw],
                        self._param_buffer[kw],
                        val["zero_comm"],
                    )
                nccl.groupEnd()

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(config["load_stream"])

        for kw in self.block._storage_info.keys():
            if self.block._storage_info[kw]['world_size'] == 1:
                continue
            if flag != 2:
                self._param_buffer[kw].record_stream(current_stream)
            if requires_grad and kw in self._grad_buffer:
                self._grad_buffer[kw].record_stream(current_stream)

        for param in self.block._param_info:
            kw_name = param["kw_name"]
            offset = param["offset"]
            shape = param["shape"]
            numel = shape.numel()

            if self.block._storage_info[kw_name]["world_size"] == 1:
                continue

            if flag != 2:
                param["parameter"].data = self._param_buffer[kw_name][offset:offset+numel].view(shape)
            else:
                param["parameter"].data = self.ctx_dict[kw_name][offset:offset+numel].view(shape)

            if (
                requires_grad
                and kw_name in self._grad_buffer
                and param["parameter"].requires_grad
            ):
                param["parameter"].grad = self._grad_buffer[kw_name][offset:offset+numel].view(shape)

    def __enter__(self):
        self.enter()

    def exit(self, flag=0, backward=False):
        """
        Reduce scatter gradients when backward and release all parameters from buffer to block_storge when forward is done.
        """
        if not self._need_release:
            return
        self._need_release = False
        self.block._ready = False
        if backward:
            for kw, val in self.block._storage_info.items():

                if local_param.requires_grad:
                    if local_param.grad is None:
                        local_param.grad = torch.zeros(
                            val["partition_size"],
                            dtype=local_param.dtype,
                            device=local_param.device,
                        )
                    else:
                        self._grad_buffer[kw][
                            val["begin"] : val["end"]
                        ] += local_param.grad

            current_stream = torch.cuda.current_stream()
            config["load_stream"].wait_stream(current_stream)

            with torch.cuda.stream(config["load_stream"]):
                nccl.groupStart()
                for kw, val in self.block._storage_info.items():

                    if local_param.requires_grad:
                        nccl.reduceScatter(
                            self._grad_buffer[kw],
                            local_param.grad,
                            "sum",
                            val["zero_comm"],
                        )
                nccl.groupEnd()

            for kw in self._grad_buffer.keys():
                self._grad_buffer[kw].record_stream(config["load_stream"])

        for param in self.block._param_info:
            kw_name = param["kw_name"]
            if self.block._storage_info[kw_name]["world_size"] == 1:
                continue
            dtype = self.block._storage_params[kw_name].dtype
            device = self.block._storage_params[kw_name].device
            if "begin" not in param:
                param["parameter"].data = torch.tensor([], dtype=dtype, device=device)
                param["parameter"].grad = None
                continue
            begin = param["begin"]
            end = param["end"]
            size = end[0] if isinstance(end, tuple) else end
            param["parameter"].data = self.block._storage_params[kw_name].view(-1)[begin:begin+size]
            if (
                param["parameter"].requires_grad
                and self.block._storage_params[kw_name].grad is not None
            ):
                param["parameter"].grad = self.block._storage_params[kw_name].grad.view(-1)[begin:begin+size]
        if flag == 1:
            for i in self._param_buffer:
                self.ctx_dict[i] = self._param_buffer[i]
        self._grad_buffer = {}
        self._param_buffer = {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()
