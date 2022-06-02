from typing import Optional
import torch
from .. import debug
from .. import nccl
from ..global_var import config
import math


class InspectTensor:
    """This object is returned by `InspectTensorManager`.

    You can get the tensors recorded by `record_tensor`.

    """
    def __init__(self):
        self._summary = []
    
    def _set_summary(self, summary):
        self._summary = []
        group_cnt = {}
        for item in summary:
            group = item["group"]
            name = item["name"]
            if group not in group_cnt:
                group_cnt[group] = {}
            if name not in group_cnt[group]:
                group_cnt[group][name] = 0
            group_cnt[group][name] += 1
        
        group_idx = {}
        for item in summary:
            group = item["group"]
            name = item["name"]
            if group not in group_idx:
                group_idx[group] = {}
            if name not in group_idx[group]:
                group_idx[group][name] = 0

            group_name_prefix = f"{group}." if group is not None else ""
            if group_cnt[group][name] > 1:
                item["name"] = f"{group_name_prefix}{group_idx[group][name]}.{name}"
            else:
                item["name"] = f"{group_name_prefix}{name}"
            if not item["requires_grad"]:
                x = item["tensor"]
                info = torch.empty(2, dtype=x.dtype, device=x.device)
                info[0] = x.mean()
                info[1] = x.var()
                nccl.allReduce(
                    info.storage(),
                    info.storage(),
                    "avg",
                    config['comm']
                )
                x_mean = info[0].cpu().item()
                x_std = math.sqrt(info[1].cpu().item())
                item["mean"] = x_mean
                item["std"] = x_std

                info[0] = x.max()
                info[1] = -x.min()
                nccl.allReduce(
                    info.storage(),
                    info.storage(),
                    'max',
                    config['comm']
                )
                x_max = info[0].cpu().item()
                x_min = - info[1].cpu().item()
                item["max"] = x_max
                item["min"] = x_min

            self._summary.append(item)
            group_idx[group][name] += 1
    
    def get_summary(self):
        """Get the summary of the tensors recorded by `record_tensor`.

        Returns:
            A list of dicts. Each dict contains the following keys:
                - name: The name of the tensor.
                - min: The minimum value of the tensor.
                - max: The maximum value of the tensor.
                - mean: The mean value of the tensor.
                - std: The standard deviation of the tensor.
                - shape: The shape of the tensor.
                - grad_mean: The mean value of the gradient of the tensor.
                - grad_std: The standard deviation of the gradient of the tensor.

        **Note:** This method must be called outside of the `with` block.

        """
        nw_summary = []
        for item in self._summary:
            if item["requires_grad"] and item["tensor"].grad is not None:
                x = item["tensor"]
                info = torch.empty(4, dtype=x.dtype, device=x.device)
                info[0] = x.mean()
                info[1] = x.var()
                info[2] = x.grad.mean()
                info[3] = x.grad.var()
                nccl.allReduce(
                    info.storage(),
                    info.storage(),
                    "avg",
                    config['comm']
                )
                x_mean = info[0].cpu().item()
                x_std = math.sqrt(info[1].cpu().item())
                grad_mean = info[2].cpu().item()
                grad_std = math.sqrt(info[3].cpu().item())
                
                info[0] = x.max()
                info[1] = -x.min()
                nccl.allReduce(
                    info.storage(),
                    info.storage(),
                    'max',
                    config['comm']
                )
                x_max = info[0].cpu().item()
                x_min = - info[1].cpu().item()

                nw_summary.append({
                    "name": item["name"],
                    "group": item["group"],
                    "requires_grad" : False,
                    "min": x_min,
                    "max": x_max,
                    "mean": x_mean,
                    "std": x_std,
                    "shape": item["shape"],
                    "grad_mean" : grad_mean,
                    "grad_std" : grad_std,
                    "tensor": item["tensor"]
                })
            else:
                nw_summary.append(item)
        
        ret = []
        self._summary = nw_summary
        for item in self._summary:
            if item["requires_grad"]:
                ret.append({
                    "name": item["name"],
                    "min": item["min"],
                    "max": item["max"],
                    "mean": item["mean"],
                    "std": item["std"],
                    "shape": item["shape"],
                    "grad_mean" : None,
                    "grad_std" : None
                })
            else:
                ret.append({
                    "name": item["name"],
                    "min": item["min"],
                    "max": item["max"],
                    "mean": item["mean"],
                    "std": item["std"],
                    "shape": item["shape"],
                    "grad_mean" : item["grad_mean"],
                    "grad_std" : item["grad_std"]
                })
        return ret
    
    def get_tensor(self, name : str, group : Optional[str] = None, index : Optional[int] = None) -> torch.Tensor:
        """Get the tensor recorded by `record_tensor` by name, group and index.

        Args:
            name (str): The name of the tensor.
            group (Optional[str]): The group of the tensor.
            index (Optional[int]): The index of the tensor.
        
        Returns:
            The tensor if found, otherwise None.
        
        """
        group_name_prefix = f"{group}." if group is not None else ""

        all_names = []
        if index is None:
            all_names.append(f"{group_name_prefix}{name}")
            all_names.append(f"{group_name_prefix}0.{name}")
        else:
            all_names.append(f"{group_name_prefix}{index}.{name}")

        for item in self._summary:
            if item["name"] in all_names:
                return item["tensor"]
        return None


class InspectTensorManager:
    def __init__(self) -> None:
        self._inspector = None

    def __enter__(self) -> InspectTensor:
        self.prev_val = debug.get("_inspect_tensor", False)
        if not self.prev_val:
            debug.set("_inspect_tensor", True)
            self._inspector = InspectTensor()
            return self._inspector
        else:
            raise RuntimeError("InspectTensorManager is already in use")
    
    def __exit__(self, *args):
        if not self.prev_val:
            debug.set("_inspect_tensor", self.prev_val)
            summary = debug.get("_inspect_hidden_states", [])
            self._inspector._set_summary(summary)
            self._inspector = None
            debug.set("_inspect_hidden_states", [])
    

def inspect_tensor() -> InspectTensorManager:
    """**inspect_tensor** returns a context manager that can be used to get the intermediate results of the model computations and their gradients.

    Example:
        >>> with bmt.inspect.inspect_tensor() as inspector:
        >>>     loss = model(inputs)
        >>>     loss.backward()
        >>> summary = inspector.get_summary()
        >>> text_summary = bmt.inspect.format_summary(summary)
        >>> bmt.print_rank(text_summary)
        name   shape     max     min     std     mean    grad_std  grad_mean
        ...

    **Note:** loss.backward() must be called inside the context manager, otherwise the gradients will not be recorded.
    **Note:** Calling get_summary() has significant overhead.

    """

    return InspectTensorManager()

def record_tensor(x : torch.Tensor, name : str, group = None, requires_grad = True):
    """Record the tensor for inspection.

    Args:
        x (torch.Tensor): The tensor to be recorded.
        name (str): The name of the tensor.
        group (str): The group name of the tensor.
        requires_grad (bool): Whether records the gradients of the tensor.
    
    **Note:** This function is only available in inspect_tensor context.
    **Note:** Recording too many tensors may cause memory issues.
    
    """
    if isinstance(x, torch.nn.Parameter):
        raise RuntimeError("Cannot inspect Parameter")
    
    if not debug.get("_inspect_tensor", False):
        # do nothing
        return
    
    x_shape = tuple((x.size(0) * config["world_size"],) + x.size()[1:])

    if requires_grad and x.requires_grad:
        x.retain_grad()
    debug.append("_inspect_hidden_states", {
        "name": name,
        "group": group,
        "requires_grad": requires_grad and x.requires_grad,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "shape": x_shape,
        "grad_mean" : None,
        "grad_std" : None,
        "tensor": x
    })
