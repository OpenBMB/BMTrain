from typing import Optional
import torch
from .. import debug
from .. import nccl
from ..global_var import config
import math


class InspectTensor:
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

            self._summary.append(item)
            group_idx[group][name] += 1
    
    def get_summary(self):
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
                nw_summary.append({
                    "name": item["name"],
                    "group": item["group"],
                    "requires_grad" : False,
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
                    "mean": item["mean"],
                    "std": item["std"],
                    "shape": item["shape"],
                    "grad_mean" : None,
                    "grad_std" : None
                })
            else:
                ret.append({
                    "name": item["name"],
                    "mean": item["mean"],
                    "std": item["std"],
                    "shape": item["shape"],
                    "grad_mean" : item["grad_mean"],
                    "grad_std" : item["grad_std"]
                })
        return ret
    
    def get_tensor(self, name : str, group : Optional[str] = None, index : Optional[int] = None) -> torch.Tensor:
        group_name_prefix = f"{group}." if group is not None else ""

        all_names = []
        if index is None:
            all_names.append(f"{group_name_prefix}{name}")
            all_names.append(f"{group_name_prefix}0.{name}")
        else:
            all_names.append(f"{group_name_prefix}{index}.{name}")

        for item in self._summary:
            if item[name] in all_names:
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
    return InspectTensorManager()

def record_tensor(x : torch.Tensor, name : str, group = None, requires_grad = True):
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
        "requires_grad": True,
        "mean": None,
        "std": None,
        "shape": x_shape,
        "grad_mean" : None,
        "grad_std" : None,
        "tensor": x
    })
