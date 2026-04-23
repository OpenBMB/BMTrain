"""
Communicator: wraps a torch.distributed ProcessGroup with storage-level
collective methods.

Notes
-----
* For ordinary tensor-level collectives, prefer torch.distributed directly.
  The methods here are thin adapters that let ZeRO / pipeline internals
  operate on raw ``torch.Storage`` objects through torch.distributed.

* ``groupcall`` provides a ``ncclGroupStart/End``-equivalent on top of
  ``torch.distributed``. When the running PyTorch supports
  ``_coalescing_manager`` it is used so that multiple P2P / collective
  calls are issued as a single fused operation. Otherwise it degrades to a
  no-op and the inner calls just execute sequentially (semantically still
  correct, only slower).
"""

import contextlib
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist

# ---- compatibility shims for older PyTorch -----------------------------------
_all_gather_into_tensor = (
    getattr(dist, "all_gather_into_tensor", None)
    or getattr(dist, "_all_gather_base", None)
)
_reduce_scatter_tensor = (
    getattr(dist, "reduce_scatter_tensor", None)
    or getattr(dist, "_reduce_scatter_base", None)
)

_REDUCE_OP = {
    "sum": dist.ReduceOp.SUM,
    "prod": dist.ReduceOp.PRODUCT,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
}
if hasattr(dist.ReduceOp, "AVG"):
    _REDUCE_OP["avg"] = dist.ReduceOp.AVG


def _to_op(name: str) -> "dist.ReduceOp":
    if name not in _REDUCE_OP:
        raise ValueError(f"Unknown reduce op: {name}")
    return _REDUCE_OP[name]


def _as_flat(obj) -> torch.Tensor:
    """Zero-copy view of a torch.Storage (or an already-tensor) as 1-D tensor."""
    if torch.is_tensor(obj):
        return obj.view(-1)
    return torch.tensor([], dtype=obj.dtype, device=obj.device).set_(obj)


# ---- groupcall ---------------------------------------------------------------
_coalescing_manager = getattr(dist, "_coalescing_manager", None)


@contextlib.contextmanager
def groupcall(group: Optional["dist.ProcessGroup"] = None):
    """``ncclGroupStart/End``-equivalent context.

    Tries ``torch.distributed._coalescing_manager`` (PyTorch >= 2.0) so that
    every collective / P2P launched inside the ``with`` block is fused into a
    single NCCL group. Falls back to a no-op on older PyTorch.
    """
    if _coalescing_manager is None:
        yield
        return
    try:
        with _coalescing_manager(group=group, async_ops=True):
            yield
    except TypeError:
        # Older signature: _coalescing_manager(group, device, reqs)
        with _coalescing_manager():
            yield
    except UnboundLocalError:
        # PyTorch >=2.x raises this when the with-block produced 0 ops
        # (the manager tries to use a `work` local that was never assigned).
        # Treat it as a successful no-op group.
        pass


# ---- the Communicator --------------------------------------------------------
class Communicator:
    """Wraps a ``torch.distributed.ProcessGroup`` with local/global rank
    mapping and storage-level collective helpers.

    Attributes
    ----------
    group : ProcessGroup
    rank : int           - rank inside this group
    world_size : int     - number of members
    """

    def __init__(self, group, rank: int, world_size: int, global_ranks) -> None:
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self._global_ranks = list(global_ranks)

    # ---- introspection ---------------------------------------------------
    def local_to_global(self, local_rank: int) -> int:
        return self._global_ranks[local_rank]

    def global_to_local(self, global_rank: int) -> int:
        return self._global_ranks.index(global_rank)

    # ---- storage-level collectives --------------------------------------
    def all_reduce(self, src, dst, op: str = "sum"):
        dst_t = _as_flat(dst)
        if not torch.is_tensor(src) and not torch.is_tensor(dst):
            if src.data_ptr() != dst.data_ptr():
                dst_t.copy_(_as_flat(src))
        else:
            src_t = _as_flat(src)
            if src_t.data_ptr() != dst_t.data_ptr():
                dst_t.copy_(src_t)
        dist.all_reduce(dst_t, op=_to_op(op), group=self.group)

    def all_gather(self, src, dst):
        _all_gather_into_tensor(_as_flat(dst), _as_flat(src), group=self.group)

    def reduce_scatter(self, src, dst, op: str = "sum"):
        _reduce_scatter_tensor(
            _as_flat(dst), _as_flat(src), op=_to_op(op), group=self.group,
        )

    def broadcast(self, src, dst, root: int):
        dst_t = _as_flat(dst)
        if not torch.is_tensor(src) and not torch.is_tensor(dst):
            same = src.data_ptr() == dst.data_ptr()
        else:
            same = _as_flat(src).data_ptr() == dst_t.data_ptr()
        if not same and self.rank == root:
            dst_t.copy_(_as_flat(src))
        dist.broadcast(dst_t, src=self.local_to_global(root), group=self.group)

    # ---- P2P ------------------------------------------------------------
    def send(self, src, peer: int):
        dist.send(_as_flat(src), dst=self.local_to_global(peer), group=self.group)

    def recv(self, dst, peer: int):
        dist.recv(_as_flat(dst), src=self.local_to_global(peer), group=self.group)

    def isend(self, src, peer: int):
        return dist.isend(
            _as_flat(src), dst=self.local_to_global(peer), group=self.group
        )

    def irecv(self, dst, peer: int):
        return dist.irecv(
            _as_flat(dst), src=self.local_to_global(peer), group=self.group
        )

    # ---- batched P2P ----------------------------------------------------
    def batch_isend_irecv(self, ops: Iterable):
        """Schedule a batch of P2P ops in one shot.

        ``ops`` is an iterable of ``(kind, tensor_or_storage, peer)`` tuples
        where ``kind`` is either ``"send"`` or ``"recv"``. Returns the list
        of ``Work`` handles produced by torch.distributed.
        """
        p2p_ops: List["dist.P2POp"] = []
        for kind, buf, peer in ops:
            tensor = _as_flat(buf)
            global_peer = self.local_to_global(peer)
            if kind == "send":
                p2p_ops.append(
                    dist.P2POp(dist.isend, tensor, global_peer, group=self.group)
                )
            elif kind == "recv":
                p2p_ops.append(
                    dist.P2POp(dist.irecv, tensor, global_peer, group=self.group)
                )
            else:
                raise ValueError(f"unknown p2p op kind: {kind}")
        if not p2p_ops:
            return []
        return dist.batch_isend_irecv(p2p_ops)

    # ---- misc -----------------------------------------------------------
    def barrier(self):
        dist.barrier(group=self.group)
