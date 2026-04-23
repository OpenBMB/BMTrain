import datetime
import os
import random

import torch
import torch.distributed as dist

from .comm import Communicator
from .global_var import config
from .synchronize import synchronize
from .utils import print_dict


def init_distributed(
    init_method: str = "env://",
    seed: int = 0,
    pipe_size: int = -1,
    num_micro_batches: int = None,
    tp_size: int = 1,
):
    """Initialize distributed training.

    This is the ``torch.distributed`` based replacement of the original
    NCCL-direct implementation. It builds one ``ProcessGroup`` per parallel
    sub-axis and wraps them with :class:`bmtrain.comm.Communicator`.

    Args:
        seed (int): The random seed.
        pipe_size (int): pipe_size means that all processes will be divided
            into ``pipe_size`` groups along the pipeline axis.
        num_micro_batches (int): means that the input batchs will be divided
            into ``num_micro_batches`` small batches. Used in pipeline mode.
        tp_size (int): the size of each tensor parallel group.

    **init_distributed** reads the following environment variables:

    * ``WORLD_SIZE``
    * ``RANK``
    * ``MASTER_ADDR``
    * ``MASTER_PORT``
    * ``LOCAL_RANK``

    Normally, all the environments variables above are set by the pytorch
    distributed launcher.
    """
    torch.backends.cudnn.enabled = False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "10010"
    addr = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    master = addr + ":" + port
    timeout = datetime.timedelta(seconds=1800)

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )

    config["initialized"] = True
    config["pipe_size"] = pipe_size if pipe_size > 0 else 1
    config["pipe_enabled"] = pipe_size > 1
    config["local_rank"] = local_rank
    config["local_size"] = local_size
    config["rank"] = rank
    config["world_size"] = world_size
    config["calc_stream"] = torch.cuda.current_stream()
    config["load_stream"] = torch.cuda.Stream(priority=-1)
    config["tp_comm_stream"] = torch.cuda.Stream(priority=-1)
    config["pp_comm_stream"] = torch.cuda.Stream(priority=-1)
    config["barrier_stream"] = torch.cuda.Stream()
    config["load_event"] = torch.cuda.Event()
    config["tp_size"] = tp_size if tp_size > 0 else 1
    config["topology"] = topology(config)
    config["pipe_rank"] = config["topology"].get_group_rank("pipe")
    config["zero_rank"] = config["topology"].get_group_rank("zero")
    config["tp_rank"] = config["topology"].get_group_rank("tp")
    config["tp_zero_rank"] = config["topology"].get_group_rank("tp_zero")
    config["save_param_to_cpu"] = True
    config["save_param_gather"] = True
    config["load_param_gather"] = True
    cpus_this_worker = None

    all_available_cpus = sorted(list(os.sched_getaffinity(0)))

    cpus_per_worker = len(all_available_cpus) // local_size

    if cpus_per_worker < 1:
        cpus_this_worker = all_available_cpus
        torch.set_num_threads(1)
    else:
        cpus_this_worker = all_available_cpus[
            local_rank * cpus_per_worker : (local_rank + 1) * cpus_per_worker
        ]
        os.sched_setaffinity(0, cpus_this_worker)
        torch.set_num_threads(len(cpus_this_worker))

    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    topo = config["topology"]
    pp_size = config["pipe_size"]
    tp_size = config["tp_size"]
    stage_size = world_size // pp_size

    # ---- main world communicator ---------------------------------------
    world_ranks = list(range(world_size))
    config["comm"] = Communicator(dist.group.WORLD, rank, world_size, world_ranks)

    config["micros"] = num_micro_batches if num_micro_batches else config["pipe_size"]

    # ---- pipe_comm: same pipe_idx, vary stage_id -----------------------
    # Always create a pipe_comm (even when pipe_size == 1) to keep API parity
    # with the original NCCL-based bmt-nw, where pipe_comm is unconditionally
    # constructed.
    for p in range(stage_size):
        ranks = [p + s * stage_size for s in range(pp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            config["pipe_comm"] = Communicator(
                group, ranks.index(rank), len(ranks), ranks
            )

    # ---- pipe_tied_comm: 2-rank link between first and last stages -----
    # One per pipe column, members are (stage 0, stage pipe_size-1) of the
    # same pipe_idx.
    if config["pipe_enabled"]:
        for p in range(stage_size):
            ranks = [p, p + (pp_size - 1) * stage_size]
            group = dist.new_group(ranks)
            if rank in ranks:
                config["pipe_tied_comm"] = Communicator(
                    group, ranks.index(rank), len(ranks), ranks
                )

    # ---- pp_zero_comm: within one stage --------------------------------
    for s in range(pp_size):
        ranks = list(range(s * stage_size, (s + 1) * stage_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            config["pp_zero_comm"] = Communicator(
                group, ranks.index(rank), len(ranks), ranks
            )

    # ---- tp_comm: contiguous tp_size ranks -----------------------------
    for t in range(world_size // tp_size):
        ranks = [t * tp_size + i for i in range(tp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            config["tp_comm"] = Communicator(
                group, ranks.index(rank), len(ranks), ranks
            )

    # ---- tp_zero_comm: same tp_id across all tp groups -----------------
    for t in range(tp_size):
        ranks = [t + i * tp_size for i in range(world_size // tp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            config["tp_zero_comm"] = Communicator(
                group, ranks.index(rank), len(ranks), ranks
            )

    # ---- pp_tp_zero_comm: within stage, across tp groups, same tp_id ---
    dp_size = stage_size // tp_size
    for s in range(pp_size):
        for t in range(tp_size):
            ranks = [s * stage_size + t + k * tp_size for k in range(dp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                config["pp_tp_zero_comm"] = Communicator(
                    group, ranks.index(rank), len(ranks), ranks
                )

    config["zero_comm"] = config["comm"]

    for i in range(world_size):
        if i == rank:
            print_dict(
                "Initialization",
                {
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "local_size": local_size,
                    "master": master,
                    "device": torch.cuda.current_device(),
                    "cpus": cpus_this_worker,
                },
            )
        synchronize()


class topology:
    """A helper class to keep parallel information when using different
    parallel methods together.

    The semantics of the fields here exactly mirror the original bmt-nw
    topology so that downstream pipe / TP code keeps working unchanged.
    """

    def __init__(self, config):
        self.rank = config["rank"]
        pp_size = config["pipe_size"]
        tp_size = config["tp_size"]
        world_size = config["world_size"]
        assert world_size % (pp_size * tp_size) == 0, (
            "The nums of GPUs must be divisible by "
            "the pipeline parallel size * tensor parallel size"
        )

        dp_size = world_size // (pp_size * tp_size)
        config["tp_zero_size"] = dp_size
        config["zero_size"] = world_size // pp_size
        self.pipe_size = config["pipe_size"]
        self.dp_size = dp_size
        self.tp_size = tp_size
        stage_size = world_size // pp_size
        for _ in range(world_size):
            self.pipe_idx = self.rank % stage_size
            self.pipe_rank = self.rank // stage_size
            self.tp_id = self.rank % tp_size
            self.tp_idx = self.rank // tp_size
            # pp -> zero
            self.pp_zero_idx = self.pipe_rank
            self.pp_zero_id = self.pipe_idx
            # tp -> zero
            self.tp_zero_idx = self.tp_id
            self.tp_zero_id = self.tp_idx
            # pp -> tp -> zero
            self.pp_tp_zero_idx = self.pipe_rank * tp_size + self.tp_id
            self.pp_tp_zero_id = self.pipe_idx // tp_size
        # only zero
        self.zero_idx = 0
        self.zero_id = self.rank

    def get_group_id(self, group_name):
        if group_name == "pipe":
            return self.pipe_idx
        elif group_name == "zero":
            return self.zero_idx
        elif group_name == "tp_zero":
            return self.tp_zero_idx
        elif group_name == "tp":
            return self.tp_idx

    def get_group_rank(self, group_name):
        if group_name == "pipe":
            return self.pipe_rank
        elif group_name == "zero":
            return self.zero_id
        elif group_name == "tp_zero":
            return self.tp_zero_id
        elif group_name == "tp":
            return self.tp_id

    def is_first_rank(self, group_name="pipe"):
        if group_name == "pipe":
            return self.pipe_rank == 0
        elif group_name == "zero":
            return self.zero_id == 0
        elif group_name == "tp":
            return self.tp_id == 0

    def is_last_rank(self, group_name="pipe"):
        if group_name == "pipe":
            return self.pipe_rank == self.pipe_size - 1
        elif group_name == "zero":
            return self.zero_id == self.dp_size - 1
        elif group_name == "tp":
            return self.tp_id == self.tp_size - 1


def is_initialized() -> bool:
    return config["initialized"]
