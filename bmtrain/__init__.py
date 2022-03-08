from .global_var import config, world_size, rank
from .init import init_distributed

from .parameter import DistributedParameter, ParameterInitializer
from .layer import DistributedModule
from .param_init import init_parameters, grouped_parameters
from .utils import print_block, print_dict, print_rank
from .synchronize import synchronize, sum_loss, wait_loader
from .checkpointing import checkpoint
from .block_layer import CheckpointBlock, TransformerBlockList
from .backward import optim_step
from .gpu_utils import see_memory
from .clip_grad import clip_grad_norm

from . import debug
from .store import save, load

from . import benchmark
from . import optim
from . import inspect
from . import lr_scheduler
from . import loss

def print_inspect(model, name):
    print_rank(
        inspect.format_summary(
            inspect.inspect_model(model, name)
        )
    )