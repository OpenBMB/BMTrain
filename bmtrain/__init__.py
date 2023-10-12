try:
    from . import nccl
except:
    load_nccl_pypi()
from .global_var import config, world_size, rank
from .init import init_distributed

from .parameter import DistributedParameter, ParameterInitializer
from .layer import DistributedModule
from .param_init import init_parameters, grouped_parameters
from .synchronize import synchronize

from . import distributed
