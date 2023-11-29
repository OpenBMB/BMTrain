import subprocess
from tqdm import tqdm


tq = tqdm([
    ("different_output_shape", 1),
    ("load_ckpt", 1),
    # ("init_parameters", 1),
    ("init_parameters_multi_gpu", 4),
    ("optim_state", 4),

    ("has_inf_nan", 1),
    ("dropout", 1),
    ("loss_func", 1),

    ("model_wrapper", 4),

    ("nccl_backward", 4),

    ("training", 4),
])

for t, num_gpu in tq:
    PREFIX = f"export CUDA_VISIBLE_DEVICES=4,5,6,7;python3 -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node={num_gpu} --master_addr=localhost --master_port=32123"
    SUFFIX = f"> test_log.txt 2>&1"
    command = f"{PREFIX} test_{t}.py {SUFFIX}"
    completedProc = subprocess.run(command, shell=True)
    assert completedProc.returncode == 0, f"test {t} failed, see test_log.txt for more details."
    print(f"finish testing {t}")
