export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING
torchrun --nproc 4  test_pipe.py
# torchrun --nproc 4 test_send_recv.py
