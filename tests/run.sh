export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost $1
