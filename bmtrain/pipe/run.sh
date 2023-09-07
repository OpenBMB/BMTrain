if [ "$1" = "dp" ]; then
    nproc=1
else
    nproc=4
fi
torchrun --nnodes=1 --nproc_per_node=$nproc --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost example.py $1
