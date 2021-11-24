
echo benchmarking gpt megatron hybrid parallelism...

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    benchmark/megatron/gpt.py > mydata/MagicCube/expdata/8B.2V100.Megatron.txt

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    benchmark/megatron/gpt.py > mydata/MagicCube/expdata/8B.4V100.Megatron.txt

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    benchmark/megatron/gpt.py > mydata/MagicCube/expdata/8B.8V100.Megatron.txt
