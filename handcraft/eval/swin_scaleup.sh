
# Swin cube maximal scaling
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NID \
    --master_addr=worker-0 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_hybrid.py \
        --layer0 2 8 1 \
        --layer1 2 1 8 \
        --layer2 2 1 8 \
        --layer3 2 1 8 \
        --gbs 8 --mbs 8

# Swin Megatron maximal scaling
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NID \
    --master_addr=worker-0 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_hybrid.py \
        --layer0 2 8 1 \
        --layer1 2 1 8 \
        --layer2 2 1 8 \
        --layer3 2 1 8 \
        --gbs 8 --mbs 8
