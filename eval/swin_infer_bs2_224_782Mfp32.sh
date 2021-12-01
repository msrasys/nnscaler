
logfile=expinfer_224_782M_fp32_bs2

mkdir -p ${logfile}

# ================== Maximal Tensor Parallel ===============
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt_infer.py --bs 2 \
        --layer0 1 1 1 \
        --layer1 1 1 1 \
        --layer2 1 1 1 \
        --layer3 1 1 1 \
    > ${logfile}/1gpu_tp.txt

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt_infer.py --bs 2 \
        --layer0 2 1 1 \
        --layer1 2 1 1 \
        --layer2 2 1 1 \
        --layer3 2 1 1 \
    > ${logfile}/2gpu_tp.txt

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt_infer.py --bs 2 \
        --layer0 2 1 2 \
        --layer1 2 1 2 \
        --layer2 2 1 2 \
        --layer3 2 1 2 \
    > ${logfile}/4gpu_tp.txt

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt_infer.py --bs 2 \
        --layer0 2 1 4 \
        --layer1 2 1 4 \
        --layer2 2 1 4 \
        --layer3 2 1 4 \
    > ${logfile}/8gpu_tp.txt


# ================== Window + Tensor Parallel ===============

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt_infer.py --bs 2 \
        --layer0 2 1 1 \
        --layer1 2 1 1 \
        --layer2 2 1 1 \
        --layer3 2 1 1 \
    > ${logfile}/2gpu_2wp2tp.txt

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt_infer.py --bs 2 \
        --layer0 2 2 1 \
        --layer1 2 2 1 \
        --layer2 2 2 1 \
        --layer3 2 1 2 \
    > ${logfile}/4gpu_4wp4tp.txt

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt_infer.py --bs 2 \
        --layer0 2 4 1 \
        --layer1 2 4 1 \
        --layer2 2 4 1 \
        --layer3 2 1 4 \
    > ${logfile}/8gpu_8wp8tp.txt

