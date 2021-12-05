bs=$1

logfile=exptrain_782M_bs${bs}_fp32_384

mkdir -p ${logfile}

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt.py --bs ${bs} \
        --layer0 1 1 1 \
        --layer1 1 1 1 \
        --layer2 1 1 1 \
        --layer3 1 1 1 \
    > ${logfile}/single.txt

# ================== Megatron Policy Parallel ===============

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt.py --bs ${bs} \
        --layer0 1 1 2 \
        --layer1 1 1 2 \
        --layer2 1 1 2 \
        --layer3 1 1 2 \
    > ${logfile}/2gpu_maxdp2tp.txt

# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=127.0.0.1 \
#     --master_port=8004 \
#     --use_env \
#     examples/swin/swin_dwt.py --bs ${bs} \
#         --layer0 2 1 2 \
#         --layer1 2 1 2 \
#         --layer2 2 1 2 \
#         --layer3 2 1 2 \
#     > ${logfile}/4gpu_maxdp2tp.txt


python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt.py --bs ${bs} \
        --layer0 4 1 2 \
        --layer1 4 1 2 \
        --layer2 4 1 2 \
        --layer3 4 1 2 \
    > ${logfile}/8gpu_maxdp2tp.txt

# ================== Maximal Tensor Parallel ===============

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt.py --bs ${bs} \
        --layer0 1 1 2 \
        --layer1 1 1 2 \
        --layer2 1 1 2 \
        --layer3 1 1 2 \
    > ${logfile}/2gpu_maxtp.txt

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt.py --bs ${bs} \
        --layer0 1 1 4 \
        --layer1 1 1 4 \
        --layer2 1 1 4 \
        --layer3 1 1 4 \
    > ${logfile}/4gpu_maxtp.txt


python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt.py --bs ${bs} \
        --layer0 2 1 4 \
        --layer1 2 1 4 \
        --layer2 2 1 4 \
        --layer3 2 1 4 \
    > ${logfile}/8gpu_maxtp.txt

# ================== Window + Tensor Parallel ===============

# python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=127.0.0.1 \
#     --master_port=8004 \
#     --use_env \
#     examples/swin/swin_dwt.py --bs ${bs} \
#         --layer0 1 2 1 \
#         --layer1 1 2 1 \
#         --layer2 1 1 2 \
#         --layer3 1 1 2 \
#     > ${logfile}/2gpu_2wp2tp.txt
# 
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=127.0.0.1 \
#     --master_port=8004 \
#     --use_env \
#     examples/swin/swin_dwt.py --bs ${bs} \
#         --layer0 1 4 1 \
#         --layer1 1 4 1 \
#         --layer2 1 1 4 \
#         --layer3 1 1 4 \
#     > exptrain_782M_bs8_fp32/4gpu_4wp4tp.txt
# 
# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=127.0.0.1 \
#     --master_port=8004 \
#     --use_env \
#     examples/swin/swin_dwt.py --bs ${bs} \
#         --layer0 1 8 1 \
#         --layer1 1 1 8 \
#         --layer2 1 1 8 \
#         --layer3 1 1 8 \
#     > ${logfile}/8gpu_8wp8tp.txt


# ================== Data + Tensor Parallel ===============
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dt.py --bs ${bs} \
        --layer0 2 1 \
        --layer1 2 1 \
        --layer2 1 2 \
        --layer3 1 2 \
    > ${logfile}/2gpu_dt_2dp2tp.txt

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dt.py --bs ${bs} \
        --layer0 4 1 \
        --layer1 4 1 \
        --layer2 1 4 \
        --layer3 1 4 \
    > ${logfile}/4gpu_dt_4dp4tp.txt

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dt.py --bs ${bs} \
        --layer0 8 1 \
        --layer1 8 1 \
        --layer2 1 8 \
        --layer3 1 8 \
    > ${logfile}/8gpu_dt_8dp8tp.txt


# ================== Pure Data Parallel =============

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt.py --bs ${bs} \
        --layer0 2 1 1 \
        --layer1 2 1 1 \
        --layer2 2 1 1 \
        --layer3 2 1 1 \
    > ${logfile}/2gpu_maxdp.txt


python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt.py --bs ${bs} \
        --layer0 4 1 1 \
        --layer1 4 1 1 \
        --layer2 4 1 1 \
        --layer3 4 1 1 \
    > ${logfile}/4gpu_maxdp.txt

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt.py --bs ${bs} \
        --layer0 8 1 1 \
        --layer1 8 1 1 \
        --layer2 8 1 1 \
        --layer3 8 1 1 \
    > ${logfile}/8gpu_maxdp.txt
