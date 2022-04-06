evaldir=eval/mbart-fp32-v100-32gb
mkdir -p ${evaldir}

bs=256

test_mix_tp_1f1b()
{
    layers=$1
    hidden=$2
    heads=$3
    gpus=$4
    echo "testing ${gpus}-dev mixture-1f1b: L${layers}E${hidden}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=8 \
      --nnodes=4 \
      --node_rank=${REMOTE_NODE_RANK} \
      --master_addr="${REMOTE_MASTER_IP}" \
      --master_port=${REMOTE_MASTER_PORT} \
      handcraft/mbart/train.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --bs ${bs} --micro-bs 1 \
        --pp-size ${gpus} --tp-size 1 \
        --schedule tp1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b.txt
    sleep 5
    killall python
    sleep 5
    killall python
}

test_tp()
{
    layers=$1
    hidden=$2
    heads=$3
    gpus=$4
    echo "testing ${gpus}-dev pure tp: L${layers}E${hidden}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=8 \
      --nnodes=4 \
      --node_rank=${REMOTE_NODE_RANK} \
      --master_addr="${REMOTE_MASTER_IP}" \
      --master_port=${REMOTE_MASTER_PORT} \
      handcraft/mbart/train.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --bs 8 --micro-bs 1 \
        --pp-size 1 --tp-size ${gpus} \
        --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt
    sleep 5
    killall python
    sleep 5
    killall python
}

test_pp()
{
    layers=$1
    hidden=$2
    heads=$3
    gpus=$4
    echo "testing ${gpus}-dev pure pp: L${layers}E${hidden}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=8 \
      --nnodes=4 \
      --node_rank=${REMOTE_NODE_RANK} \
      --master_addr="${REMOTE_MASTER_IP}" \
      --master_port=${REMOTE_MASTER_PORT} \
      handcraft/mbart/train.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --bs ${bs} --micro-bs 1 \
        --pp-size ${gpus} --tp-size 1 \
        --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-pp.txt
    sleep 5
    killall python
    sleep 5
    killall python
}

test_pp_swap()
{
    layers=$1
    hidden=$2
    heads=$3
    gpus=$4
    echo "testing ${gpus}-dev pure pp swap: L${layers}E${hidden}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=8 \
      --nnodes=4 \
      --node_rank=${REMOTE_NODE_RANK} \
      --master_addr="${REMOTE_MASTER_IP}" \
      --master_port=${REMOTE_MASTER_PORT} \
      handcraft/mbart/train.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --bs ${bs} --micro-bs 1 \
        --pp-size ${gpus} --tp-size 1 \
        --schedule 1f1b --use-swap > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-pp-swap.txt
    sleep 5
    killall python
    sleep 5
    killall python
}

test_hybrid_tp_pp()
{
    layers=$1
    hidden=$2
    heads=$3
    gpus=$4

    if [ ${gpus} == 32 ]
    then
        echo "testing ${gpus}-dev tp:pp=16:2 | L${layers}E${hidden}H${heads}"
        OMP_NUM_THREADS=4 torchrun \
          --nproc_per_node=8 \
          --nnodes=4 \
          --node_rank=${REMOTE_NODE_RANK} \
          --master_addr="${REMOTE_MASTER_IP}" \
          --master_port=${REMOTE_MASTER_PORT} \
          handcraft/mbart/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --bs ${bs} --micro-bs 1 \
            --pp-size 2 --tp-size 16 \
            --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp16pp2.txt
        sleep 5
        killall python
        sleep 5
        killall python

        # echo "testing ${gpus}-dev tp:pp=8:4 | L${layers}E${hidden}H${heads}"
        # OMP_NUM_THREADS=4 torchrun \
        #   --nproc_per_node=8 \
        #   --nnodes=4 \
        #   --node_rank=${REMOTE_NODE_RANK} \
        #   --master_addr="${REMOTE_MASTER_IP}" \
        #   --master_port=${REMOTE_MASTER_PORT} \
        #   handcraft/mbart/mbart_hybrid.py \
        #     --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        #     --bs ${bs} --micro-bs 1 \
        #     --pp-size 4 --tp-size 8 \
        #     --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp8pp4.txt
        # sleep 5
        # killall python
        # sleep 5
        # killall python
        # 
        # echo "testing ${gpus}-dev tp:pp=4:8 | L${layers}E${hidden}H${heads}"
        # OMP_NUM_THREADS=4 torchrun \
        #   --nproc_per_node=8 \
        #   --nnodes=4 \
        #   --node_rank=${REMOTE_NODE_RANK} \
        #   --master_addr="${REMOTE_MASTER_IP}" \
        #   --master_port=${REMOTE_MASTER_PORT} \
        #   handcraft/mbart/mbart_hybrid.py \
        #     --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        #     --bs ${bs} --micro-bs 1 \
        #     --pp-size 8 --tp-size 4 \
        #     --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp4pp8.txt
        # sleep 5
        # killall python
        # sleep 5
        # killall python
    fi
}


# =================================================
# selected experiments
# =================================================


test_mix_tp_1f1b  48 6144 32 32
test_tp           48 6144 32 32
test_hybrid_tp_pp 48 6144 32 32

python scripts/keep.py --gpus 8

# OOM: --layers 64 --hidden-size 6144 --heads 32
# OOM: --layers 52 --hidden-size 6144 --heads 32 -- 29.64GB
# SUC: --layers 48 --hidden-size 6144 --heads 32 -- 29.64GB
# SUC: --layers 48 --hidden-size 5120 --heads 32

# OMP_NUM_THREADS=4 torchrun \
#       --nproc_per_node=8 \
#       --nnodes=4 \
#       --node_rank=${REMOTE_NODE_RANK} \
#       --master_addr="${REMOTE_MASTER_IP}" \
#       --master_port=${REMOTE_MASTER_PORT} \
#       handcraft/mbart/train.py \
#         --layers 48 --hidden-size 6144 --heads 32 \
#         --bs 32 --micro-bs 1 \
#         --pp-size 32 --tp-size 1 \
#         --schedule tp1f1b
# 
# 
# OMP_NUM_THREADS=4 torchrun \
#       --nproc_per_node=8 \
#       --nnodes=4 \
#       --node_rank=${REMOTE_NODE_RANK} \
#       --master_addr="${REMOTE_MASTER_IP}" \
#       --master_port=${REMOTE_MASTER_PORT} \
#       handcraft/mbart/train.py \
#         --layers 52 --hidden-size 6144 --heads 32 \
#         --bs 4 --micro-bs 1 \
#         --pp-size 1 --tp-size 32 \
#         --schedule 1f1b