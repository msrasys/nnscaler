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
      --nnodes=2 \
      --node_rank=${NODE_RANK} \
      --master_addr="${MASTER_IP}" \
      --master_port=${MASTER_PORT} \
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
      --nnodes=2 \
      --node_rank=${NODE_RANK} \
      --master_addr="${MASTER_IP}" \
      --master_port=${MASTER_PORT} \
      handcraft/mbart/train.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --bs ${bs} --micro-bs 1 \
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
      --nnodes=2 \
      --node_rank=${NODE_RANK} \
      --master_addr="${MASTER_IP}" \
      --master_port=${MASTER_PORT} \
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
      --nnodes=2 \
      --node_rank=${NODE_RANK} \
      --master_addr="${MASTER_IP}" \
      --master_port=${MASTER_PORT} \
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

    if [ ${gpus} == 16 ]
    then
        echo "testing ${gpus}-dev tp:pp=8:2 | L${layers}E${hidden}H${heads}"
        OMP_NUM_THREADS=4 torchrun \
          --nproc_per_node=8 \
          --nnodes=2 \
          --node_rank=${NODE_RANK} \
          --master_addr="${MASTER_IP}" \
          --master_port=${MASTER_PORT} \
          handcraft/mbart/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --bs ${bs} --micro-bs 1 \
            --pp-size 2 --tp-size 8 \
            --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp8pp2.txt
        sleep 5
        killall python
        sleep 5
        killall python

        # echo "testing ${gpus}-dev tp:pp=4:4 | L${layers}E${hidden}H${heads}"
        # OMP_NUM_THREADS=4 torchrun \
        #   --nproc_per_node=8 \
        #   --nnodes=2 \
        #   --node_rank=${NODE_RANK} \
        #   --master_addr="${MASTER_IP}" \
        #   --master_port=${MASTER_PORT} \
        #   handcraft/mbart/train.py \
        #     --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        #     --bs ${bs} --micro-bs 1 \
        #     --pp-size 4 --tp-size 4 \
        #     --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp4pp4.txt
        # sleep 5
        # killall python
        # sleep 5
        # killall python
        # 
        # echo "testing ${gpus}-dev tp:pp=2:8 | L${layers}E${hidden}H${heads}"
        # OMP_NUM_THREADS=4 torchrun \
        #   --nproc_per_node=8 \
        #   --nnodes=2 \
        #   --node_rank=${NODE_RANK} \
        #   --master_addr="${MASTER_IP}" \
        #   --master_port=${MASTER_PORT} \
        #   handcraft/mbart/train.py \
        #     --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        #     --bs ${bs} --micro-bs 1 \
        #     --pp-size 8 --tp-size 2 \
        #     --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp8pp2.txt
        # sleep 5
        # killall python
        # sleep 5
        # killall python
    fi
}


# =================================================
# selected experiments
# =================================================

# strong scalability test
# test_mix_tp_1f1b  16 3072 24 16
# test_tp           16 3072 24 16

# model scaling test
test_mix_tp_1f1b  36 5120 32 16
test_tp           36 5120 32 16
# test_hybrid_tp_pp 40 5120 32 16  # --> OOM
# test_hybrid_tp_pp 36 5120 32 16  # --> OOM

python scripts/keep.py --gpus 8
