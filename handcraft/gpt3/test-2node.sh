####
# 2-Node Model Scaling Test
####
evaldir=eval/gpt3-coshard-v100-32gb
mkdir -p ${evaldir}

bs=256

test_pp()
{
    layers=$1
    hidden=$2
    heads=$3
    seqlen=$4
    gpus=$5
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    echo "testing pipeline 1f1b: ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=${NODE_RANK} \
        --master_addr="${MASTER_IP}" \
        --master_port=${MASTER_PORT} \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --pp-size ${gpus} --tp-size 1 \
            --seqlen ${seqlen} --bs ${bs} --micro-bs 1 \
            --fp16 > ${evaldir}/${gpus}dev-${arch}-pp.txt
    sleep 5
    killall python
    sleep 5
    killall python
}

test_pp_coshard()
{
    layers=$1
    hidden=$2
    heads=$3
    seqlen=$4
    gpus=$5
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    echo "testing coshard: ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=${NODE_RANK} \
        --master_addr="${MASTER_IP}" \
        --master_port=${MASTER_PORT} \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --pp-size ${gpus} --tp-size 1 \
            --seqlen ${seqlen} --bs ${bs} --micro-bs 1 --fp16 \
            --use-coshard --coshard-num 8 > ${evaldir}/${gpus}dev-${arch}-pp-coshard.txt
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
    seqlen=$4
    gpus=$5
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    echo "testing tp: ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=${NODE_RANK} \
        --master_addr="${MASTER_IP}" \
        --master_port=${MASTER_PORT} \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --pp-size ${gpus} --tp-size 1 \
            --seqlen ${seqlen} --bs 16 --micro-bs 1 \
            --fp16 > ${evaldir}/${gpus}dev-${arch}-tp.txt
    sleep 5
    killall python
    sleep 5
    killall python
}

test_hybrid()
{
    layers=$1
    hidden=$2
    heads=$3
    seqlen=$4
    gpus=$5
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    # echo "testing hybrid: dp:pp=2:8 : ${arch}"
    # OMP_NUM_THREADS=4 torchrun \
    #     --nproc_per_node=8 \
    #     --nnodes=2 \
    #     --node_rank=${NODE_RANK} \
    #     --master_addr="${MASTER_IP}" \
    #     --master_port=${MASTER_PORT} \
    #     handcraft/gpt3/train.py \
    #         --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    #         --dp-size 2 --pp-size 8 \
    #         --seqlen ${seqlen} --bs  ${bs} --micro-bs 1 \
    #         --fp16 > ${evaldir}/${gpus}dev-${arch}-dp2pp8.txt
    # sleep 5
    # killall python
    # sleep 5
    # killall python

    echo "testing hybrid: tp:pp=2:8 : ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=${NODE_RANK} \
        --master_addr="${MASTER_IP}" \
        --master_port=${MASTER_PORT} \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --tp-size 2 --pp-size 8 \
            --seqlen ${seqlen} --bs  ${bs} --micro-bs 1 \
            --fp16 > ${evaldir}/${gpus}dev-${arch}-tp2pp8.txt
    sleep 5
    killall python
    sleep 5
    killall python

    # echo "testing hybrid: dp:pp=4:4 : ${arch}"
    # OMP_NUM_THREADS=4 torchrun \
    #     --nproc_per_node=8 \
    #     --nnodes=2 \
    #     --node_rank=${NODE_RANK} \
    #     --master_addr="${MASTER_IP}" \
    #     --master_port=${MASTER_PORT} \
    #     handcraft/gpt3/train.py \
    #         --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    #         --dp-size 4 --pp-size 4 \
    #         --seqlen ${seqlen} --bs ${bs} --micro-bs 1 \
    #         --fp16 > ${evaldir}/${gpus}dev-${arch}-dp4pp4.txt
    # sleep 5
    # killall python
    # sleep 5
    # killall python
    # 
    # echo "testing hybrid: dp:pp=8:2 : ${arch}"
    # OMP_NUM_THREADS=4 torchrun \
    #     --nproc_per_node=8 \
    #     --nnodes=2 \
    #     --node_rank=${NODE_RANK} \
    #     --master_addr="${MASTER_IP}" \
    #     --master_port=${MASTER_PORT} \
    #     handcraft/gpt3/train.py \
    #         --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    #         --dp-size 8 --pp-size 2 \
    #         --seqlen ${seqlen} --bs ${bs} --micro-bs 1 \
    #         --fp16 > ${evaldir}/${gpus}dev-${arch}-dp8pp2.txt
    # sleep 5
    # killall python
    # sleep 5
    # killall python
}


test_dp()
{
    layers=$1
    hidden=$2
    heads=$3
    seqlen=$4
    gpus=$5
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    echo "testing dp: ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=${NODE_RANK} \
        --master_addr="${MASTER_IP}" \
        --master_port=${MASTER_PORT} \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --dp-size ${gpus} --pp-size 1 --tp-size 1 \
            --seqlen ${seqlen} --bs 16 --micro-bs 1 \
            --fp16 > ${evaldir}/${gpus}dev-${arch}-dp.txt
    sleep 5
    killall python
    sleep 5
    killall python
}

test_dp_coshard()
{
    layers=$1
    hidden=$2
    heads=$3
    seqlen=$4
    gpus=$5
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    echo "testing DP coshard: ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=${NODE_RANK} \
        --master_addr="${MASTER_IP}" \
        --master_port=${MASTER_PORT} \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --dp-size ${gpus} --pp-size 1 --tp-size 1 \
            --seqlen ${seqlen} --bs ${bs} --micro-bs 1 --fp16 \
            --use-coshard --coshard-num 8 > ${evaldir}/${gpus}dev-${arch}-dp-coshard.txt
    sleep 5
    killall python
    sleep 5
    killall python
}


test_hybrid_coshard()
{
    layers=$1
    hidden=$2
    heads=$3
    seqlen=$4
    gpus=$5
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    # echo "testing coshard hybrid: dp:pp=2:8 : ${arch}"
    # OMP_NUM_THREADS=4 torchrun \
    #     --nproc_per_node=8 \
    #     --nnodes=2 \
    #     --node_rank=${NODE_RANK} \
    #     --master_addr="${MASTER_IP}" \
    #     --master_port=${MASTER_PORT} \
    #     handcraft/gpt3/train.py \
    #         --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    #         --dp-size 2 --pp-size 8 \
    #         --seqlen ${seqlen} --bs  ${bs} --micro-bs 1 \
    #         --fp16 --use-coshard --coshard-num 8 > ${evaldir}/${gpus}dev-${arch}-dp2pp8-coshard.txt
    # sleep 5
    # killall python
    # sleep 5
    # killall python

    echo "testing coshard hybrid: dp:pp=4:4 : ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=${NODE_RANK} \
        --master_addr="${MASTER_IP}" \
        --master_port=${MASTER_PORT} \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --dp-size 4 --pp-size 4 \
            --seqlen ${seqlen} --bs ${bs} --micro-bs 1 \
            --fp16 --use-coshard --coshard-num 8 > ${evaldir}/${gpus}dev-${arch}-dp4pp4-coshard.txt
    sleep 5
    killall python
    sleep 5
    killall python

    # echo "testing coshard hybrid: dp:pp=8:2 : ${arch}"
    # OMP_NUM_THREADS=4 torchrun \
    #     --nproc_per_node=8 \
    #     --nnodes=2 \
    #     --node_rank=${NODE_RANK} \
    #     --master_addr="${MASTER_IP}" \
    #     --master_port=${MASTER_PORT} \
    #     handcraft/gpt3/train.py \
    #         --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    #         --dp-size 8 --pp-size 2 \
    #         --seqlen ${seqlen} --bs ${bs} --micro-bs 1 \
    #         --fp16 --use-coshard --coshard-num 8 > ${evaldir}/${gpus}dev-${arch}-dp4pp4-coshard.txt
    # sleep 5
    # killall python
    # sleep 5
    # killall python

}

# 15B
test_hybrid         48 5120 32 8192 16
test_hybrid_coshard 48 5120 32 8192 16

# ===========================

# test_pp 24 8192 64 2048 8  # 15.45 GB
# test_pp 24 8192 64 4096 8  # 22.84 GB
# test_pp 24 8192 64 8192 8  # OOM
# test_tp 24 8192 64 8192 8

# 2.6B
# test_pp_coshard 32 2560 32 2048 1  # 12.24 GB
# test_pp         32 2560 32 2048 1  # can run
# test_pp         32 2560 32 4096 1 # 15.5GB
# test_pp         32 2560 32 8192 1 # 28.38 GB
# test_dp         32 2560 32 8192 4 # 28.38 GB


# 6.7B
# test_dp         32 4096 32 4096 8 # OOM
# test_hybrid     32 4096 32 4096 8 # 18.99GB
# test_hybrid     32 4096 32 8192 8  # pp2dp4 oom, pp4dp2: 26.06GB
# test_dp_coshard 32 4096 32 8192 8 # OOM
# test_hybrid_coshard 32 4096 32 8192 8 # pp2dp4: 20.4GB
# test_hybrid     32 4096 32 12288 8  # all OOM
# test_pp     32 4096 32 12288 8 # OOM
# test_pp_coshard 32 4096 32 12288 8 #  16.73GB
# test_hybrid_coshard 32 4096 32 12288 8 # dp4pp2 OOM, dp2pp4: 25.17GB

# 15B
# test_hybrid         48 5120 32 4096 16  -> pp8tp2 15.62GB
# test_hybrid         48 5120 32 8192 16 # pp-dp OOM, pp8tp2: can run
# test_pp             48 5120 32 8192 16 # OOM
# test_hybrid_coshard 48 5120 32 8192 16 # can run
# test_hybrid         48 5120 32 6144 16 # pp8dp2 can run
# test_hybrid         32 4096 32 12288 16 # pp8dp2 OOM, pp8tp2 25.10G
# test_pp             32 4096 32 12288 16 # OOM