####
# 2-Node Model Scaling Test
####
evaldir=eval/gpt3-coshard-v100-32gb
mkdir -p ${evaldir}

bs=256


test_hybrid()
{
    layers=$1
    hidden=$2
    heads=$3
    seqlen=$4
    gpus=$5
    dp=$6
    pp=$7
    tp=$8
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    echo "testing hybrid: dp:pp:tp=${dp}:${pp}:${tp} : ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=${NODE_RANK} \
        --master_addr="${MASTER_IP}" \
        --master_port=${MASTER_PORT} \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --dp-size ${dp} --pp-size ${pp} --tp-size ${tp} \
            --seqlen ${seqlen} --bs  ${bs} --micro-bs 1 \
            --fp16 # > ${evaldir}/${gpus}dev-${arch}-dp${dp}pp${pp}tp${tp}.txt
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
    dp=$6
    pp=$7
    tp=$8
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    echo "testing coshard hybrid: dp:pp=${dp}:${pp}:${tp} : ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=${NODE_RANK} \
        --master_addr="${MASTER_IP}" \
        --master_port=${MASTER_PORT} \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --dp-size ${dp} --pp-size ${pp} --tp-size ${tp} \
            --seqlen ${seqlen} --bs ${bs} --micro-bs 1 \
            --fp16 --use-coshard --coshard-num 8 # > ${evaldir}/${gpus}dev-${arch}-dp${dp}pp${pp}tp${tp}-coshard.txt
    sleep 5
    killall python
    sleep 5
    killall python
}

# 15B

test_hybrid         48 5120 32 2048 16 4 4 1  # dp8pp2 OOM dp4pp4 18.91GB
test_hybrid_coshard 48 5120 32 2048 16 4 4 1 # dp4pp4 20.93

# test_hybrid         48 5120 32 4096 16 16 1 1 # pp16
# test_hybrid         48 5120 32 4096 16 1 1 16 # tp16
test_hybrid         48 5120 32 4096 16 4 4 1 # dp4pp4 15.62
test_hybrid_coshard 48 5120 32 4096 16 4 4 1 # dp4pp4 20.93

# test_hybrid         48 5120 32 8192 16 16 1 1 # pp16 OOM
test_hybrid         48 5120 32 8192 16 1 8 2 # pp8tp2 # pp2tp2 17.17GB
test_hybrid_coshard 48 5120 32 8192 16 4 4 1 # dp4pp4 # dp4pp4 26.73GB

test_hybrid         48 5120 32 12288 16 1 4 4 # pp8tp2 OOM pp4tp4 20.29GB
test_hybrid_coshard 48 5120 32 12288 16 2 8 1 # dp4pp4 OOM dp2pp8 26.88GB

# ===========================

# 15B
# test_hybrid         48 5120 32 2048 16 4 4 1  # dp8pp2 OOM dp4pp4 18.91GB

# test_pp             48 5120 32 4096 16 # 12.42GB
# test_hybrid         48 5120 32 4096 16 2 8 1  # dp2pp8 15.62GB
# test_hybrid         48 5120 32 4096 16 4 4 1 # dp4pp4 15.62
# test_hybrid         48 5120 32 4096 16 8 2 1 # dp8pp2 OOM
# test_hybrid_coshard 48 5120 32 4096 16 4 4 1 # dp16 OOM dp8pp2 OOM dp4pp4 can run

# test_hybrid         48 5120 32 8192 16 # pp-dp OOM, pp8tp2: can run
# test_pp             48 5120 32 8192 16 # OOM
# test_hybrid_coshard 48 5120 32 8192 16 4 4 1 # dp4pp4

# test_hybrid         48 5120 32 12288 16 1 4 4 # pp8tp2 OOM pp4tp4 20.29GB
# test_hybrid_coshard 48 5120 32 12288 16 2 8 1 # dp4pp4 OOM dp2pp8 26.88GB
