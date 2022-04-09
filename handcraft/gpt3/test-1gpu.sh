####
# Single Node Model Scaling Test
####
evaldir=eval/gpt3-coshard-v100-32gb
mkdir -p ${evaldir}

bs=4

test_naive()
{
    layers=$1
    hidden=$2
    heads=$3
    seqlen=$4
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    echo "testing naive (recompute): ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=1 \
        --nnodes=1 \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --seqlen ${seqlen} --bs ${bs} --micro-bs 1 \
            --fp16 # > ${evaldir}/1dev-${arch}-naive.txt
    sleep 5
    killall python
    sleep 5
    killall python
}

test_coshard()
{
    layers=$1
    hidden=$2
    heads=$3
    seqlen=$4
    arch=L${layers}E${hidden}H${heads}-seq${seqlen}

    echo "testing coshard: ${arch}"
    OMP_NUM_THREADS=4 torchrun \
        --nproc_per_node=1 \
        --nnodes=1 \
        handcraft/gpt3/train.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --seqlen ${seqlen} --bs ${bs} --micro-bs 1 --fp16 \
            --use-coshard --coshard-num 8 # > ${evaldir}/1dev-${arch}-coshard.txt
    sleep 5
    killall python
    sleep 5
    killall python
}


test_naive   48 5120 32 2048
test_naive   48 5120 32 4096
test_naive   48 5120 32 8192
test_naive   48 5120 32 12288

# test_naive 24 2048 32 2048
# test_naive 24 2048 32 4096
# test_naive 24 2048 32 8192
# # test_naive 24 2048 32 12288  # --# > OOM
# # test_naive 24 2048 32 16384  # --# > OOM
# 
# test_coshard 24 2048 32 2048
# test_coshard 24 2048 32 4096
# test_coshard 24 2048 32 8192
# test_coshard 24 2048 32 12288
# test_coshard 24 2048 32 16384
# test_coshard 24 2048 32 20480
# test_coshard 24 2048 32 24576
