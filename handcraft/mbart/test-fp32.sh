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
    OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
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
    OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
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
    OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
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
    OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
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

    if [ ${gpus} == 4 ]
    then
        echo "testing ${gpus}-dev tp:pp=2:2 | L${layers}E${hidden}H${heads}"
        OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
            handcraft/mbart/mbart_hybrid.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --bs ${bs} --micro-bs 1 \
            --pp-size 2 --tp-size 2 \
            --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp2pp2.txt
        sleep 5
        killall python
        sleep 5
        killall python
    fi

    if [ ${gpus} == 8 ]
    then
        echo "testing ${gpus}-dev tp:pp=4:2 | L${layers}E${hidden}H${heads}"
        OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
            handcraft/mbart/mbart_hybrid.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --bs ${bs} --micro-bs 1 \
            --pp-size 2 --tp-size 4 \
            --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp4pp2.txt
        sleep 5
        killall python
        sleep 5
        killall python

        echo "testing ${gpus}-dev tp:pp=2:4 | L${layers}E${hidden}H${heads}"
        OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
            handcraft/mbart/mbart_hybrid.py \
            --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
            --bs ${bs} --micro-bs 1 \
            --pp-size 2 --tp-size 4 \
            --schedule 1f1b > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp2pp4.txt
        sleep 5
        killall python
        sleep 5
        killall python
    fi
}


# =================================================
# selected experiments
# =================================================
test_tp           8  2048 16 2
test_mix_tp_1f1b  8  2048 16 2
test_hybrid_tp_pp 8  2048 16 2

test_mix_tp_1f1b  16 3072 24 4
test_tp           16 3072 24 4
test_mix_tp_1f1b  16 3072 24 8
test_tp           16 3072 24 8
# test_mix_tp_1f1b  16 3072 24 16
# test_tp           16 3072 24 16

test_mix_tp_1f1b  16 3072 24 4
test_tp           16 3072 24 4

test_mix_tp_1f1b  24 4096 32 8
test_tp           24 4096 32 8

python scripts/keep.py --gpus 8
