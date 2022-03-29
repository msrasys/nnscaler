evaldir=eval/mbart-v100-32gb-pcie-recompute

mkdir -p ${evaldir}

# =================================================
# 4 gpus: arch layer 21,21, hidden 1792, heads 28
# =================================================
layers=24
hidden=2048
heads=32
gpus=4

echo "testing pure-1f1b-swap: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-1f1b --nmb 256 --iter-nmb 256\
    --use-recompute --use-swap > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-1f1b-swap.txt



layers=24
hidden=2560
heads=32
gpus=4

echo "testing pure-1f1b-swap: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-1f1b --nmb 256 --iter-nmb 256\
    --use-recompute --use-swap > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-1f1b-swap.txt


layers=18
hidden=3072
heads=32
gpus=4

echo "testing pure-1f1b-swap: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-1f1b --nmb 256 --iter-nmb 256\
    --use-recompute --use-swap > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-1f1b-swap.txt


layers=27
hidden=2304
heads=36
gpus=4

echo "testing pure-1f1b-swap: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-1f1b --nmb 256 --iter-nmb 256\
    --use-recompute --use-swap > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-1f1b-swap.txt


layers=30
hidden=2560
heads=40
gpus=8

echo "testing pure-1f1b-swap: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-1f1b --nmb 256 --iter-nmb 256\
    --use-recompute --use-swap > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-1f1b-swap.txt


layers=33
hidden=2816
heads=48
gpus=8

echo "testing pure-1f1b-swap: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-1f1b --nmb 256 --iter-nmb 256\
    --use-recompute --use-swap > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-1f1b-swap.txt


layers=24
hidden=4096
heads=32
gpus=8

echo "testing pure-1f1b-swap: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-1f1b --nmb 256 --iter-nmb 256\
    --use-recompute --use-swap > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-1f1b-swap.txt


# python scripts/keep.py --gpus 8