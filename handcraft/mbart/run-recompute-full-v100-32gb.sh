evaldir=eval/mbart-v100-32gb-pcie-recompute

mkdir -p ${evaldir}

# =================================================
# 4 gpus: arch layer 21,21, hidden 1792, heads 28
# =================================================
layers=21
hidden=1792
heads=28
gpus=4

echo "testing mixture-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-tp1f1b-pack --nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b-pack.txt

echo "testing pure-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-1f1b --nmb 256 --iter-nmb 256\
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-1f1b.txt

echo "testing pure tensor parallelism: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --tp-size ${gpus} --pp-size 1 --nmb 256 --iter-nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt

echo "testing tensor x pipeline parallelism 2x2: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
        handcraft/mbart/mbart_hybrid.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --tp-size 2 --pp-size 2 --nmb 256 --iter-nmb 256 \
        --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp2pp2.txt


# =================================================
# 4 gpus: arch layer 24,24, hidden 2048, heads 32
# =================================================
layers=24
hidden=2048
heads=32
gpus=4

echo "testing mixture-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-tp1f1b-pack --nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b-pack.txt

echo "testing pure-1f1b: L${layers}E${hidden}H${heads}"
echo "Will be OOM"

echo "testing pure tensor parallelism: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --tp-size ${gpus} --pp-size 1 --nmb 256 --iter-nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt

echo "testing tensor x pipeline parallelism 2x2: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
        handcraft/mbart/mbart_hybrid.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --tp-size 2 --pp-size 2 --nmb 256 --iter-nmb 256 \
        --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp2pp2.txt


# =================================================
# 4 gpus: arch layer 24,24, hidden 2560, heads 32
# =================================================
layers=24
hidden=2560
heads=32
gpus=4

echo "testing mixture-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-tp1f1b-pack --nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b-pack.txt

echo "testing pure-1f1b: L${layers}E${hidden}H${heads}"
echo "Will be OOM"

echo "testing pure tensor parallelism: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --tp-size ${gpus} --pp-size 1 --nmb 256 --iter-nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt

echo "testing tensor x pipeline parallelism 2x2: L${layers}E${hidden}H${heads}"
echo "Will be OOM"


# =================================================
# 4 gpus: arch layer 18,18, hidden 3072, heads 32
# =================================================
layers=18
hidden=3072
heads=32
gpus=4

echo "testing mixture-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-tp1f1b-pack --nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b-pack.txt

echo "testing pure-1f1b: L${layers}E${hidden}H${heads}"
echo "Will be OOM"

echo "testing pure tensor parallelism: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --tp-size ${gpus} --pp-size 1 --nmb 256 --iter-nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt

echo "testing tensor x pipeline parallelism 2x2: L${layers}E${hidden}H${heads}"
echo "Will be OOM"


# =================================================
# 4 gpus: arch layer 27,27, hidden 3072, heads 32
# =================================================
layers=27
hidden=2304
heads=36
gpus=4

echo "testing mixture-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-tp1f1b-pack --nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b-pack.txt

echo "testing pure-1f1b: L${layers}E${hidden}H${heads}"
echo "Will be OOM"

echo "testing pure tensor parallelism: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --tp-size ${gpus} --pp-size 1 --nmb 256 --iter-nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt

echo "testing tensor x pipeline parallelism 2x2: L${layers}E${hidden}H${heads}"
echo "Will be OOM"


# =================================================
# 8 gpus: arch layer 24,24, hidden 2048, heads 32
# =================================================
layers=24
hidden=2048
heads=32
gpus=8

echo "testing mixture-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-tp1f1b-pack --nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b-pack.txt

echo "testing pure-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-1f1b --nmb 256 --iter-nmb 256\
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-1f1b.txt


echo "testing pure tensor parallelism: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --tp-size ${gpus} --pp-size 1 --nmb 256 --iter-nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt

echo "testing pure tensor parallelism 2x4: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
        handcraft/mbart/mbart_hybrid.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --tp-size 2 --pp-size 4 --nmb 256 --iter-nmb 256 \
        --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp2pp4.txt

echo "testing tensor x pipeline parallelism 2x4: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
        handcraft/mbart/mbart_hybrid.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --tp-size 4 --pp-size 2 --nmb 256 --iter-nmb 256 \
        --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp4pp2.txt

# =================================================
# 8 gpus: arch layer 30,30, hidden 2560, heads 40
# =================================================
layers=30
hidden=2560
heads=40
gpus=8

echo "testing mixture-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-tp1f1b-pack --nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b-pack.txt

echo "testing pure tensor parallelism: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --tp-size ${gpus} --pp-size 1 --nmb 256 --iter-nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt

echo "testing pure tensor parallelism 2x4: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
        handcraft/mbart/mbart_hybrid.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --tp-size 2 --pp-size 4 --nmb 256 --iter-nmb 256 \
        --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp2pp4.txt

echo "testing tensor x pipeline parallelism 2x4: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
        handcraft/mbart/mbart_hybrid.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --tp-size 4 --pp-size 2 --nmb 256 --iter-nmb 256 \
        --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp4pp2.txt


# =================================================
# 8 gpus: arch layer 33,33, hidden 2816, heads 40
# =================================================
layers=33
hidden=2816
heads=48
gpus=8

echo "testing mixture-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-tp1f1b-pack --nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b-pack.txt

echo "testing pure tensor parallelism: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --tp-size ${gpus} --pp-size 1 --nmb 256 --iter-nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt


echo "testing tensor x pipeline parallelism 4x2: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
        handcraft/mbart/mbart_hybrid.py \
        --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
        --tp-size 4 --pp-size 2 --nmb 256 --iter-nmb 256 \
        --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp4pp2.txt

# =================================================
# 8 gpus: arch layer 24,24, hidden 4096, heads 32
# =================================================
layers=24
hidden=4096
heads=32
gpus=8

echo "testing mixture-1f1b: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-tp1f1b-pack --nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b-pack.txt

echo "testing pure tensor parallelism: L${layers}E${hidden}H${heads}"
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --tp-size ${gpus} --pp-size 1 --nmb 256 --iter-nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt


echo "testing tensor x pipeline parallelism 4x2: L${layers}E${hidden}H${heads}"
echo "Will OOM"


echo 'done!!!'
# python scripts/keep.py --gpus 8
