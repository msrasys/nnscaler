evaldir=/data/MagicCube/scale-mbart-recompute

mkdir -p ${evaldir}

# 4 gpus recompute scale=3,4,5

# TP-1F1B
# OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
#     handcraft/mbart/mbart.py --use-tp1f1b-pack --nmb 256 \
#     --scale 3 --use-recompute > ${evaldir}/4dev256nmb-scale3-tp1f1b-pack.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart.py --use-tp1f1b-pack --nmb 256 \
    --scale 4 --use-recompute > ${evaldir}/4dev256nmb-scale4-tp1f1b-pack.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart.py --use-tp1f1b-pack --nmb 256 \
    --scale 5 --use-recompute > ${evaldir}/4dev256nmb-scale5-tp1f1b-pack.txt

# Pure 1F1B
# OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
#     handcraft/mbart/mbart.py --use-1f1b --nmb 256 \
#     --scale 3 --iter-nmb 256 --use-recompute > ${evaldir}/4dev256nmb-scale3-1f1b.txt

# OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
#     handcraft/mbart/mbart.py --use-1f1b --nmb 256 \
#     --scale 4 --iter-nmb 256 --use-recompute > ${evaldir}/4dev256nmb-scale4-1f1b.txt

# OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
#     handcraft/mbart/mbart.py --use-1f1b --nmb 256 \
#     --scale 5 --iter-nmb 256 --use-recompute > ${evaldir}/4dev256nmb-scale5-1f1b.txt

# Pure TP
# OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
#     handcraft/mbart/mbart_hybrid.py --tp-size 4 --pp-size 1 --nmb 256 \
#     --scale 3 --use-recompute > ${evaldir}/4dev256nmb-scale3-tp.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 4 --pp-size 1 --nmb 256 \
    --scale 4 --use-recompute > ${evaldir}/4dev256nmb-scale4-tp.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 4 --pp-size 1 --nmb 256 \
    --scale 5 --use-recompute > ${evaldir}/4dev256nmb-scale5-tp.txt

# Hybrid TP-PP: TP=2, PP=2

# OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
#     handcraft/mbart/mbart_hybrid.py --tp-size 2 --pp-size 2 --nmb 256 \
#     --scale 3 --iter-nmb 256 --use-recompute > ${evaldir}/4dev256nmb-scale3-tp2pp2.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 2 --pp-size 2 --nmb 256 \
    --scale 4 --iter-nmb 256 --use-recompute > ${evaldir}/4dev256nmb-scale4-tp2pp2.txt

# OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
#     handcraft/mbart/mbart_hybrid.py --tp-size 2 --pp-size 2 --nmb 256 \
#     --scale 5 --iter-nmb 256 --use-recompute > ${evaldir}/4dev256nmb-scale5-tp2pp2.txt


# 8 gpus recompute scale=6,7

# TP-1F1B
OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart.py --use-tp1f1b-pack --nmb 256 \
    --scale 6 --use-recompute > ${evaldir}/8dev256nmb-scale6-tp1f1b-pack.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart.py --use-tp1f1b-pack --nmb 256 \
    --scale 7 --use-recompute > ${evaldir}/8dev256nmb-scale7-tp1f1b-pack.txt

# Pure 1F1B
# OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
#     handcraft/mbart/mbart.py --use-1f1b --nmb 256 \
#     --scale 6 --iter-nmb 256 --use-recompute > ${evaldir}/8dev256nmb--scale6-1f1b.txt

# OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
#     handcraft/mbart/mbart.py --use-1f1b --nmb 256 \
#     --scale 6 --iter-nmb 256 --use-recompute > ${evaldir}/8dev256nmb--scale7-1f1b.txt


# Pure TP
OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 8 --pp-size 1 --nmb 256 \
    --scale 6 --use-recompute > ${evaldir}/8dev256nmb-scale6-tp.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 8 --pp-size 1 --nmb 256 \
    --scale 7 --use-recompute > ${evaldir}/8dev256nmb-scale7-tp.txt


# Hybrid TP-PP: TP2-PP4, TP4-PP2
OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 2 --pp-size 4 --nmb 256 \
    --scale 6 --iter-nmb 256 --use-recompute > ${evaldir}/8dev256nmb-scale6-tp2pp4.txt

# OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
#     handcraft/mbart/mbart_hybrid.py --tp-size 2 --pp-size 4 --nmb 256 \
#     --scale 7 --iter-nmb 256 --use-recompute > ${evaldir}/8dev256nmb-scale7-tp2pp4.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 4 --pp-size 2 --nmb 256 \
    --scale 6 --iter-nmb 256 --use-recompute > ${evaldir}/8dev256nmb-scale6-tp4pp2.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 4 --pp-size 2 --nmb 256 \
    --scale 7 --iter-nmb 256 --use-recompute > ${evaldir}/8dev256nmb-scale7-tp4pp2.txt

# 16 gpus

# OMP_NUM_THREADS=4 torchrun --nproc_per_node=16 --nnodes=1 \
#     handcraft/mbart/mbart.py --use-tp1f1b-pack --nmb 256 \
#     --scale 6 > ${evaldir}/16dev256nmb-tp1f1b.txt
# 
# OMP_NUM_THREADS=4 torchrun --nproc_per_node=16 --nnodes=1 \
#     handcraft/mbart/mbart_hybrid.py --tp-size 16 --pp-size 1 --nmb 256 \
#     --scale 6 > ${evaldir}/16dev256nmb-tp.txt
# 
# OMP_NUM_THREADS=4 torchrun --nproc_per_node=16 --nnodes=1 \
#     handcraft/mbart/mbart_hybrid.py --tp-size 4 --pp-size 4 --nmb 256 \
#     --scale 6 --iter-nmb 256 > ${evaldir}/16dev256nmb-tp4pp4.txt
# 
# OMP_NUM_THREADS=4 torchrun --nproc_per_node=16 --nnodes=1 \
#     handcraft/mbart/mbart_hybrid.py --tp-size 8 --pp-size 2 --nmb 256 \
#     --scale 6 --iter-nmb 256 > ${evaldir}/16dev256nmb-tp8pp2.txt
# 
# OMP_NUM_THREADS=4 torchrun --nproc_per_node=16 --nnodes=1 \
#     handcraft/mbart/mbart_hybrid.py --tp-size 2 --pp-size 8 --nmb 256 \
#     --scale 6 --iter-nmb 256 > ${evaldir}/16dev256nmb-tp2pp8.txt

echo 'done!!!'
python scripts/keep.py --gpus 8
