evaldir=/data/MagicCube/scale-mbart

mkdir -p ${evaldir}

# 4 gpus

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart.py --use-tp1f1b-pack --nmb 256 \
    --scale 3 > ${evaldir}/4dev256nmb-tp1f1b-pack.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart.py --use-1f1b --nmb 256 \
    --scale 3 --iter-nmb 1 > ${evaldir}/4dev256nmb-1f1b.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 4 --pp-size 1 --nmb 256 \
    --scale 3 > ${evaldir}/4dev256nmb-tp.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 2 --pp-size 2 --nmb 256 \
    --scale 3 --iter-nmb 256 > ${evaldir}/4dev256nmb-tp2pp2.txt


# 8 gpus

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart.py --use-tp1f1b-pack --nmb 256 \
    --scale 4 > ${evaldir}/8dev256nmb-tp1f1b-pack.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart.py --use-1f1b --nmb 256 \
    --scale 4 --iter-nmb 1 > ${evaldir}/8dev256nmb-1f1b.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 8 --pp-size 1 --nmb 256 \
    --scale 4 > ${evaldir}/8dev256nmb-tp.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 4 --pp-size 2 --nmb 256 \
    --scale 4 --iter-nmb 256 > ${evaldir}/8dev256nmb-tp4pp2.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py --tp-size 2 --pp-size 4 --nmb 256 \
    --scale 4 --iter-nmb 256 > ${evaldir}/8dev256nmb-tp2pp4.txt


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
