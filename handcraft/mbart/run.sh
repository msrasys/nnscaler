# 4 gpus

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart.py --use-tp1f1b-pack --nmb 64 > 4dev64nmb-tp1f1b-pack.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/mbart/mbart.py --use-naive --nmb 64 > 4dev64nmb-naive.txt

# OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
#     handcraft/pipeline/dummy_hybrid.py --tp-size 2 --pp-size 2 --nmb 64 > 4dev64nmb-2tp2pp.txt

# 8 gpus

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart.py --use-tp1f1b-pack --nmb 128 > 8dev128nmb-tp1f1b-pack.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/mbart/mbart.py --use-naive --nmb 128 > 8dev128nmb-naive.txt
