# 4 gpus

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-tp1f1b --nmb 4 > 4dev4nmb-tp1f1b.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-tp1f1b --nmb 8 > 4dev8nmb-tp1f1b.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-tp1f1b --nmb 64 > 4dev64nmb-tp1f1b.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-naive --nmb 4 > 4dev4nmb-naive.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-naive --nmb 8 > 4dev8nmb-naive.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-naive --nmb 64 > 4dev64nmb-naive.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-tp --nmb 64 > 4dev64nmb-tp.txt


# 8 gpus

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-tp1f1b --nmb 8 > 8dev8nmb-tp1f1b.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-tp1f1b --nmb 16 > 8dev16nmb-tp1f1b.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-tp1f1b --nmb 128 > 8dev128nmb-tp1f1b.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-naive --nmb 8 > 8dev8nmb-naive.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-naive --nmb 16 > 8dev16nmb-naive.txt

OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 --nnodes=1 \
    handcraft/pipeline/dummy.py --use-naive --nmb 128 > 8dev128nmb-naive.txt

