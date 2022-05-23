datadir=eval/sharding
mkdir -p ${datadir}

# hidden=768
# heads=12
# 
# OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=1 \
#     --nnodes=1 \
#     handcraft/playground/transformers.py \
#     --hidden-size ${hidden} --heads ${heads} \
#     --seq 1 > ${datadir}/sharding-E${hidden}H${heads}-naive.txt
# 
# OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=1 \
#     --nnodes=1 \
#     handcraft/playground/transformers.py \
#     --hidden-size ${hidden} --heads ${heads} \
#     --seq 8 > ${datadir}/sharding-E${hidden}H${heads}-shard8.txt


hidden=1024
heads=16

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 1 > ${datadir}/sharding-E${hidden}H${heads}-naive.txt

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 8 > ${datadir}/sharding-E${hidden}H${heads}-shard8.txt


hidden=1536
heads=16

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 1 > ${datadir}/sharding-E${hidden}H${heads}-naive.txt

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 8 > ${datadir}/sharding-E${hidden}H${heads}-shard8.txt

hidden=2304
heads=24

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 1 > ${datadir}/sharding-E${hidden}H${heads}-naive.txt

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 8 > ${datadir}/sharding-E${hidden}H${heads}-shard8.txt


hidden=2560
heads=32

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 1 > ${datadir}/sharding-E${hidden}H${heads}-naive.txt

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 8 > ${datadir}/sharding-E${hidden}H${heads}-shard8.txt



hidden=4096
heads=32

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 1 > ${datadir}/sharding-E${hidden}H${heads}-naive.txt

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 8 > ${datadir}/sharding-E${hidden}H${heads}-shard8.txt

hidden=5120
heads=40

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 1 > ${datadir}/sharding-E${hidden}H${heads}-naive.txt

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 8 > ${datadir}/sharding-E${hidden}H${heads}-shard8.txt


hidden=12288
heads=96

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 1 > ${datadir}/sharding-E${hidden}H${heads}-naive.txt

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/playground/transformers.py \
    --hidden-size ${hidden} --heads ${heads} \
    --seq 8 > ${datadir}/sharding-E${hidden}H${heads}-shard8.txt