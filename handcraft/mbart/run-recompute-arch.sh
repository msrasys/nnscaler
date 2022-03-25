layers=24
hidden=4096
heads=32
gpus=8

evaldir=eval/mbart-v100-32gb-pcie-recompute
mkdir -p ${evaldir}


# TP-1F1B
echo 'testing mixture-1f1b'
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --use-tp1f1b-pack --nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp1f1b-pack.txt


# # Pure 1F1B
# echo 'testing pure 1f1b'
# OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
#     handcraft/mbart/mbart.py \
#     --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
#     --use-1f1b --nmb 256 --iter-nmb 256\
#     --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-1f1b.txt

# Pure TP
echo 'testing pure tensor parallelism'
OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
    handcraft/mbart/mbart_hybrid.py \
    --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
    --tp-size ${gpus} --pp-size 1 --nmb 256 --iter-nmb 256 \
    --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp.txt

# # Hybrid TP-1F1B -- 4 GPU
# if [ ${gpus} == 4 ]
# then
#     echo 'testing hybrid tp:pp=2:2'
#     OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
#         handcraft/mbart/mbart_hybrid.py \
#         --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
#         --tp-size 2 --pp-size 2 --nmb 256 --iter-nmb 256 \
#         --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp2pp2.txt
#     sleep 5
#     killall python
#     sleep 5
#     killall python
# fi
# 
# # Hybrid TP-1F1B -- 8 GPU
# if [ ${gpus} == 8 ]
# then
#     echo 'testing hybrid tp:pp=4:2'
#     OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
#         handcraft/mbart/mbart_hybrid.py \
#         --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
#         --tp-size 4 --pp-size 2 --nmb 256 --iter-nmb 256 \
#         --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp4pp2.txt
#     sleep 5
#     killall python
#     sleep 5
#     killall python
# 
#     echo 'testing hybrid tp:pp=2:4'
#     OMP_NUM_THREADS=4 torchrun --nproc_per_node=${gpus} --nnodes=1 \
#         handcraft/mbart/mbart_hybrid.py \
#         --layers ${layers} --hidden-size ${hidden} --heads ${heads} \
#         --tp-size 2 --pp-size 4 --nmb 256 --iter-nmb 256 \
#         --use-recompute > ${evaldir}/${gpus}dev-L${layers}E${hidden}H${heads}-tp2pp4.txt
#     sleep 5
#     killall python
#     sleep 5
#     killall python
# fi

python scripts/keep.py --gpus 8
