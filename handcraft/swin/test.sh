# swin transformer constant head dim == 32

evaldir=eval/swin-coshard
mkdir -p ${evaldir}

# Swin-Giant
layers=42
dim=512
heads=16
img_size=1536
window_size=48
coshard=16
gpus=4

OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size ${gpus} --tp-size 1 --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard 16 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard${coshard}.txt


OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size 1 --tp-size ${gpus} --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp${gpus}-coshard1.txt
