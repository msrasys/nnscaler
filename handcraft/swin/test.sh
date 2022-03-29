# swin transformer constant head dim == 32

evaldir=eval/swin-coshard
mkdir -p ${evaldir}


layers=18
dim=192
heads=6
img_size=1536
window_size=48
coshard=6
gpus=4

OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size ${gpus} --tp-size 1 --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard ${coshard} --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard${coshard}.txt


OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size 1 --tp-size ${gpus} --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp${gpus}-coshard1.txt


OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size ${gpus} --tp-size 1 --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard1.txt



layers=26
dim=384
heads=12
img_size=1536
window_size=48
coshard=12
gpus=4

OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size ${gpus} --tp-size 1 --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard ${coshard} --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard${coshard}.txt


OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size 1 --tp-size ${gpus} --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp${gpus}-coshard1.txt


OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size ${gpus} --tp-size 1 --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard1.txt



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
    --bs 256 --micro-bs 1 --coshard ${coshard} --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard${coshard}.txt


OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size 1 --tp-size ${gpus} --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp${gpus}-coshard1.txt


OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size ${gpus} --tp-size 1 --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard1.txt



layers=50
dim=768
heads=24
img_size=1536
window_size=48
coshard=16
gpus=8

OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size ${gpus} --tp-size 1 --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard ${coshard} --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard${coshard}.txt


OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size 1 --tp-size ${gpus} --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp${gpus}-coshard1.txt


OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=${gpus} \
  --nnodes=1 \
  handcraft/swin/train.py \
    --layers ${layers} --dim ${dim} --heads ${heads} \
    --img-size ${img_size} --window-size ${window_size} \
    --pp-size ${gpus} --tp-size 1 --dp-size 1  \
    --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard1.txt


python scripts/keep.py --gpus 8
