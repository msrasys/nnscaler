# swin transformer constant head dim == 32

evaldir=eval/swin-coshard
mkdir -p ${evaldir}


img_size=1536
window_size=48


test()
{
  layers=$1
  dim=$2
  heads=$3
  coshard=$4
  gpus=$5

  echo "testing ${gpus}-dev: PP-Coshard${coshard}: L${layers}E${dim}H${heads}"
  echo "OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=${gpus} \
    --nnodes=1 \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size ${gpus} --tp-size 1 --dp-size 1  \
      --bs 256 --micro-bs 1 --coshard ${coshard} --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard${coshard}.txt"

  echo "testing ${gpus}-dev: TP-Coshard1: L${layers}E${dim}H${heads}"
  echo "OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=${gpus} \
    --nnodes=1 \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size 1 --tp-size ${gpus} --dp-size 1  \
      --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp${gpus}-coshard1.txt"

  echo "testing ${gpus}-dev: PP-Coshard1: L${layers}E${dim}H${heads}"
  echo "OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=${gpus} \
    --nnodes=1 \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size ${gpus} --tp-size 1 --dp-size 1  \
      --bs 256 --micro-bs 1 --coshard 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard1.txt"

  killall python
  sleep 5
  killall python
}

# test Layers Dim Heads Coshard GPUs
test 18 256 8  8  4
test 18 512 16 16 4
test 18 768 24 24 4

test 26 512 16 16 4
test 26 768 24 24 4
test 26 1024 32 32 4

test 34 256 8  8  8
test 34 512 16 16 8
test 34 768 24 24 8
test 34 1024 32 32 8

python scripts/keep.py --gpus 8
