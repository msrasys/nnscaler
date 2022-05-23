# swin transformer constant head dim == 32

evaldir=eval/swin-coshard
mkdir -p ${evaldir}

bs=256
img_size=1536
window_size=48


test_naive_pp()
{
  layers=$1
  dim=$2
  heads=$3
  gpus=$4

  echo "testing ${gpus}-dev: Pure PP${coshard}: L${layers}E${dim}H${heads}"
  OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=${gpus} \
    --nnodes=1 \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size ${gpus} --tp-size 1 --dp-size 1  \
      --bs ${bs} --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}.txt
  sleep 5
  killall python
  sleep 5
  killall python
}

test_naive_tp()
{
  layers=$1
  dim=$2
  heads=$3
  gpus=$4

  echo "testing ${gpus}-dev: Pure TP: L${layers}E${dim}H${heads}"
  OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=${gpus} \
    --nnodes=1 \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size 1 --tp-size ${gpus} --dp-size 1  \
      --bs 16 --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp${gpus}.txt
  sleep 5
  killall python
  sleep 5
  killall python
}

test_naive_hybrid_tp_pp()
{
  layers=$1
  dim=$2
  heads=$3
  gpus=$4

  if [ ${gpus} == 4 ]
  then
    echo "testing ${gpus}-dev: TP2-PP2: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=${gpus} \
      --nnodes=1 \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 2 --tp-size 2 --dp-size 1  \
        --bs ${bs} --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp2pp2.txt
    sleep 5
    killall python
    sleep 5
    killall python
  fi

  # Hybrid TP-1F1B -- 8 GPU
  if [ ${gpus} == 8 ]
  then
    echo "testing ${gpus}-dev: TP4-PP2: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=${gpus} \
      --nnodes=1 \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 2 --tp-size 4 --dp-size 1  \
        --bs ${bs} --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp4pp2.txt
    sleep 5
    killall python
    sleep 5
    killall python

    echo "testing ${gpus}-dev: TP2-PP4: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=${gpus} \
      --nnodes=1 \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 4 --tp-size 2 --dp-size 1  \
        --bs ${bs} --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp2pp4.txt
    sleep 5
    killall python
    sleep 5
    killall python
  fi
}

test_coshard_pp()
{
  layers=$1
  dim=$2
  heads=$3
  gpus=$4

  echo "testing ${gpus}-dev: Coshard PP: L${layers}E${dim}H${heads}"
  OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=${gpus} \
    --nnodes=1 \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size ${gpus} --tp-size 1 --dp-size 1  \
      --bs ${bs} --micro-bs 1 --use-coshard --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-pp${gpus}-coshard.txt
  sleep 5
  killall python
  sleep 5
  killall python
}

test_coshard_dp()
{
  layers=$1
  dim=$2
  heads=$3
  gpus=$4

  echo "testing ${gpus}-dev: Coshard DP: L${layers}E${dim}H${heads}"
  OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=${gpus} \
    --nnodes=1 \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size 1 --tp-size 1 --dp-size ${gpus} \
      --bs ${bs} --micro-bs 1 --use-coshard \
      --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-dp${gpus}-coshard.txt
  sleep 5
  killall python
  sleep 5
  killall python
}

test_coshard_hybrid_tp_pp()
{
  layers=$1
  dim=$2
  heads=$3
  gpus=$4

  if [ ${gpus} == 4 ]
  then
    echo "testing ${gpus}-dev: TP2-PP2: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=${gpus} \
      --nnodes=1 \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 2 --tp-size 2 --dp-size 1  \
        --bs ${bs} --micro-bs 1 --use-coshard \
        --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp2pp2-coshard.txt
    sleep 5
    killall python
    sleep 5
    killall python
  fi

  # Hybrid TP-1F1B -- 8 GPU
  if [ ${gpus} == 8 ]
  then
    echo "testing ${gpus}-dev: TP4-PP2: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=${gpus} \
      --nnodes=1 \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 2 --tp-size 4 --dp-size 1  \
        --bs ${bs} --micro-bs 1 --use-coshard \
        --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp4pp2-coshard.txt
    sleep 5
    killall python
    sleep 5
    killall python

    echo "testing ${gpus}-dev: TP2-PP4: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=${gpus} \
      --nnodes=1 \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 4 --tp-size 2 --dp-size 1  \
        --bs ${bs} --micro-bs 1 --use-coshard \
        --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp2pp4-coshard.txt
    sleep 5
    killall python
    sleep 5
    killall python
  fi
}

test_all()
{
  layers=$1
  dim=$2
  heads=$3
  gpus=$4
  test_naive_pp $layers $dim $heads $gpus
  test_naive_tp $layers $dim $heads $gpus
  test_naive_hybrid_tp_pp $layers $dim $heads $gpus
  test_coshard_pp $layers $dim $heads $gpus
}

# =================================================
# selected experiments
# =================================================
test_naive_tp     6  96   3  1
test_naive_tp     10 128  4  1
test_naive_tp     14 192  6  1
# test_naive_tp     18 256  8  1  # --> OOM
# test_naive_tp     26 512  16 1  # --> OOM
test_coshard_pp   6  96   3  1
test_coshard_pp   10 128  4  1
test_coshard_pp   14 192  6  1
test_coshard_pp   18 256  8  1
test_coshard_pp   26 512  16 1

test_coshard_dp         18 256 8  2
test_naive_tp           18 256 8  2
test_naive_hybrid_tp_pp 18 256 8  2

test_coshard_pp         26 512 16 4
test_naive_tp           26 512 16 4
# test_naive_hybrid_tp_pp 26 512 16 4  # --> OOM

test_coshard_pp         42 768 24 8
test_naive_tp           42 768 24 8
# test_naive_hybrid_tp_pp 42 768 24 8  # --> OOM

python scripts/keep.py --gpus 8


# for test
# coshard-pp
# gpus=4
# OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=${gpus} \
#     --nnodes=1 \
#     handcraft/swin/train.py \
#       --layers 26 --dim 512 --heads 16 \
#       --img-size 1536 --window-size 48 \
#       --pp-size ${gpus} --tp-size 1 --dp-size 1  \
#       --bs 16 --micro-bs 1 --use-coshard --fp16