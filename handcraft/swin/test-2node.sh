# swin transformer constant head dim == 32

evaldir=eval/swin-coshard
mkdir -p ${evaldir}

rm -f notify.py
wget https://raw.githubusercontent.com/zhiqi-0/EnvDeployment/master/email/notify.py

img_size=1536
window_size=48
bs=256


test_naive_pp()
{
  layers=$1
  dim=$2
  heads=$3
  nodes=$4
  gpus=$5
  arch=L${layers}E${dim}H${heads}-${img_size}

  echo "testing ${gpus}-dev: Pure PP${coshard}: L${layers}E${dim}H${heads}"
  OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=${nodes} \
    --node_rank=${NODE_RANK} \
    --master_addr="${MASTER_IP}" \
    --master_port=${MASTER_PORT} \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size ${gpus} --tp-size 1 --dp-size 1  \
      --bs ${bs} --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-${arch}-pp${gpus}.txt
  sleep 5
  killall python
  sleep 5
  killall python
  python notify.py --sender zhiqi.0@qq.com --code uyakwgslumknbfgg --recver zhiqi.0@outlook.com \
    --msg "Test Results Swin PP | Node ${NODE_RANK} | ${evaldir}/${gpus}dev-${arch}-pp${gpus}.txt" \
    --file ${evaldir}/${gpus}dev-${arch}-pp${gpus}.txt
}

test_naive_tp()
{
  layers=$1
  dim=$2
  heads=$3
  nodes=$4
  gpus=$5
  arch=L${layers}E${dim}H${heads}-${img_size}

  echo "testing ${gpus}-dev: Pure TP: L${layers}E${dim}H${heads}"
  OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=${nodes} \
    --node_rank=${NODE_RANK} \
    --master_addr="${MASTER_IP}" \
    --master_port=${MASTER_PORT} \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size 1 --tp-size ${gpus} --dp-size 1  \
      --bs 16 --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-${arch}-tp${gpus}.txt
  sleep 5
  killall python
  sleep 5
  killall python
  python notify.py --sender zhiqi.0@qq.com --code uyakwgslumknbfgg --recver zhiqi.0@outlook.com \
    --msg "Test Results Swin TP | Node ${NODE_RANK} | ${evaldir}/${gpus}dev-${arch}-tp${gpus}.txt" \
    --file ${evaldir}/${gpus}dev-${arch}-tp${gpus}.txt
}

test_naive_hybrid_tp_pp()
{
  layers=$1
  dim=$2
  heads=$3
  nodes=$4
  gpus=$5

  # Hybrid TP-1F1B -- 16 GPU
  if [ ${gpus} == 16 ]
  then
    echo "testing ${gpus}-dev: TP8-PP2: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=8 \
      --nnodes=${nodes} \
      --node_rank=${NODE_RANK} \
      --master_addr="${MASTER_IP}" \
      --master_port=${MASTER_PORT} \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 2 --tp-size 8 --dp-size 1  \
        --bs ${bs} --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp8pp2.txt
    sleep 5
    killall python
    sleep 5
    killall python

    echo "testing ${gpus}-dev: TP4-PP4: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=8 \
      --nnodes=${nodes} \
      --node_rank=${NODE_RANK} \
      --master_addr="${MASTER_IP}" \
      --master_port=${MASTER_PORT} \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 4 --tp-size 4 --dp-size 1  \
        --bs ${bs} --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp4pp4.txt
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
  nodes=$4
  gpus=$5
  arch=L${layers}E${dim}H${heads}-${img_size}

  echo "testing ${gpus}-dev: Pure TP: L${layers}E${dim}H${heads}"
  OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=${nodes} \
    --node_rank=${NODE_RANK} \
    --master_addr="${MASTER_IP}" \
    --master_port=${MASTER_PORT} \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size ${gpus} --tp-size 1 --dp-size 1  \
      --bs ${bs} --micro-bs 1 --use-coshard
      --fp16 > ${evaldir}/${gpus}dev-${arch}-pp${gpus}-coshard.txt
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
  nodes=$4
  gpus=$5
  arch=L${layers}E${dim}H${heads}-${img_size}

  # Hybrid TP-1F1B -- 8 GPU
  if [ ${gpus} == 16 ]
  then
    # echo "testing ${gpus}-dev: TP8-PP2: L${layers}E${dim}H${heads}"
    # OMP_NUM_THREADS=4 torchrun \
    #   --nproc_per_node=8 \
    #   --nnodes=${nodes} \
    #   --node_rank=${NODE_RANK} \
    #   --master_addr="${MASTER_IP}" \
    #   --master_port=${MASTER_PORT} \
    #   handcraft/swin/train.py \
    #     --layers ${layers} --dim ${dim} --heads ${heads} \
    #     --img-size ${img_size} --window-size ${window_size} \
    #     --pp-size 2 --tp-size 8 --dp-size 1  \
    #     --bs 64 --micro-bs 1 --use-coshard --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp8pp2-coshard.txt
    # sleep 5
    # killall python
    # sleep 5
    # killall python
  
    echo "testing coshard ${gpus}-dev: TP4-PP4: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=8 \
      --nnodes=${nodes} \
      --node_rank=${NODE_RANK} \
      --master_addr="${MASTER_IP}" \
      --master_port=${MASTER_PORT} \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 4 --tp-size 4 --dp-size 1  \
        --bs ${bs} --micro-bs 1 --use-coshard --use-inner-coshard \
        --fp16 > ${evaldir}/${gpus}dev-${arch}-tp4pp4-coshard.txt
    sleep 5
    killall python
    sleep 5
    killall python
    python notify.py --sender zhiqi.0@qq.com --code uyakwgslumknbfgg --recver zhiqi.0@outlook.com \
        --msg "Test Results Swin TP4-PP4+Coshard | Node ${NODE_RANK} | ${evaldir}/${gpus}dev-${arch}-tp4pp4-coshard.txt" \
        --file ${evaldir}/${gpus}dev-${arch}-tp4pp4-coshard.txt
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

test_naive_tp             50 1024 32 2 16
test_coshard_hybrid_tp_pp 50 1024 32 2 16
# test_naive_hybrid_tp_pp   50 1024 32 2 16  # -> OOM

python scripts/keep.py --gpus 8

# OMP_NUM_THREADS=4 torchrun \
#   --nproc_per_node=8 \
#   --nnodes=2 \
#   --node_rank=${NODE_RANK} \
#   --master_addr="${MASTER_IP}" \
#   --master_port=${MASTER_PORT} \
#   handcraft/swin/train.py \
#     --layers 50 --dim 1024 --heads 32 \
#     --img-size 1536 --window-size 48 \
#     --pp-size 4 --tp-size 4 --dp-size 1  \
#     --bs 256 --micro-bs 1 --use-coshard --use-inner-coshard \
#     --fp16
