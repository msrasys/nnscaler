# swin transformer constant head dim == 32

evaldir=eval/swin-coshard
mkdir -p ${evaldir}


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
    --node_rank=${REMOTE_NODE_RANK} \
    --master_addr="${REMOTE_MASTER_IP}" \
    --master_port=${REMOTE_MASTER_PORT} \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size ${gpus} --tp-size 1 --dp-size 1  \
      --bs ${bs} --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-${arch}-pp${gpus}.txt
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
  nodes=$4
  gpus=$5
  arch=L${layers}E${dim}H${heads}-${img_size}

  echo "testing ${gpus}-dev: Pure TP: L${layers}E${dim}H${heads}"
  OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=${nodes} \
    --node_rank=${REMOTE_NODE_RANK} \
    --master_addr="${REMOTE_MASTER_IP}" \
    --master_port=${REMOTE_MASTER_PORT} \
    handcraft/swin/train.py \
      --layers ${layers} --dim ${dim} --heads ${heads} \
      --img-size ${img_size} --window-size ${window_size} \
      --pp-size 1 --tp-size ${gpus} --dp-size 1  \
      --bs 16  --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-${arch}-tp${gpus}.txt
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
  nodes=$4
  gpus=$5
  arch=L${layers}E${dim}H${heads}-${img_size}

  # Hybrid TP-1F1B -- 16 GPU
  if [ ${gpus} == 32 ]
  then
    echo "testing ${gpus}-dev: TP16-PP2: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=8 \
      --nnodes=${nodes} \
      --node_rank=${REMOTE_NODE_RANK} \
      --master_addr="${REMOTE_MASTER_IP}" \
      --master_port=${REMOTE_MASTER_PORT} \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 2 --tp-size 16 --dp-size 1  \
        --bs ${bs} --micro-bs 1 \
        --fp16 > ${evaldir}/${gpus}dev-${arch}-tp8pp2.txt
    sleep 5
    killall python
    sleep 5
    killall python

    # echo "testing ${gpus}-dev: TP8-PP4: L${layers}E${dim}H${heads}"
    # OMP_NUM_THREADS=4 torchrun \
    #   --nproc_per_node=8 \
    #   --nnodes=${nodes} \
    #   --node_rank=${REMOTE_NODE_RANK} \
    #   --master_addr="${REMOTE_MASTER_IP}" \
    #   --master_port=${REMOTE_MASTER_PORT} \
    #   handcraft/swin/train.py \
    #     --layers ${layers} --dim ${dim} --heads ${heads} \
    #     --img-size ${img_size} --window-size ${window_size} \
    #     --pp-size 4 --tp-size 8 --dp-size 1  \
    #     --bs ${bs} --micro-bs 1 --fp16 > ${evaldir}/${gpus}dev-L${layers}E${dim}H${heads}-${img_size}-tp4pp4.txt
    # sleep 5
    # killall python
    # sleep 5
    # killall python
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
    --node_rank=${REMOTE_NODE_RANK} \
    --master_addr="${REMOTE_MASTER_IP}" \
    --master_port=${REMOTE_MASTER_PORT} \
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
  if [ ${gpus} == 32 ]
  then
    # echo "testing ${gpus}-dev: TP16-PP2: L${layers}E${dim}H${heads}"
    # OMP_NUM_THREADS=4 torchrun \
    #   --nproc_per_node=8 \
    #   --nnodes=${nodes} \
    #   --node_rank=${REMOTE_NODE_RANK} \
    #   --master_addr="${REMOTE_MASTER_IP}" \
    #   --master_port=${REMOTE_MASTER_PORT} \
    #   handcraft/swin/train.py \
    #     --layers ${layers} --dim ${dim} --heads ${heads} \
    #     --img-size ${img_size} --window-size ${window_size} \
    #     --pp-size 2 --tp-size 16 --dp-size 1  \
    #     --bs 64 --micro-bs 1 --use-coshard --use-inner-coshard \
    #     --fp16 > ${evaldir}/${gpus}dev-${arch}-tp16pp2-coshard.txt
    # sleep 5
    # killall python
    # sleep 5
    # killall python
  
    echo "testing coshard ${gpus}-dev: TP8-PP4: L${layers}E${dim}H${heads}"
    OMP_NUM_THREADS=4 torchrun \
      --nproc_per_node=8 \
      --nnodes=${nodes} \
      --node_rank=${REMOTE_NODE_RANK} \
      --master_addr="${REMOTE_MASTER_IP}" \
      --master_port=${REMOTE_MASTER_PORT} \
      handcraft/swin/train.py \
        --layers ${layers} --dim ${dim} --heads ${heads} \
        --img-size ${img_size} --window-size ${window_size} \
        --pp-size 4 --tp-size 8 --dp-size 1  \
        --bs ${bs} --micro-bs 1 --use-coshard --use-inner-coshard \
        --fp16 > ${evaldir}/${gpus}dev-${arch}-tp8pp4-coshard.txt
    sleep 5
    killall python
    sleep 5
    killall python

    # echo "testing coshard ${gpus}-dev: TP4-PP8: L${layers}E${dim}H${heads}"
    # OMP_NUM_THREADS=4 torchrun \
    #   --nproc_per_node=8 \
    #   --nnodes=${nodes} \
    #   --node_rank=${REMOTE_NODE_RANK} \
    #   --master_addr="${REMOTE_MASTER_IP}" \
    #   --master_port=${REMOTE_MASTER_PORT} \
    #   handcraft/swin/train.py \
    #     --layers ${layers} --dim ${dim} --heads ${heads} \
    #     --img-size ${img_size} --window-size ${window_size} \
    #     --pp-size 8 --tp-size 4 --dp-size 1  \
    #     --bs ${bs} --micro-bs 1 --use-coshard --use-inner-coshard \
    #     --fp16 > ${evaldir}/${gpus}dev-${arch}-tp8pp4-coshard.txt
    # sleep 5
    # killall python
    # sleep 5
    # killall python
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

test_naive_tp             58 1536 32 4 32
test_coshard_hybrid_tp_pp 58 1536 32 4 32
# test_naive_hybrid_tp_pp   58 1536 32 4 32  # -> OOM

python scripts/keep.py --gpus 8



# ============ exp
# Fail: 50 1280 32 | COSHARD-TP: TP4PP8 Fail TP: ? Hybrid-TP: ?
# TEST: 50 1536 32 | COSHARD-TP: ? TP4PP8 ALL Fail TP8PP4 SUC TP: ? Hybrid-TP: ?
# TEST: 58 1536 32 | COSHARD-TP: ? TP4PP8 ? Fail TP8PP4 SUC TP: ? Hybrid-TP: ?
# FAIL: 50 1536 64 | COSHARD-TP: ? TP8PP4 Fail TP: ? Hybrid-TP: ?
# FAIL: 50 2048 64 | COSHARD-TP: ? TP8PP4 Fail TP: ? Hybrid-TP: ?


# coshard
# layers=58
# dim=1536
# heads=32
# OMP_NUM_THREADS=4 torchrun \
#       --nproc_per_node=8 \
#       --nnodes=4 \
#       --node_rank=${REMOTE_NODE_RANK} \
#       --master_addr="${REMOTE_MASTER_IP}" \
#       --master_port=${REMOTE_MASTER_PORT} \
#       handcraft/swin/train.py \
#         --layers ${layers} --dim ${dim} --heads ${heads} \
#         --img-size 1536 --window-size 48 \
#         --pp-size 8 --tp-size 4 --dp-size 1  \
#         --bs 8 --micro-bs 1 --use-coshard --use-inner-coshard \
#         --fp16
# 
# # hybrid tp
# OMP_NUM_THREADS=4 torchrun \
#   --nproc_per_node=8 \
#   --nnodes=4 \
#   --node_rank=${REMOTE_NODE_RANK} \
#   --master_addr="${REMOTE_MASTER_IP}" \
#   --master_port=${REMOTE_MASTER_PORT} \
#   handcraft/swin/train.py \
#     --layers ${layers} --dim ${dim} --heads ${heads} \
#     --img-size 1536 --window-size 48 \
#     --pp-size 2 --tp-size 16 --dp-size 1  \
#     --bs 4 --micro-bs 1 \
#     --fp16
# 
# OMP_NUM_THREADS=4 torchrun \
#   --nproc_per_node=8 \
#   --nnodes=4 \
#   --node_rank=${REMOTE_NODE_RANK} \
#   --master_addr="${REMOTE_MASTER_IP}" \
#   --master_port=${REMOTE_MASTER_PORT} \
#   handcraft/swin/train.py \
#     --layers ${layers} --dim ${dim} --heads ${heads} \
#     --img-size 1536 --window-size 48 \
#     --pp-size 1 --tp-size 32 --dp-size 1  \
#     --bs 2 --micro-bs 1 --fp16
# 
# clear
# killall python
