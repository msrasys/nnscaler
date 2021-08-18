#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=62001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/mydata/LargeModel/GPT-2/webtext2/my-gpt2_text_document

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

## Optional Config ##
# --checkpoint-activations \
# NCCL_P2P_DISABLE=1
# --fp16

rm -rf /workspace/Megatron-LM/megatron/fused_kernels/build

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       /workspace/Megatron-LM/pretrain_gpt.py \
       --checkpoint-activations \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 8 \
       --num-layers 24 \
       --hidden-size 2304 \
       --num-attention-heads 24 \
       --micro-batch-size 1 \
       --global-batch-size 64 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --vocab-file /mydata/LargeModel/GPT-2/gpt2-vocab.json \
       --merge-file /mydata/LargeModel/GPT-2/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --no-masked-softmax-fusion \
       --no-bias-dropout-fusion \
       --no-bias-gelu-fusion \
       --log-interval 10
