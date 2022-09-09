
# get megatron
# git clone https://github.com/NVIDIA/Megatron-LM.git

cp pretrain_gpt_synthetic.py ./Megatron-LM/

GPUS=8

GPT_ARGS="--num-layers 24 \
          --hidden-size 1024 \
          --num-attention-heads 16 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 1 \
          --global-batch-size 1 \
          --lr 0.00015 \
          --train-iters 200 \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --lr-warmup-fraction .01 \
          --fp16 \
          --fp16-lm-cross-entropy \
          --no-query-key-layer-scaling \
          --no-masked-softmax-fusion \
          --no-bias-gelu-fusion \
          --no-bias-dropout-fusion"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"


cd Megatron-LM

OMP_NUM_THREADS=4 python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt_synthetic.py $GPT_ARGS \
       --tensor-model-parallel-size ${GPUS}\
       --pipeline-model-parallel-size 1 \
       --DDP-impl torch

cd ..