# setup megatron
# git clone https://github.com/NVIDIA/Megatron-LM.git
# pip install regex

# setup apex
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
# cd ..

cp pretrain_gpt_synthetic.py ./Megatron-LM/

NODE_GPUS=8
PP=4
TP=4

GPT_ARGS="--num-layers 32 \
          --hidden-size 4096 \
          --num-attention-heads 32 \
          --seq-length 2048 \
          --max-position-embeddings 2048 \
          --lr 0.00015 \
          --train-iters 10 \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --lr-warmup-fraction .01 \
          --fp16 \
          --fp16-lm-cross-entropy \
          --no-query-key-layer-scaling \
          --no-masked-softmax-fusion \
          --no-bias-gelu-fusion \
          --no-bias-dropout-fusion \
          --no-async-tensor-model-parallel-allreduce \
          --no-gradient-accumulation-fusion \
          --checkpoint-activations \
          --log-interval 1 \
          --num-workers 0"

# --checkpoint-activations

SINGLE_NODE="--nproc_per_node $NODE_GPUS \
             --nnodes 1 \
             --node_rank 0 \
             --master_addr localhost \
             --master_port 6000"

MULTI_NODE="--nproc_per_node $NODE_GPUS \
            --nnodes 2 \
            --node_rank ${NODE_RANK} \
            --master_addr worker-0 \
            --master_port 6012" 


cd Megatron-LM

OMP_NUM_THREADS=4 python -m torch.distributed.launch $MULTI_NODE \
       pretrain_gpt_synthetic.py $GPT_ARGS \
       --global-batch-size 128 \
       --micro-batch-size 4 \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --DDP-impl local


# OMP_NUM_THREADS=4 python -m torch.distributed.launch \
#        --nproc_per_node 1 --master_addr localhost --master_port 6112 \
#        pretrain_gpt_synthetic.py -h

cd ..