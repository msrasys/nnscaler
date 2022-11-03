#!/bin/bash
# run at MagicCube/
# ./benchmark/deepspeed/benchmark_gpt.sh

# get commit ID:
# git rev-parse --short HEAD

# installation
# pip install deepspeed==0.7.4
# git clone https://github.com/microsoft/Megatron-DeepSpeed
# git checkout 54f1cb7

# note DeepSpeed can do:
# 1) PP > 1 with constraints of Zero-Stage=1
# 2) TP > 1 with constraints of Zero-Stage < 3

cp benchmark/deepspeed/pretrain_gpt_synthetic.py \
  benchmark/deepspeed/Megatron-DeepSpeed/

Nnodes=1
TP=2
PP=2

# Model arch
Layers=12
Hidden=2048
Heads=32
Seqlen=2048

# batch size
Gbs=8
Mbs=1
Accum=$(( ${Gbs} / ( ${Nnodes} * 8 / ${TP} / ${PP} * ${Mbs} ) ))
echo "Accumulated steps: ${Accum}"

# zero stage config
Zero=1
OFFLOAD_DEVICE="none"  
CPU_OPTIM=" "
#OFFLOAD_DEVICE="cpu"
#CPU_OPTIM=" --cpu-optimizer"

cd benchmark/deepspeed/Megatron-DeepSpeed

DS_CONFIG=ds_config.json

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $Gbs,
  "train_micro_batch_size_per_gpu": $Mbs,
  "steps_per_print": 1,
  "gradient_accumulation_steps": ${Accum},
  "zero_optimization": {
    "stage": $Zero,
    "stage3_max_live_parameters": 3e9,
    "stage3_max_reuse_distance": 3e9,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_prefetch_bucket_size": 5e7,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 90000000,
    "sub_group_size": 1e9,
    "offload_optimizer": {
      "device": "$OFFLOAD_DEVICE",
      "buffer_count": 4,
      "pipeline_read": false,
      "pipeline_write": false,
      "pin_memory": true
    }
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "initial_scale_power" : 15,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "wall_clock_breakdown": true,
  "zero_allow_untested_optimizer": false,
  "aio": {
    "block_size": 1048576,
    "queue_depth": 16,
    "single_submit": false,
    "overlap_events": true,
    "thread_count": 2
  }
}
EOT

# export NCCL_DEBUG=warn 

ds_args=" "
ds_args=" --deepspeed ${ds_args}"
# ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$Zero ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


GPT_ARGS="--num-layers $Layers \
          --hidden-size $Hidden \
          --num-attention-heads $Heads \
          --seq-length $Seqlen \
          --loss-scale 15 \
          --max-position-embeddings $Seqlen \
          --train-iters 3 \
          --lr 6.0e-5 \
          --min-lr 6.0e-6 \
          --lr-decay-style cosine \
          --fp16 \
          --fp16-lm-cross-entropy \
          --no-query-key-layer-scaling \
          --no-masked-softmax-fusion \
          --no-bias-gelu-fusion \
          --no-bias-dropout-fusion \
          --checkpoint-activations \
          --adam-beta1 0.9 \
          --adam-beta2 0.95 \
          --weight-decay 0.1 \
          --clip-grad 1.0 \
          --init-method-std 0.006 \
          --log-interval 1 \
          --num-workers 0"

# deepspeed --force_multi --num_nodes
deepspeed --num_nodes=$Nnodes --num_gpus 8 \
    --master_addr localhost --master_port 6144 \
    pretrain_gpt_synthetic.py \
    $GPT_ARGS $CPU_OPTIM $ds_args \
    --global-batch-size $Gbs \
    --micro-batch-size $Mbs \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP

cd ../../..
