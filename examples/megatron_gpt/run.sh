# Usage: bash run.sh mode = {trace, compile, run, all}

MEGATRON_PATH=/home/ningshang/Megatron-LM
TENSORBOARD_DIR=/data/ningshang/megatron_gpt
DATA_PATH=/data/ningshang/torchscale_data
TORCHSCALE_PATH=/home/ningshang/anaconda3/envs/cube/lib/python3.10/site-packages/examples/fairseq
FAIRSEQ_PATH=/home/ningshang/Fairseq

export USE_TORCHFX=1
export LOG_PARSER=1
export DISABLE_CODE_LINE_INFO=0

PLAN_NGPUS=1
CUBE_SCALING_FACTOR=2


# check arg num
if [ $# -ne 1 ]
then
    echo "Usage: bash run.sh mode = {trace, compile, run}"
    exit 1
fi

MODE=$1

if [ $MODE = "trace" ]
then
    VOCAB_FILE=./gpt2-vocab.json
    MERGE_FILE=./gpt2-merges.txt
    GPT_ARGS="
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --seq-length 128 \
        --max-position-embeddings 128
    "
    USELESS_ARGS="
        --micro-batch-size 4 \
        --global-batch-size 8 \
        --lr 0.00015 \
        --train-iters 500000 \
        --lr-decay-iters 320000 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --lr-warmup-fraction .01 \
        --clip-grad 1.0
    "
    DATA_ARGS="
        --data-path $DATA_PATH/train \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --data-impl mmap \
        --split 949,50,1
    "
    OUTPUT_ARGS="
        --log-interval 100 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 10
    "
    PYTHONPATH=.:PYTHONPATH:$TORCHSCALE_PATH:$MEGATRON_PATH CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 1 --nnodes 1 convert.py $GPT_ARGS $DATA_ARGS $OUTPUT_ARGS $USELESS_ARGS >trace_log.txt 2>&1
elif [ $MODE = "compile" ]
then
    PLAN_NGPUS=$PLAN_NGPUS CUBE_SCALING_FACTOR=$CUBE_SCALING_FACTOR PYTHONPATH=.:PYTHONPATH:$TORCHSCALE_PATH:$MEGATRON_PATH:$FAIRSEQ_PATH python parallel.py >compile_log.txt 2>&1
elif [ $MODE = "run" ]
then
    PLAN_NGPUS=$PLAN_NGPUS PYTHONPATH=.:PYTHONPATH:$TORCHSCALE_PATH:$MEGATRON_PATH torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    $TORCHSCALE_PATH/train.py $DATA_PATH \
    --num-workers 2 \
    --activation-fn gelu \
    --share-decoder-input-output-embed \
    --arch lm_base_125M \
    --validate-interval-updates 1000 \
    --save-interval-updates 1000 \
    --log-interval 1 \
    --task language_modeling \
    --sample-break-mode none \
    --tokens-per-sample 128 \
    --optimizer adam \
    --adam-betas "(0.9,0.999)" \
    --adam-eps 1e-08 \
    --clip-norm 1.0 \
    --lr 6.0e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 230 \
    --dropout 0.0 \
    --attention-dropout 0.0 \
    --weight-decay 0.01 \
    --batch-size 16 \
    --update-freq 1 \
    --required-batch-size-multiple 1 \
    --total-num-update 5000 \
    --max-update 5000 \
    --seed 1234 \
    --ddp-backend=legacy_ddp \
    --cube-scaling-factor $CUBE_SCALING_FACTOR \
    --subln --xpos-rel-pos \
    --parallel-backend=cube \
    --compile=run_only \
    --tensorboard-logdir $TENSORBOARD_DIR \
    --save-dir=/data/ningshang/checkpoint >run_log.txt 2>&1
else
    echo "Usage: bash run.sh mode = {trace, compile, run}"
    exit 1
fi
