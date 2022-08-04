

# test MLP

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    examples/mlp/linears.py --policy PASSingle

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py --policy PASData

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py --policy PASCol

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py --policy PASRow

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py --policy PASHybrid

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py --policy PASMegatron

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py --policy PASOptimal


# test GPT model

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/nlp/gpt/train.py --policy PASMegatronTP --fp16

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/nlp/gpt/train.py --policy PASRoundRobin --fp16

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/nlp/gpt/train.py --policy PASMegatron --fp16


# test scientific model

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/poisson/sci.py

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    examples/wrf/wrf2.py
