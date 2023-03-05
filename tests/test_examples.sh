# NOTE: This test should run in the root directory.
# Before running this test, you should run `export PYTHONPATH=.:$PYTHONPATH` first.

# test torch.fx
# working path <repo_root>
OMP_NUM_THREADS=12 USE_TORCHFX=1 PYTHONPATH=.:$PYTHONPATH \
    python -m torch.distributed.launch \
    --nproc_per_node=1 \
    examples/mlp/linearsfx.py --policy PASData

# test MLP

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    examples/mlp/train.py --policy PASSingle

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/train.py --policy PASData

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/train.py --policy PASCol

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/train.py --policy PASRow

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/train.py --policy PASHybrid

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/train.py --policy PASMegatronTP

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/train.py --policy PASOptimal

ASYNC_COMM=1 OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/infer.py --policy PASMegatron


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
    examples/nlp/gpt/train.py --policy PAS1F1B --fp16

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/nlp/gpt/train.py --policy PASMegatron --fp16

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/nlp/gpt/train.py --policy PASMeshShard --fp16

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    examples/nlp/gpt/infer.py --policy PASDP --fp16


# test Swin model

# OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=4 \
#     --nnodes=1 \
#     examples/vision/swin/train.py --policy PASData --fp16

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/vision/swin/train.py --policy PASMegatronTP --fp16

# OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=4 \
#     --nnodes=1 \
#     examples/vision/swin/train.py --policy PASMegatron --fp16


# test scientific model

# OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=4 \
#     --nnodes=1 \
#     examples/poisson/sci.py
# 
# OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=1 \
#     --nnodes=1 \
#     examples/wrf/wrf2.py --policy PAS
# 
# OMP_NUM_THREADS=1 torchrun \
#     --nproc_per_node=4 \
#     --nnodes=1 \
#     examples/wrf/wrf2.py --policy PAS_ALL_Y
