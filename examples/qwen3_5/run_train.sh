#!/bin/bash
# Qwen3.5-4B nnscaler 8-GPU training
set -e
cd "$(dirname "$0")"
PYTHON=/home/yileiyang/agens/.venv_fa4/bin/python
$PYTHON -m torch.distributed.run --nproc_per_node=8 train.py "$@"
