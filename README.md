# MagicCube

AI System Compiler to map a semantic (single-device) model into distributed execution using policies specified by System Expert.

## Prerequisite

* Python >= 3.7

> Install Python 3.7 in the development environment for widest compatibility.

## Install

```python
pip install -r requirements.txt
python setup.py develop
```

## Run Examples

* [Micro Benchmark] Run a mutiple MLP Model

```sh
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.pys
```
