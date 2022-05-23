# MagicCube

AI System Compiler to map a semantic (single-device) model into distributed execution using policies specified by System Expert.

## Prerequisite

* Python >= 3.7
* PyTorch >= 1.9

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
