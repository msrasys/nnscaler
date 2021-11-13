# MagicCube

AI System Compiler to compile a semantic (single-device) model to distributed model using policies specified by System Expert.

## Install

```python
pip install -r requirements.txt
python setup.py develop
```

## Run Examples

* [Micro Benchmark] Run a mutiple MLP Model

```sh
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/linears.py
```
