# MagicCube

AI System Compiler to map a semantic (single-device) model into distributed execution using policies specified by System Expert.

## Prerequisite

* Python >= 3.7

> Install Python 3.7 in the development environment for widest compatibility.

Install dependent packages
```shell
pip install -r requirements.txt
```

## Option 1: Quick Start without Installation

* ### Run on repo root path:
```sh
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4  \
    --nnodes=1  \
    examples/mlp/linears.py
```

[comment]: <> (UDA_VISIBLE_DEVICES=7 PYTHONPATH=.:$PYTHONPATH python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 ./examples/wrf/wrf2.py)

* ### Debug for model parsing check on single Device
```shell
PYTHONPATH=.:$PYTHONPATH SINGLE_DEV_MODE=1 python examples/mlp/linears.py 
```


---

## Option 2: Install for Run

* ### Install

```python
pip install -r requirements.txt
python setup.py develop
```

* ### Run Example 
[Micro Benchmark] Run a mutiple MLP Model

```sh
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py
```

## Profile

### Use cProfile

Due to the multi-process architecture of `torch.distributed.launch`, instead of directly using
the command-line interface of cProfile, we need to exactly specify the scope to profile, like:

```python
import cProfile
prof = cProfile.Profile()
prof.enable()

# our code to profile goes here
@cube.compile(...)
def iter(dataloader):
    x, y = next(dataloader)
    z = model(x, y)
    return z
for i in range(N):
    iter(...)
# our code ends

prof.disabled()
pr.dump_stats('cube_%d.prof' % torch.distributed.get_rank()) # or use TID/PID, if to profile multi-thread/-process program.
```

After the modification, run the Python file using the same command line with `torchrun` as usual.
