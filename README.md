# MagicCube

AI System Compiler to map a semantic (single-device) model into distributed execution using policies specified by developers.

## Prerequisite

* Python >= 3.7

> Install Python 3.7 in the development environment for widest compatibility.

Install dependent packages
```sh
pip install -r requirements.txt

# require pytorch version >= 1.11
pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch/
# pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
```

## Option 1: Quick Start without Installation

* ### Run on repo root path:
```sh
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4  \
    --nnodes=1  \
    examples/mlp/linears.py --policy PASCol
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
python setup.py develop
```

* ### Run Example
[Micro Benchmark] Run a mutiple MLP Model

```sh
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py --policy PASCol
```


## Development Docstring

We follow [Google Style Python Docstring](https://google.github.io/styleguide/pyguide.html) for development.

Following is an typical example:

```python
class SampleClass:
    """Summary of class here.

    Longer class information...
    Longer class information...

    """

    def __init__(self, likes_spam: bool = False):
        """Initializes the instance based on spam preference.

        Args:
          likes_spam: Defines if instance exhibits this preference.
        """
        self.likes_spam = likes_spam
        self.eggs = 0

    def public_method(self, a, b):
        """Performs operation blah.

        Long description here.

        Args:
            a (int): xxx
            b (int/str): xxx

        Returns:
            t (bool): xxx
            k (int): xxx
        """
        # function implementation goes here
```

## Run unit tests

We use `tox` to run unit tests. You should install `tox` in your development environemnt
```
pip install tox
```
Currently we only use python3.10 to run tests. If you don't have python3.10 in your system, you can use conda. After conda is installed, you should install tox conda plugin by running
```
pip install tox-conda
```
After tox is ready, you can run all the unit test by running
```
tox
```
Please note tox will reuse the same virtual environment which is initialized by installing all packages listed in `requirements.txt` and `requirements-dev.txt`. If any of above files are modified, you should re-create virtual environment by running
```
tox -r
```

### Run unit tests in vscode

VS Code has a great support to unit tests. You can run/debug every tests easily in VS Code. Please refer to this document to set up your environment https://code.visualstudio.com/docs/python/testing

Another trick is, if you want to step into pakcage source code, you can add the following config to your .vscode/launch.json:
```
{
    "name": "Debug Unit Test",
    "type": "python",
    "request": "test",
    "justMyCode": false,
},
```

### Write Unit Tests
1. If you need to use torchrun, please refer to `unit_test/launch_torchrun.py`, and you can find examples in `unit_tests/runtime/test_runtime_collectives.py`. Please note that `torchrun` is very slow, you should reduce its usage as possible.
2. If you want to mock up any functions/methods, please use pytest-mock.
3. **NOTE**: The name of test files and test functions must start with `test_`
