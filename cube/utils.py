from typing import Optional

import cube
from cube.profiler.timer import print_each_rank
from cube.runtime.device import DeviceGroup
from cube.flags import RuntimeFlag

from cube.flags import RuntimeFlag


def _load_module_attr(filename: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(filename: Optional[str] = None, load_content: bool = True):
    filename = f'gencode{DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, 'GenModel')
    loaded_module: cube.runtime.module.CubeModule = module.GenModel().cuda()
    # load parameter content
    if load_content:
        print_each_rank("> loading parameter content...")
        loaded_module.load_attr_content('./fullmodel.pt')
    # initialize reducer
    for reducer in loaded_module.reducers:
        reducer.build_buckets()
    return loaded_module


def load_default_schedule(filename: Optional[str] = None):
    filename = f'gencode{DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, '_train_step')
    return module._train_step


def load_eval_schedule(filename: Optional[str] = None):
    filename = f'gencode{DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, '_infer_step')
    return module._infer_step


class accum_mode:
    """
    Make cube execution in accumulation mode, where weight
    gradient allreduce will be skipped.

    need manually call `model.reduce_grads()` to reduce gradients 
    after finish accumulation, or make `enable=False` for the last
    accumulation step.
    """
    def __init__(self, enable: bool = True):
        self.enable = enable
        self.old = None
    
    def __enter__(self):
        self.old = RuntimeFlag.accum_mode
        RuntimeFlag.accum_mode = self.enable

    def __exit__(self, *args):
        RuntimeFlag.accum_mode = self.old
