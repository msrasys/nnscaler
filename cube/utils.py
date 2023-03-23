from typing import Optional
from cube.profiler.timer import print_each_rank
from cube.runtime.device import DeviceGroup


def _load_module_attr(filename: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(filename: Optional[str] = None, load_content: bool = True):
    filename = f'gencode{DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, 'GenModel')
    loaded_module = module.GenModel().cuda()
    if load_content:
        print_each_rank("> loading parameter content...")
        loaded_module.load_attr_content('./fullmodel.pt')
    return loaded_module


def load_default_schedule(filename: Optional[str] = None):
    filename = f'gencode{DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, '_train_step')
    return module._train_step


def load_eval_schedule(filename: Optional[str] = None):
    filename = f'gencode{DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, '_infer_step')
    return module._infer_step

