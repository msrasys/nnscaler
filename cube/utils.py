import os
from typing import Optional, Tuple, Callable, List, Set, Any
import logging
from pathlib import Path
import sys
from collections import defaultdict

import cube
from cube.runtime.device import DeviceGroup
from cube.flags import RuntimeFlag, CompileFlag

import torch

_logger = logging.getLogger(__name__)


def print_each_rank(msg: str, rank_only: Optional[int] = None, logger: Optional[logging.Logger] = None):
    """Logging the message.

    Args:
        msg (str): message to be logged.
        rank_only (int, optional):
            the rank to be logged. Defaults to None, which means all ranks.
        logger (logging.Logger, optional):
            the logger to use. Defaults to print.

    Returns:
        None
    """
    logger_fn = print if logger is None else logger.info
    if CompileFlag.dev_mode:
        logger_fn(msg)
        return

    myrank = torch.distributed.get_rank()
    for rank in range(torch.distributed.get_world_size()):
        if rank_only is None:
            if myrank == rank:
                logger_fn('rank [{}]: {}'.format(rank, msg))
        else:
            if myrank == rank_only and rank_only == rank:
                logger_fn('rank [{}]: {}'.format(rank, msg))
        torch.distributed.barrier()


def _load_module_attr(filename: str, name: str):
    # TODO: use `importlib.import_module` instead
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module  # so you can find the loaded module in sys.modules
    return module


def load_model(filename: Optional[str] = None, load_content: bool = True, fullmodel_filename: Optional[str] = None):
    filename = f'gencode{DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, Path(filename).stem)
    loaded_module: cube.runtime.module.CubeModule = module.GenModel().cuda()
    # load parameter content
    if load_content:
        _logger.info("loading parameter content...")
        if not fullmodel_filename:
            fullmodel_filename = str(Path(filename).with_name('fullmodel.pt'))
        loaded_module.load_attr_content(fullmodel_filename)
    # initialize reducer
    for reducer in loaded_module.reducers:
        reducer.build_buckets()
    return loaded_module


def load_default_schedule(filename: Optional[str] = None):
    filename = f'gencode{DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, Path(filename).stem)
    return module._train_step


def load_eval_schedule(filename: Optional[str] = None):
    filename = f'gencode{DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, Path(filename).stem)
    return module._infer_step


def get_member_by_name(model: torch.nn.Module, name: str) -> Any:
    """
    Get the member of the model by its full name.
    if name is empty, return the model itself.
    """
    if not name:
        return model
    sliced_names = name.split(".")
    model_attr = model
    for sliced_name in sliced_names:
        model_attr = getattr(model_attr, sliced_name)
    return model_attr


def get_shared_params(model: torch.nn.Module) -> List[List[str]]:
    paramid2name = defaultdict(set)
    for name in model.state_dict().keys():
        param = get_member_by_name(model, name)
        paramid = id(param)
        paramid2name[paramid].add(name)
    return [list(names) for _, names in paramid2name.items() if len(names) > 1]


class accum_mode:
    """Make cube execution in gradient accumulation mode.

    This is only required when `ASYNC_REDUCER=1`.

    A typical usage is:

    ```
    for _ in range(num_iters):
        for step in range(accum_steps):
            datas = next(dataloader)
            with cube.accum_mode(begin=(step == 0), end=(step == accum_steps - 1)):
                train_iter(model, *datas)
        optimizer.step()
        optimizer.zero_grad()
    ```

    Or,

    ```
    for _ in range(num_iters):
        for step in cube.accum_mode.steps(accum_steps):
            datas = next(dataloader)
            train_iter(model, *datas)
        optimizer.step()
        optimizer.zero_grad()
    ```
    """
    def __init__(self, begin: bool = True, end: bool = True):
        """Turn on/off accumulation mode.

        Args:
            begin (bool): Whether the iteration is the first accumulation step.
                If True, the `model.zero_grad()` will be enabled to zero out gradients
                of the parameters in the reducer.
            end (bool): Whether the iteration is the last accumulation step.
                If True, the `model.reduce_grad()` will be enabled to reduce gradients at
                the end of the iteration.
        """
        self.begin: bool = begin
        self.end: bool = end
        self.old: Tuple[bool, bool] = None

    def __enter__(self):
        """Enter the accumulation mode.

        Example usage:

        ```
        for _ in range(num_iters):
            for step in range(accum_steps):
                datas = next(dataloader)
                with cube.accum_mode(begin=(step == 0), end=(step == accum_steps - 1)):
                    train_iter(model, *datas)
            optimizer.step()
            optimizer.zero_grad()
        ```

        """
        self.old = (RuntimeFlag.skip_zero_grad, RuntimeFlag.skip_reducer)
        RuntimeFlag.skip_zero_grad = (not self.begin)
        RuntimeFlag.skip_reducer = (not self.end)

    def __exit__(self, *args):
        RuntimeFlag.skip_zero_grad, RuntimeFlag.skip_reducer = self.old
        self.old = None

    @staticmethod
    def steps(nsteps: int):
        """Perform the accumulation in `nsteps` steps.

        This interface doesn't require to set the `begin` and `end` flags
        during the initilization of `accum_mode`.

        Example usage:

        ```
        for _ in range(num_iters):
            for step in cube.accum_mode.steps(accum_steps):
                datas = next(dataloader)
                train_iter(model, *datas)
            optimizer.step()
            optimizer.zero_grad()
        ```

        Args:
            nsteps (int): The number of accumulation steps.

        Yield:
            int: The current step index.
        """
        old = (RuntimeFlag.skip_zero_grad, RuntimeFlag.skip_reducer)
        for step in range(nsteps):
            RuntimeFlag.skip_zero_grad = (not (step == 0))
            RuntimeFlag.skip_reducer = (not (step == nsteps - 1))
            yield step
        RuntimeFlag.skip_zero_grad, RuntimeFlag.skip_reducer = old
