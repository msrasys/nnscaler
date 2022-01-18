"""
Register cutomized function
"""

from functools import partial
from typing import Callable, List
import inspect
import torch

from cube.graph.operator.function import CustomizeEinop

from cube.graph.parser.mapping import Sign2Op


def register(anno: str, stay: List[str] = None):
    """
    Register a function with einop annotations.
    """
    if stay is None:
        stay = list()

    def decorator(fn: Callable):
        if not callable(fn):
            raise TypeError("Expected a function")
        args = inspect.signature(fn)
        arg_names = list(args.parameters.keys())
        arg_kind = [args.parameters[name].annotation for name in arg_names]
        func_name = fn.__name__
        kwarg_idx = list()
        kwarg_name = list()
        for idx, (name, kind) in enumerate(zip(arg_names, arg_kind)):
            if kind != torch.Tensor:
                kwarg_name.append(name)
                kwarg_idx.append(idx)
        udfop = partial(CustomizeEinop,
            name=func_name,
            anno=anno, stay=stay,
            kwarg_idx=kwarg_idx, kwarg_name=kwarg_name
        )
        Sign2Op.register(func_name, udfop)
        return fn

    return decorator
