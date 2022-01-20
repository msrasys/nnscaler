"""
Register cutomized function
"""

from functools import partial
from typing import Callable, List
import inspect
import torch

from cube.graph.operator.function.einops import IREinops

from cube.graph.parser.mapping import Sign2Op


def register(anno: str, stay: List[str] = None):
    """
    Register a function with einop annotations.

    This function is cooperated with CustomizeEinop.
    User needs to define a python function with type annotations
    for each input argument. And user needs to pass dimension annotations
    as well as (optional) frozen split dimensions (i.e., the dimensions cannot split).

    For EinDims containing brackets (e.g., (3 h d)),
    user should have same argument name in the function definition
    to help system infer each dim length, e.g.,
    
    @cube.register('a (b c) -> (a b) c')
    def funcname(x: torch.Tensor, b: int = 4):
        xxx
    
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
        print(f'registering op {func_name} with {len(args.parameters) - len(kwarg_idx)} inputs and {len(kwarg_idx)} kwargs...')
        udfop = partial(IREinops,
            name=func_name,
            anno=[anno],
            kwarg_idx=kwarg_idx, kwarg_name=kwarg_name
        )
        Sign2Op.register(func_name, udfop)
        return fn

    return decorator
