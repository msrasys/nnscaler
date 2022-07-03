"""
Register cutomized function
"""

from typing import Any, Callable, List, Optional
import inspect
import torch

from cube.graph.function.dimops import IRDimops

from cube.graph.parser.mapping import Sign2Op


def register(anno: str, name: Optional[str] = None):
    """
    Register a function with einop annotations.

    This function is cooperated with IREinOp.
    User needs to define a python function that satisfies
        1). Has type annotations for each input
        2). Tensor inputs goes first then other inputs

    For DimAnnos containing brackets (e.g., (3 h d)) that can not be
    inferred by system, user should have same argument name in the
    function definition to help system infer each dim length, e.g.,
    
    @cube.register('a (b c) -> (a b) c')
    def funcname(x: torch.Tensor, b: int = 4):
        xxx
    """
    def decorator(fn: Callable):
        if not callable(fn):
            raise TypeError("Expected a function")
        fsig = fn.__name__ if name is None else name
        args = inspect.signature(fn)
        arg_names = list(args.parameters.keys())
        arg_kind = [args.parameters[name].annotation for name in arg_names]
        kwarg_names = [name for (name, kind) in zip(arg_names, arg_kind) if kind != torch.Tensor]
        nkwargs = len(kwarg_names)
        ninputs = len(arg_names) - len(kwarg_names)
        # get customized op code
        code = inspect.getsource(fn)
        code = code[code.index('def'):]

        def udfop(signature: str, inputs: List[Any]):
            tensors = inputs[:ninputs]
            kwarg_vals = inputs[ninputs:]
            kwargs = dict()
            for name, val in zip(kwarg_names, kwarg_vals):
                kwargs[name] = val
            return IRDimops(signature, [anno], tensors, **kwargs, name=fsig)

        print(f'registering op {fsig} with {ninputs} inputs and {nkwargs} kwargs...')
        Sign2Op.register(fsig, udfop, code)
        return fn

    return decorator
