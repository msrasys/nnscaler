"""
Register cutomized function
"""

from typing import Any, Callable, List, Optional
import inspect
import torch

from cube.graph.function.dimops import IRDimops, OpAnno, TransformRule

from cube.graph.parser.mapping import Sign2Op


def register(anno: str, name: Optional[str] = None, rules: Optional[List[TransformRule]] = None):
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

    Note: for Optional[torch.Tensor] type, user should annotate the
    dimension when the input is not None.
    """
    def decorator(fn: Callable):
        if not callable(fn):
            raise TypeError("Expected a function")
        fsig = fn.__name__
        op_name = name if name is not None else fsig
        args = inspect.signature(fn)
        arg_names = list(args.parameters.keys())
        arg_kinds = [args.parameters[name].annotation for name in arg_names]
        allow_types = (torch.Tensor, Optional[torch.Tensor])
        for ninputs, kind in enumerate(arg_kinds):
            if kind in allow_types:
                ninputs += 1
                continue
            assert not any(k in allow_types for k in arg_kinds[ninputs:]), \
                f"Type of {allow_types} should be consecutive in parameter order."
            break
        nkwargs = len(arg_names) - ninputs
        kwarg_names = [name for name in arg_names[ninputs:]]
        # get customized op code
        code = inspect.getsource(fn)
        code = code[code.index('def'):]

        def udfop(signature: str, inputs: List[Any]):
            manno = OpAnno(anno)
            tensors = inputs[:ninputs]
            for idx in range(ninputs):
                if arg_kinds[idx] == Optional[torch.Tensor] and tensors[idx] is None:
                    manno.set_input(idx, '?')
            kwarg_vals = inputs[ninputs:]
            kwargs = dict()
            for name, val in zip(kwarg_names, kwarg_vals):
                kwargs[name] = val
            return IRDimops(udfop, op_name, signature, [repr(manno)], tensors, transform_rules=rules, **kwargs)

        print(f'registering op {fsig} with {ninputs} inputs and {nkwargs} kwargs...')
        Sign2Op.register(fsig, udfop, code)
        return fn

    return decorator
