"""
Register cutomized function
"""

from typing import Dict, Callable, List, Optional, Any
from functools import partial
import inspect
import logging

import torch

from cube.graph.function.dimops import IRDimops, OpAnno

_logger = logging.getLogger(__name__)


class CustomizedOps:
    """Customized op registry."""

    # signature -> IRDimop creation function
    kOpMap: Dict[str, Callable] = {}
    # singature -> runtime function 
    kOpRuntime: Dict[str, Callable] = {}
    # signature -> runtime function implementation code
    kOpCodeDef: Dict[str, str] = {}
    # original signature (xxx.xxx.xxx) -> pure signature (xxx_xxx_xxx)
    kOpSignMap: Dict[str, str] = {}
    # the function in it will be autowrapped by tracer.
    kOpAutowrap: List[str] = []

    @staticmethod
    def map(signature: str) -> Callable:
        """Get IRDimop creation function by signature
        
        Args:
            signature (str): operator signature

        Returns:
            Callable: IRDimop creation function
        """
        signature = CustomizedOps.pure_signature(signature)
        if signature in CustomizedOps.kOpMap:
            return partial(CustomizedOps.kOpMap[signature], signature=signature)
        else:
            raise KeyError(f"{signature} is not found in registered ops")

    @staticmethod
    def exist(signature: str) -> bool:
        """Check if the signature is registered"""
        signature = CustomizedOps.pure_signature(signature)
        return signature in CustomizedOps.kOpMap

    @staticmethod
    def register(signature: str, op: Callable, code: str, runtime_fn: Callable,
                 keep_full_name: bool = False, trace_autowrap: bool = True):
        """Register an operator

        Args:
            signature (str): operator signature
            op (Callable): IRDimop creation function
            code (str): runtime function implementation code
            runtime_fn (Callable): runtime function
            keep_full_name (bool): if set True, the full name will be kept, `.` in name will be replaced to `_`
            trace_autowrap (bool): if set True, the function will be autowrapped by tracer.

        Returns:
            None
        """
        builtins = ['_operator', 'torch', 'cube.runtime.function']
        if any(signature.startswith(builtin) for builtin in builtins):
            raise RuntimeError(f"Cannot register operators with signature starting from any of {builtins}")
        signature = CustomizedOps.create_pure_signature(signature, keep_full_name)
        assert signature not in CustomizedOps.kOpMap, f"function {signature} is already registered"
        CustomizedOps.kOpMap[signature] = op
        CustomizedOps.kOpRuntime[signature] = runtime_fn
        CustomizedOps.kOpCodeDef[signature] = code
        if trace_autowrap and signature not in CustomizedOps.kOpAutowrap:
            CustomizedOps.kOpAutowrap.append(signature)
        elif not trace_autowrap and signature in CustomizedOps.kOpAutowrap:
            CustomizedOps.kOpAutowrap.pop(signature)

    @staticmethod
    def create_pure_signature(signature: str, keep_full_name: bool) -> str:
        if keep_full_name:
            pure_signature = signature.replace('__main__.', '', 1) if signature.startswith('__main__.') else signature
            pure_signature = pure_signature.replace('.', '_')
            CustomizedOps.kOpSignMap[signature] = pure_signature
            return pure_signature
        return signature.split('.')[-1]

    @staticmethod
    def pure_signature(signature: str) -> str:
        if signature in CustomizedOps.kOpSignMap:
            return CustomizedOps.kOpSignMap[signature]
        return signature.split('.')[-1]


def register(anno: str, name: Optional[str] = None,
             rules: Optional[List] = None,
             input_type_annos: Optional[List[Any]] = None,
             code_impl_pattern: str = 'import') -> Callable:
    """
    Register a function with einop annotations.

    This function is cooperated with IRDimops.
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

    Args:
        anno (str): operator annotation
        name (str): operator name
        rules (Optional[List[TransformRule]]):
            additional transformation rules.
        input_type_annos (Optional[List[Any]]):
            type annotations for inputs. If not provided, the function 
            should be annotated with types.
        code_impl_pattern (str):
            can only be 'import' or 'source'. If 'import', will generate code with
            import statement. If 'source', will take the source code directly.
            Default: 'import'.

    Returns:
        fn (Callable): the runtime function
    """
    from cube.graph.parser.fx.concrete_trace_utils.concrete_tracer import is_autograd_apply

    def decorator(fn: Callable):
        if not callable(fn):
            raise TypeError("Expected a function")
        if is_autograd_apply(fn):
            fsig = CustomizedOps.create_pure_signature(f'{fn.__self__.__module__}.{fn.__self__.__name__}.apply', True)
            op_name = name if name is not None else fn.__name__
            args = inspect.signature(fn.__self__.forward)
            arg_names = list(args.parameters.keys())[1:]
        else:
            fsig = fn.__name__
            op_name = name if name is not None else fsig
            args = inspect.signature(fn)
            arg_names = list(args.parameters.keys())
        # get argument types
        arg_kinds = input_type_annos if input_type_annos is not None else \
            [args.parameters[name].annotation for name in arg_names]
        assert len(arg_kinds) == len(arg_names), \
            "Number of annotations should match with number of arguments"
        # parse for number of inputs and kwargs
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
        if code_impl_pattern == 'import':
            if is_autograd_apply(fn):
                import_path = inspect.getmodule(fn.__self__).__name__
                if import_path == '__main__':
                    _logger.warning(f'Find the function {fsig} is defined in __main__ module, will take the source code directly. '
                                f'This may cause error when the function has inner functions from other modules. '
                                f'To solve this, define the function in another module and import into main', stacklevel=0)
                    code = inspect.getsource(fn.__self__)
                    code = code[code.index(f'class {fn.__self__.__name__}'):] + f'\n{fsig}={fn.__self__.__name__}.apply'
                else:
                    code = f'from {import_path} import {fn.__self__.__name__}\n{fsig}={fn.__self__.__name__}.apply'
            else:
                import_path = inspect.getmodule(fn).__name__
                if import_path == '__main__':
                    _logger.warning(f'Find the function {fsig} is defined in __main__ module, will take the source code directly. '
                                f'This may cause error when the function has inner functions from other modules. '
                                f'To solve this, define the function in another module and import into main', stacklevel=0)
                    code = inspect.getsource(fn)
                    code = code[code.index('def'):]
                else:
                    code = f'from {import_path} import {fsig}'
        elif code_impl_pattern == 'source':
            assert not is_autograd_apply(fn), 'Only support code_impl_pattern="import" for autograd.Function.apply.'
            code = inspect.getsource(fn)
            code = code[code.index('def'):]
        else:
            raise ValueError(f'code_impl_pattern should be either "import" or "source", got {code_impl_pattern}')

        def udfop(*args, signature=None, **kwargs):
            manno = OpAnno(anno)
            tensors = args[:ninputs]
            for idx in range(ninputs):
                if arg_kinds[idx] == Optional[torch.Tensor] and tensors[idx] is None:
                    manno.set_input(idx, '?')
            kwarg_vals = args[ninputs:]
            for name, val in zip(kwarg_names, kwarg_vals):
                kwargs[name] = val
            return IRDimops(udfop, op_name, signature, [repr(manno)], tensors, transform_rules=rules, **kwargs)

        _logger.info(f'registering op {fsig} with {ninputs} inputs and {nkwargs} kwargs...')
        if is_autograd_apply(fn):
            CustomizedOps.register(f'{fn.__self__.__module__}.{fn.__self__.__name__}.apply', udfop, code, fn, True, False)
        else:
            CustomizedOps.register(fsig, udfop, code, fn)
        return fn

    return decorator
