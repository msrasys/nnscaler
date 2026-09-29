#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Register cutomized function
"""

from typing import Dict, Callable, Optional, Union, List, Tuple
from functools import partial
import inspect
import logging

import torch
from torch import ScriptFunction

from nnscaler.graph.function.dimops import IRDimops, OpAnno, TransformRule
from nnscaler.graph.tracer.wrap_utils import is_autograd_apply, is_autograd_op
from nnscaler.ir.operator import IRTensor, IRFwOperation

_logger = logging.getLogger(__name__)


class CustomizedOps:
    """Customized op registry."""

    # signature -> IRDimop creation function
    kOpMap: Dict[str, Callable] = {}
    # signature -> runtime function
    kOpRuntime: Dict[str, Callable] = {}
    # signature -> fake runtime function
    # TODO: autograd function cannot have fake runtime function now
    kOpFakeRuntime: Dict[str, Optional[Callable]] = {}
    # signature -> runtime function implementation code
    kOpCodeDef: Dict[str, str] = {}
    # signature -> special emit function, will not store if emit_fn is None
    # It accepts the node, repred args, repred kwargs, runtime_devid, plan_ndevs, runtime_ndevs
    # as input and returns the generated code.
    kOpEmit: Dict[str, Callable[[IRFwOperation, List[str], Dict[str, str], int, int, int], str]] = {}
    # signature -> input generator function
    # It accepts the IRFwOperation as input and returns the list of input tensors, which is used
    # during operator profiling.
    kOpInputGen: Dict[str, Callable[[IRFwOperation], List[torch.Tensor]]] = {}
    # runtime function -> signature
    kOpSigs: dict[Callable, str] = {}

    @staticmethod
    def map(signature: str) -> Callable:
        """Get IRDimop creation function by signature

        Args:
            signature (str): operator signature

        Returns:
            Callable: IRDimop creation function
        """
        if signature not in CustomizedOps.kOpMap:
            raise KeyError(f"{signature} is not found in registered ops")
        return partial(CustomizedOps.kOpMap[signature], signature=signature)

    @staticmethod
    def exist(signature: str) -> bool:
        """Check if the signature is registered"""
        return signature in CustomizedOps.kOpMap

    @staticmethod
    def get_fake_runtime(runtime_fn: Callable) -> Optional[Callable]:
        """Get the fake runtime registered for an exact runtime callable."""
        sig = CustomizedOps.kOpSigs.get(runtime_fn)
        if sig is not None:
            return CustomizedOps.kOpFakeRuntime.get(sig)
        return None

    @staticmethod
    def register(
        signature: str, op_create_fn: Callable, code: str, runtime_fn: Callable,
        *,
        emit_fn: Callable[[IRFwOperation, List[str], Dict[str, str], int, int, int], str] = None,
        input_gen_fn: Callable[[IRFwOperation], List[torch.Tensor]] = None,
        fake_fn: Optional[Callable] = None
    ) -> None:
        """Register an operator

        Args:
            signature (str): operator signature
            op_create_fn (Callable): IRDimops creation function
            code (str): runtime function implementation code
            runtime_fn (Callable): runtime function
            emit_fn (Callable): special emit function for codegen, will use default emit function if emit_fn is None.
                                It accepts the node, repred args, repred kwargs, runtime_devid, plan_ndevs, runtime_ndevs
                                as input and returns the generated code.
            input_gen_fn (Callable): input generator function for profiler, will use default input generator function
                                     if input_gen_fn is None. kwargs are same as that in the input node.
            fake_fn (Callable): a lightweight substitute for runtime_fn during tracing.
                It must have the same signature (inputs and outputs) as runtime_fn so that
                the two are interchangeable.
                If fake_fn is None, runtime_fn will be used,
                which may cause errors if runtime_fn contains operations
                that cannot run during tracing (e.g., distributed communication ops).
        Returns:
            None
        """
        builtins = ['_operator.', 'torch.', 'nnscaler.runtime.function.']
        if any(signature.startswith(builtin) for builtin in builtins):
            raise RuntimeError(f"Cannot register operators with signature starting from any of {builtins}")
        assert signature not in CustomizedOps.kOpMap, f"function {signature} is already registered"
        CustomizedOps.kOpMap[signature] = op_create_fn
        CustomizedOps.kOpRuntime[signature] = runtime_fn
        CustomizedOps.kOpSigs[runtime_fn] = signature
        CustomizedOps.kOpFakeRuntime[signature] = fake_fn
        CustomizedOps.kOpCodeDef[signature] = code
        if emit_fn is not None:
            CustomizedOps.kOpEmit[signature] = emit_fn
        if input_gen_fn is not None:
            CustomizedOps.kOpInputGen[signature] = input_gen_fn


def register_op(annotation: Union[str, Callable], name: Optional[str] = None,
                code_impl_pattern: str = 'import',
                *,
                emit_fn: Callable[[IRFwOperation, List[str], Dict[str, str], int, int, int], str] = None,
                transform_rules: Tuple[TransformRule] = None,
                input_gen_fn: Callable[[IRFwOperation], List[torch.Tensor]] = None,
                fake_fn: Optional[Callable] = None,
    ) -> Callable:
    """
    Register a function with IRDimops annotations.

    This function is cooperated with IRDimops. Users can only register functions defined under a module, instead of
    ones defined inside a function / class or __main__ scope.

    The annotation (`annotation`) specifies the number of inputs as *args,
    and treat all the rest inputs as **kwargs.

    For tensor-type inputs, the annotation should be a string of identifiers separated by space, e.g., `'a b'`;
    For non-tensor-type inputs, the annotation should be specified '?'.

    Examples:

    ```python
    import nnscaler
    from third_party import func

    nnscaler.graph.parser.register('a (b c) -> (a b) c')(func)
    ```

    or,

    ```python
    import nnscaler
    from third_party import func

    @nnscaler.graph.parser.register('a (b c) -> (a b) c')
    def func(x, b = 4):
        xxx
    ```

    or,

    ```python
    import nnscaler
    from third_party import func

    def anno_fn(*inputs, **kwargs):
        return 'a (b c) -> (a b) c'

    nnscaler.graph.parser.register(anno_fn)(func)
    ```

    Args:
        annotation (str | Callable): operator annotation of IRDimops or callable function that generates IRFwOperation.
            - op annotation: e.g., 'a (b c) -> (a b) c'
            - a callable function that generates op annotation (str). The function
            taks inputs and kwargs as arguments and returns the operator annotation.
        name (str | None): operator name. Only usable when node_repr is a string.
        code_impl_pattern (str):
            can only be 'import' or 'source'. If 'import', will generate code with
            import statement. If 'source', will take the source code directly.
            Default: 'import'.
        emit_fn (Callable): special emit function for codegen, it accepts the node, repred args, repred kwargs, runtime_devid,
            plan_ndevs, runtime_ndevs as input and returns the generated code. Check examples/customized_ops/ring_attention/zigzag_attn.py
            for more details.
            Default: None.
        transform_rules (Tuple[TransformRule]): a tuple of special TransformRules which will be used when partitioning the node.
            Default: None.
        input_gen_fn (Callable): input generator function for profiler, this function accepts the IRFwOperation as input and returns
            the list of input tensors, which is used during operator profiling. kwargs are same as that in the input node. By default, the
            profiler will use `torch.rand` for floating point data types and `torch.zeros` for special types like `torch.int64` and `torch.bool`.
            However, input tensors' contents may influence the speed dramatically. The mask in attention and dispatched expert index in MoE
            are real examples. To handle this scenario, user can provide the customized `input_gen_fn`.
            Default: None.
        fake_fn (Callable): a lightweight substitute for runtime_fn during tracing.
            It must have the same signature (inputs and outputs) as runtime_fn so that
            the two are interchangeable.
            If fake_fn is None, runtime_fn will be used directly,
            which may cause errors if runtime_fn contains operations
            that cannot run during tracing (e.g., distributed communication ops).
            Default: None.

    Returns:
        fn (Callable): the runtime function
    """

    def decorator(fn: Callable):
        nonlocal code_impl_pattern

        if not callable(fn):
            raise TypeError("Expected a runtime function")

        if fake_fn is not None and not callable(fake_fn):
            raise TypeError("Expected a fake function")

        # TODO: add support for autograd function in the future
        if fake_fn is not None and is_autograd_op(fn):
            raise ValueError("Autograd function cannot have fake runtime function. "
                             "Please wrap the autograd function and register the wrapper function instead.")

        if inspect.isclass(fn) and is_autograd_op(fn):
            _ = decorator(fn.apply)  # register `apply` method of the autograd function
            return fn  # return the class itself

        # step 1. get function signature and inputs
        def get_import_path(fn: Callable) -> str:
            if is_autograd_apply(fn):
                import_path = inspect.getmodule(fn.__self__).__name__
            elif isinstance(fn, ScriptFunction):
                # fn._torchdynamo_inline is the original function
                import_path = inspect.getmodule(fn._torchdynamo_inline).__name__
            else:
                import_path = inspect.getmodule(fn).__name__
            return import_path

        import_path = get_import_path(fn)
        if import_path == '__main__':
            raise NotImplementedError(
                f"Cannot register function {fn} in __main__ module. "
                f"Try to define it in another module and import into main")

        if is_autograd_apply(fn):
            fsig = f'{import_path}.{fn.__self__.__name__}.apply'
            op_name = name if name is not None else fn.__self__.__name__
            args = inspect.signature(fn.__self__.forward)
            arg_names = list(args.parameters.keys())[1:]
        elif isinstance(fn, ScriptFunction):
            # fn._torchdynamo_inline is the original function
            fsig = f'{import_path}.{fn._torchdynamo_inline.__name__}'
            op_name = name if name is not None else fn.name
            args = inspect.signature(fn._torchdynamo_inline)
            arg_names = list(args.parameters.keys())
        else:
            fsig = f'{import_path}.{fn.__name__}'
            op_name = name if name is not None else fn.__name__
            args = inspect.signature(fn)
            arg_names = list(args.parameters.keys())

        # step 2. get customized op code
        def get_source_code(fn: Callable) -> str:
            if is_autograd_apply(fn):
                code = inspect.getsource(fn.__self__)
                code = code[code.index(f'class {fn.__self__.__name__}'):]
            elif isinstance(fn, ScriptFunction):
                raise NotImplementedError('Do not support get source code for ScriptFunction.')
            else:
                code = inspect.getsource(fn)
                code = code[code.index('def'):]
            return code

        def get_import_code(fn: Callable) -> str:
            import_path = get_import_path(fn)
            code = f'import {import_path}'
            return code

        if code_impl_pattern == 'import':
            code = get_import_code(fn)
        elif code_impl_pattern == 'source':
            code = get_source_code(fn)
        else:
            raise ValueError(f'code_impl_pattern should be either "import" or "source", got {code_impl_pattern}')

        # step 3. define customized IRDimops creation function
        if not (isinstance(annotation, str) or callable(annotation)):
            raise TypeError(f"annotation should be either str or callable, got {type(annotation)}")

        def udfop(*args, signature=None, **kwargs):
            anno = annotation if isinstance(annotation, str) else annotation(*args, **kwargs)
            if not isinstance(anno, str):
                raise TypeError(f"node_repr should return a string, but got {type(anno)}: {anno}")
            anno = OpAnno(anno)
            ninputs = len(anno.inputs())
            if len(args) < ninputs:
                # try to fill args with kwargs
                args = list(args)
                kwargs = dict(kwargs)
                for idx in range(len(args), ninputs):
                    if arg_names[idx] in kwargs:
                        args.append(kwargs.pop(arg_names[idx]))
                    else:
                        raise ValueError(f"calling function {signature} should include at least {ninputs} *args")
            tensors = args[:ninputs]
            for idx, t in enumerate(tensors):
                # argument check
                if not anno.input(idx).ignore:
                    if not isinstance(t, IRTensor):
                        raise ValueError(
                            f"{idx}-th input needs IRTensor, but got {type(t)}: {t}\n"
                            f"signature: {signature}\n"
                            f"annotation: {anno}")
            kwarg_names = [name for name in arg_names[ninputs:]]
            kwarg_vals = args[ninputs:]
            for name, val in zip(kwarg_names, kwarg_vals):
                kwargs[name] = val
            return IRDimops(udfop, op_name, signature, [repr(anno)], tensors, **kwargs, transform_rules=transform_rules)

        # step 4. register in CustomizedOps
        _logger.info(f'registering op {fsig}...')
        CustomizedOps.register(
            fsig, udfop, code, fn,
            emit_fn=emit_fn,
            input_gen_fn=input_gen_fn,
            fake_fn=fake_fn
        )
        return fn

    return decorator


# [Deprecated] register_op alias
# Will remove in future.
register = register_op
no_change = object()


def update_op(
    runtime_fn: Callable,
    *,
    op_create_fn: Optional[Callable] = no_change,
    code: Optional[str] = no_change,
    emit_fn: Callable[[IRFwOperation, List[str], Dict[str, str], int, int, int], str] = no_change,
    input_gen_fn: Callable[[IRFwOperation], List[torch.Tensor]] = no_change,
    fake_fn: Optional[Callable] = no_change,
) -> Callable:
    """Update optional metadata for an existing operator.

    Unspecified metadata keeps using the existing customized or system-defined
    value.
    """
    if not callable(runtime_fn):
        raise TypeError("Expected a runtime function")
    if fake_fn is not no_change and fake_fn is not None and not callable(fake_fn):
        raise TypeError("Expected a fake function")
    if op_create_fn is not no_change and op_create_fn is not None and not callable(op_create_fn):
        raise TypeError("Expected an op creation function")
    if code is not no_change and code is not None and not isinstance(code, str):
        raise TypeError("Expected code to be a string")
    if emit_fn is not no_change and emit_fn is not None and not callable(emit_fn):
        raise TypeError("Expected an emit function")
    if input_gen_fn is not no_change and input_gen_fn is not None and not callable(input_gen_fn):
        raise TypeError("Expected an input generator function")
    if fake_fn is not no_change and fake_fn is not None and is_autograd_op(runtime_fn):
        raise ValueError("Autograd function cannot have fake runtime function. "
                         "Please wrap the autograd function and register the wrapper function instead.")

    from nnscaler.graph.parser.mapping import SignFx2Op

    signature = SignFx2Op.rmap(runtime_fn)
    if signature is None:
        raise ValueError(f"{runtime_fn} is not an existing operator")

    system_op_create_fn = None
    if signature in SignFx2Op.kOpMap:
        system_op_create_fn = partial(SignFx2Op.kOpMap[signature], signature=signature)

    # Materialize system defaults so an updated built-in has the same complete
    # CustomizedOps entry as an operator registered through register_op.
    if signature not in CustomizedOps.kOpMap:
        CustomizedOps.kOpMap[signature] = system_op_create_fn
        CustomizedOps.kOpCodeDef[signature] = ''
        CustomizedOps.kOpFakeRuntime[signature] = None

    CustomizedOps.kOpRuntime[signature] = runtime_fn

    if op_create_fn is not no_change:
        if op_create_fn is None:
            if system_op_create_fn is None:
                raise ValueError(f"{runtime_fn} has no system-defined op creation function")
            CustomizedOps.kOpMap[signature] = system_op_create_fn
        else:
            CustomizedOps.kOpMap[signature] = op_create_fn

    if code is not no_change:
        CustomizedOps.kOpCodeDef[signature] = code or ''
    if fake_fn is not no_change:
        CustomizedOps.kOpFakeRuntime[signature] = fake_fn

    if emit_fn is not no_change:
        if emit_fn is None:
            CustomizedOps.kOpEmit.pop(signature, None)
        else:
            CustomizedOps.kOpEmit[signature] = emit_fn
    if input_gen_fn is not no_change:
        if input_gen_fn is None:
            CustomizedOps.kOpInputGen.pop(signature, None)
        else:
            CustomizedOps.kOpInputGen[signature] = input_gen_fn

    return runtime_fn
