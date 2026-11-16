#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import builtins
import time
from typing import TYPE_CHECKING, Any, Tuple, Dict, Callable, Type

import torch
from torch.fx.node import Target
from torch.overrides import is_tensor_method_or_property

from . import pytree_utils, metadata, wrap_utils
from .metadata import EmptyResult
from .operator_patcher import OperatorPatcherContext

if TYPE_CHECKING:
    from .concrete_tracer import ConcreteTracer

import logging
_logger = logging.getLogger(__name__)


# Lower-cased fragments of the PyTorch errors raised when an operator needs tensor data
# (or a meta kernel) that meta tensors do not have, e.g.
#   "Tensor.item() cannot be called on meta tensors"            (.item(), bool(), int(), `in`)
#   "Cannot copy out of meta tensor; no data!"                   (.tolist(), .cpu(), .numpy() paths)
#   "can't convert meta device type tensor to numpy"
#   "... raises unimplemented by default, as a correct data-independent implementation does not exist" (nonzero, masks)
#   "aten::_unique2: attempted to run this operator with Meta tensors, but there was no fake impl or Meta kernel"
#   "Could not run 'aten::...' with arguments from the 'Meta' backend"
_META_DATA_DEPENDENT_FRAGMENTS = ('meta tensor', 'meta device', 'data-independent implementation', "'meta' backend")


def is_meta_data_dependency(exc: BaseException) -> bool:
    """Whether ``exc`` means an operator needs tensor data (or a meta kernel) that meta tensors do not carry."""
    if not isinstance(exc, (RuntimeError, NotImplementedError, TypeError)):
        return False
    message = str(exc).lower()
    return any(fragment in message for fragment in _META_DATA_DEPENDENT_FRAGMENTS)


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    # `__name__` over `__qualname__`: torch functions report `_VariableFunctionsClass.nonzero` otherwise
    return getattr(target, '__name__', None) or getattr(target, '__qualname__', None) or repr(target)


def meta_guidance(name: str) -> str:
    return (
        f"trace_strategy='meta' cannot execute `{name}`: it reads tensor values (or has no meta kernel), "
        "and meta tensors carry no data. nnScaler traces a static graph, so either trace with a strategy that "
        "runs on real data ('cpu', 'cuda', 'cuda_run_cpu_offload'), or give the operator a stand-in result: "
        "nnscaler.trace_fake(fn, fake, *args, **kwargs) at the call site, nnscaler.update_op(fn, fake_fn=...) "
        "for a torch function, or nnscaler.register_op(..., fake_fn=...) for the enclosing user function."
    )


def has_meta_tensors(module: torch.nn.Module) -> bool:
    """
    Whether ``module`` holds meta parameters or buffers, i.e. it was built on meta or captured by
    DeferredInitialization. Tensors created outside the capture (module-level constants, tensors passed
    to the constructor) stay real, so a captured module can mix both; it is still traced in place.
    """
    return any(t.is_meta for t in list(module.parameters()) + list(module.buffers()))


def place_real_tensors_to_cpu(module: torch.nn.Module) -> torch.nn.Module:
    """``module.cpu()`` that leaves meta tensors alone (they have no data to move)."""
    return module._apply(lambda t: t if t.is_meta else t.cpu())


class BaseTraceStrategy:
    """
    The base class for trace strategy, which executes the function target with concrete arguments and return the result.

    There are six kinds of node in a fx.Graph:

    - placeholder:
      `placeholder` means this node has no parent. A placeholder node usually means it is the input of the traced function.
      The target of placeholder node is the name of the object, for example, the input argument name.

    - get_attr:
      `get_attr` specifically refers to obtaining attributes from root module. The target is the name path of the attribute.
      For example, 'layer1.weight', it means get root.layer1.weight.

    - call_function:
      The target of `call_function` is a callable function, executed by target(*args, **kwargs).

    - call_method:
      The target of `call_method` is a string of the bound method name, the method is bound to the first element of `args`.
      Executed by getattr(args[0], target)(*args[1:], **kwargs).

    - call_module:
      The target of `call_module` is a string which means the sub-module path of the root module. For example, 'layer1.linear'.
      Executed by fetch_attr(root, target)(*args, **kwargs).

    - output: the output node of the graph.
      The target of `output` node is not matter, usually is string 'output'. Only used to identify the output of the graph.
      So here we assume the `args` is an one element tuple and `kwargs` is empty, we directly take the first argument as output result.
    """

    # identify the name of the strategy
    _name: str

    def __init__(self, tracer: 'ConcreteTracer', main_device: str) -> None:
        self.tracer = tracer
        self.main_device = main_device

    @property
    def name(self):
        return self._name

    @staticmethod
    def _place_module_to(module: torch.nn.Module, device: str) -> torch.nn.Module:
        if device == 'cpu':
            module.cpu()
        elif device == 'cuda':
            module.cuda()
        elif device == 'meta':
            # NOTE: this device move is not recoverable, the data will lose
            module.to_empty(device='meta')
        else:
            raise ValueError(f'unsupported device type: {device}')
        return module

    @staticmethod
    def _place_tensors_to(*args, device: str) -> Tuple[Any]:
        # In most cases, device placement operation keeps the source tensor's `requires_grad` attributes.
        # The context `torch.no_grad` enforces requires_grad=False for all tensors that generated in its scope.
        # As a result, behavior of device placement operation is unexpected.
        # To handle this case, we need manually set the `requires_grad` field after device placement operation.
        if device not in ['cpu', 'cuda', 'meta']:
            raise ValueError(f'unsupported device type: {device}')
        return pytree_utils.tree_map_only(torch.Tensor, lambda x: x.to(device).requires_grad_(x.requires_grad), args)

    # Tensor methods that are the method form of a device argument (`x.cuda()`); see `_map_devices_to`.
    _DEVICE_METHODS = ('cuda',)

    @classmethod
    def _is_mapped_device(cls, value: Any) -> bool:
        """
        Whether a value in a device position must be replaced by the execution device.
        The value is a `torch.device`, a string, or an int, which is a cuda index there (`device=0`, `x.to(0)`).

        Only cuda is replaced: it stands for "the runtime device".
        An explicit 'cpu' is the user's choice, kept in the generated code and so also honored during the trace.
        """
        if isinstance(value, bool):
            # a bool is an int, but `x.to(False)` is `x.to(torch.bool)`, not a device index
            return False
        if isinstance(value, int):
            return True
        if isinstance(value, str):
            value = torch.device(value)
        return isinstance(value, torch.device) and value.type == 'cuda'

    @classmethod
    def _map_devices_to(cls, args: Tuple[Any, ...], kwargs: Dict[str, Any], device: str,
                        first_arg_device: bool = False) -> Tuple[Tuple, Dict]:
        # `ConcreteAttrProxy` pins `<tensor>.device` to cuda so the generated code targets GPUs,
        # and user code may also hard-code `device='cuda'`.
        # Executing with that device would allocate on a real GPU, and fail on a machine without one,
        # even though the strategy runs everything on `device`.
        # Only the copy that is executed is mapped; the original args are what gets recorded in the graph,
        # so the device constants stay as written there.
        # A `torch.device` object is a device wherever it appears.
        # A string or int is one only where its position says so:
        # the `device=` keyword, and the first positional argument of `Tensor.to` (`first_arg_device`);
        # anywhere else it is data (`d['cuda']`, `getattr(x, 'cuda')`, `x.view(0)`).
        # Rebuilding the containers would break in-place container ops (e.g. `list.append` on the executed copy),
        # so they are returned untouched unless they hold a mapped device.
        def to_device(x):
            if not cls._is_mapped_device(x):
                return x
            return torch.device(device) if isinstance(x, torch.device) else device

        if any(isinstance(leaf, torch.device) and cls._is_mapped_device(leaf)
               for leaf in pytree_utils.tree_flatten((args, kwargs))[0]):
            args, kwargs = pytree_utils.tree_map_only(torch.device, to_device, (args, kwargs))
        if 'device' in kwargs and cls._is_mapped_device(kwargs['device']):
            kwargs = {**kwargs, 'device': device}
        if first_arg_device and args and cls._is_mapped_device(args[0]):
            args = (device, *args[1:])
        return args, kwargs

    @classmethod
    def _call_function(cls, target: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any], device: str) -> Any:
        """Call ``target(*args, **kwargs)`` for execution on ``device``."""
        args, kwargs = cls._map_devices_to(args, kwargs, device)
        return target(*args, **kwargs)

    @classmethod
    def _call_method(cls, self_obj: Any, target: str, args: Tuple[Any, ...], kwargs: Dict[str, Any], device: str) -> Any:
        """Call ``self_obj.target(*args, **kwargs)`` for execution on ``device``."""
        is_tensor = isinstance(self_obj, torch.Tensor)
        if is_tensor and target in cls._DEVICE_METHODS:
            # `.cuda()` is the method form of device='cuda' (see `_map_devices_to`).
            # The graph keeps the call, but executing it would allocate on a real GPU,
            # so it moves to the execution device instead.
            kwargs = {key: value for key, value in kwargs.items() if key in ('non_blocking', 'memory_format')}
            return self_obj.to(device, **kwargs)

        args, kwargs = cls._map_devices_to(args, kwargs, device, first_arg_device=is_tensor and target == 'to')
        return getattr(self_obj, target)(*args, **kwargs)

    def run_placeholder(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        return self.tracer.placeholder_dict[target], args, kwargs

    def run_get_attr(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        assert isinstance(target, str)
        return self.tracer.fetch_attr(target), args, kwargs

    def run_output(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        return args[0], args, kwargs

    def _run_call_function_on(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any], device: str) -> Tuple[Any, Tuple, Dict]:
        if not isinstance(target, Callable):
            raise ValueError(f'the target of "call_function" should be a callable function, but get target {target}')
        args, kwargs = self._place_tensors_to(args, kwargs, device=device)
        return self._call_function(target, args, kwargs, device), args, kwargs

    def _run_call_method_on(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any], device: str) -> Tuple[Any, Tuple, Dict]:
        if not isinstance(target, str):
            raise ValueError(f'the target of "call_method" should be a string, a bound method name of the first argument, but get target {target}')
        args, kwargs = self._place_tensors_to(args, kwargs, device=device)
        self_obj, *args_tail = args
        return self._call_method(self_obj, target, tuple(args_tail), kwargs, device), args, kwargs

    def _run_call_module_on(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any], device: str) -> Tuple[Any, Tuple, Dict]:
        if not isinstance(target, str):
            raise ValueError(f'the target of "call_module" should be a string, a name of the nn module, but get target {target}')
        args, kwargs = self._place_tensors_to(args, kwargs, device=device)
        return self._call_function(self.tracer.fetch_attr(target), args, kwargs, device), args, kwargs

    def run_call_function(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        return self._run_call_function_on(target, args, kwargs, device=self.main_device)

    def run_call_method(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        return self._run_call_method_on(target, args, kwargs, device=self.main_device)

    def run_call_module(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        return self._run_call_module_on(target, args, kwargs, device=self.main_device)

    def _run_target(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        start_time = time.time()

        if kind == 'placeholder':
            result = self.run_placeholder(target, args, kwargs)
        elif kind == 'get_attr':
            result = self.run_get_attr(target, args, kwargs)
        elif kind == 'call_function':
            result = self.run_call_function(target, args, kwargs)
        elif kind == 'call_method':
            result = self.run_call_method(target, args, kwargs)
        elif kind == 'call_module':
            result = self.run_call_module(target, args, kwargs)
        elif kind == 'output':
            result = self.run_output(target, args, kwargs)
        else:
            raise RuntimeError(f'unexpected kind {kind}')

        cost = time.time() - start_time
        if cost > 0.05:
            cost_msg = f'Run time cost -- [{kind}][{target.__module__ if callable(target) else ""}][{str(target) if not callable(target) else getattr(target, "__qualname__", getattr(target, "__name__"))}]: {cost}s'
            _logger.debug(cost_msg)
        return result

    def place_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Place the model to the preference device.
        """
        return self._place_module_to(model,  device=self.main_device)

    def place_inputs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple, Dict]:
        """
        Place the tensor in the inputs to the preference device.
        """
        return self._place_tensors_to(args, kwargs, device=self.main_device)

    def run_fake(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], fake: Any) -> Tuple[Any, Tuple, Dict]:
        """
        The result of a `trace_fake` call: ``fake`` placed where this strategy keeps results,
        with the inputs placed as for an executed operator. The operator itself does not run.
        """
        args, kwargs = self._place_tensors_to(args, kwargs, device=self.main_device)
        value, = self._place_tensors_to(fake, device=self.main_device)
        return value, args, kwargs

    def run_target(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                   fake: Any = EmptyResult) -> Tuple[Any, Tuple, Dict]:
        """
        Concrete execute the target and return the result.

        Args:
            kind (str) : one of "placeholder", "call_function", "call_method", "call_module", "get_attr", "output".
            fake: the result given to `trace_fake`; when present it is used instead of executing the target,
                like the ``fake_fn`` of a registered operator.
        """
        if fake is not EmptyResult:
            return self.run_fake(args, kwargs, fake)
        return self._run_target(kind, target, args, kwargs)


class CpuStrategy(BaseTraceStrategy):
    """
    Pure cpu strategy, model is placed on cpu, run target is on cpu, intermediate results are on cpu.
    """
    _name = 'cpu'

    def __init__(self, tracer: 'ConcreteTracer'):
        super().__init__(tracer, 'cpu')


class CudaStrategy(BaseTraceStrategy):
    """
    Pure cuda strategy, model is placed on cuda, run target is on cuda, intermediate results are on cuda.
    """
    _name = 'cuda'

    def __init__(self, tracer: 'ConcreteTracer'):
        super().__init__(tracer, 'cuda')


class MetaStrategy(BaseTraceStrategy):
    """
    Meta strategy, run target is on meta, intermediate results are on meta.
    A model that is already entirely on meta is traced there; any other model is placed on cpu.

    Operators that read tensor values (``.item()``, ``nonzero``, branch conditions, ...) cannot run on meta
    and raise an error naming the operator; `trace_fake` or the ``fake_fn`` of a registered operator gives
    such a call a stand-in result.
    """
    _name = 'meta'
    # On meta there is nothing to move and nothing can be copied out of it (`x.cpu()` raises),
    # so every device argument executes as the strategy device, cuda or not;
    # the graph still records what the user wrote.
    _DEVICE_METHODS = ('cuda', 'cpu')

    @classmethod
    def _is_mapped_device(cls, value: Any) -> bool:
        return isinstance(value, (int, str, torch.device)) and not isinstance(value, bool)

    def __init__(self, tracer: 'ConcreteTracer'):
        super().__init__(tracer, 'meta')

    def place_model(self, model: torch.nn.Module) -> torch.nn.Module:
        # Meta tensors stay on meta: they come from a model built on meta or captured by DeferredInitialization,
        # and shard initialization re-creates the model at runtime, so nothing is lost.
        # Real tensors are kept on cpu rather than moved to meta because their values are still needed:
        # a cpu instance passed to `parallelize` is the source the runtime copies the shard values from.
        # Execution does not need them on meta: `_place_tensors_to` makes a meta copy per operator.
        return place_real_tensors_to_cpu(model)

    def run_target(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                   fake: Any = EmptyResult) -> Tuple[Any, Tuple, Dict]:
        if fake is not EmptyResult:
            return super().run_target(kind, target, args, kwargs, fake)
        # Meta kernels are Python code (torch._meta_registrations, torch._decomp, torch._prims_common)
        # that compares dtypes against the builtin `float`/`int`/`bool` and asserts via `torch._check`.
        # With the tracer's builtins wrappers installed they fail even for `a + b`
        # ("Expected builtin numeric types, found clz_wrapper_clz").
        # CPU/CUDA kernels run in C++ and never see the wrappers.
        # Only builtins are restored: a full `patcher.revert()` would undo and re-apply every torch wrapper
        # for every node.
        try:
            with self.tracer.patcher.revert_namespace(builtins):
                return super().run_target(kind, target, args, kwargs)
        except (RuntimeError, NotImplementedError, TypeError) as exc:
            if not is_meta_data_dependency(exc):
                raise
            raise RuntimeError(meta_guidance(_target_name(target))) from exc


class CudaRunCpuOffloadStrategy(BaseTraceStrategy):
    """
    This is the previous tracer run target logic (nnscaler <= v0.2).

    Model is placed on cpu, run target is on cuda, intermediate results are on cpu.
    If detect OOM during run target, will retry run target on cpu.
    """
    _name = 'cuda_run_cpu_offload'

    def __init__(self, tracer: 'ConcreteTracer'):
        super().__init__(tracer, 'cpu')

    def run_call_function(self, target: Target, args: Tuple[Any], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        try:
            result, args_cuda, kwargs_cuda = self._run_call_function_on(target, args, kwargs, device='cuda')
            result, args_cpu, kwargs_cpu = self._place_tensors_to(result, args_cuda, kwargs_cuda, device='cpu')
            return result, args_cpu, kwargs_cpu
        except torch.cuda.OutOfMemoryError as e:
            _logger.info(f'tracing {target} on cuda failed, try to trace on cpu, error message is: {str(e)}')
            return self._run_call_function_on(target, args, kwargs, 'cpu')

    def run_call_method(self, target: Target, args: Tuple[Any], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        try:
            result, args_cuda, kwargs_cuda = self._run_call_method_on(target, args, kwargs, device='cuda')
            result, args_cpu, kwargs_cpu = self._place_tensors_to(result, args_cuda, kwargs_cuda, device='cpu')
            return result, args_cpu, kwargs_cpu
        except torch.cuda.OutOfMemoryError as e:
            _logger.info(f'tracing {target} on cuda failed, try to trace on cpu, error message is: {str(e)}')
            return self._run_call_method_on(target, args, kwargs, device='cpu')

    def run_call_module(self, target: Target, args: Tuple[Any], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        if not isinstance(target, str):
            raise ValueError(f'the target of "call_module" should be a string, a name of the nn module, but get target {target}')
        mod: torch.nn.Module = self.tracer.fetch_attr(target)
        try:
            mod.cuda()
            result, args_cuda, kwargs_cuda = self._run_call_module_on(target, args, kwargs, device='cuda')
            mod.cpu()
            result, args_cpu, kwargs_cpu = self._place_tensors_to(result, args_cuda, kwargs_cuda, device='cpu')
            return result, args_cpu, kwargs_cpu
        except torch.cuda.OutOfMemoryError as e:
            _logger.info(f'tracing {target} on cuda failed, try to trace on cpu, error message is: {str(e)}')
            mod.cpu()
            return self._run_call_module_on(target, args, kwargs, device='cpu')


class ReuseCacheStrategy(CudaRunCpuOffloadStrategy):
    """
    In this strategy, the result of a node will be cached, and next time the same op with same input
    (for tensor, only check if the tensor meta is the same, please view class TensorMetadata in metadata.py for more information),
    will directly return the previous cached result.

    Please note that this strategy break the data dependence and might give tensor with wrong shape as result, for example:

        x = torch.nonzero(torch.tensor([0, 1, 2]))
        y = torch.nonzero(torch.tensor([0, 0, 2]))
        # in this case, during tracing, because the function is the same, and the input tensor has same meta data,
        # then y is not calculted and directly use x as result for the second torch.nonzero call.
    """
    _name = 'reuse_cache'

    def __init__(self, tracer: 'ConcreteTracer') -> None:
        super().__init__(tracer)
        self.cache: Dict[str, Any] = {}
        self.cache_size = 0

        # some ops don't use gpu, so directly run them on cpu and don't cache them
        # TODO: add functions to optimize the cache memory cost.
        self.force_cpu_ops = []

    def force_cpu_run(self, kind, target):
        if kind == 'call_function':
            if target.__module__ == 'builtins':
                return True
            if target in self.force_cpu_ops:
                return True
        return False

    @staticmethod
    def hash_input(kind, target, args, kwargs):
        assert kind != 'call_module', 'call_module is not supported hash input'
        args, kwargs = pytree_utils.tree_map_only(torch.Tensor, metadata._extract_tensor_metadata, (args, kwargs))
        # NOTE: here torch.is_grad_enabled is used to detect if under the torch.no_grad context,
        # the tensor in the result might have different requires_grad although the operation and its inputs are the same.
        # TODO: we don't know if args and kwargs are all hashable, so here simply use str as their hash value,
        # for example, list is widly used, but it is not hashable, there is a risk if the str is not good enough to identity the different inputs can reuse the cached output,
        # should improve the implementation of the hash_input if we can find a better way to category the inputs that can reuse the output.
        return str((kind, target, args, kwargs, torch.is_grad_enabled()))

    @staticmethod
    def count_tensor_memory_cost(t: torch.Tensor):
        return t.dtype.itemsize * t.numel()

    def run_call_function(self, target: Target, args: Tuple[Any], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        if self.force_cpu_run('call_function', target):
            return self._run_call_function_on(target, args, kwargs, device='cpu')

        input_hash = self.hash_input('call_function', target, args, kwargs)
        if input_hash in self.cache:
            return self.cache[input_hash]
        else:
            result = super().run_call_function(target, args, kwargs)
            self.cache[input_hash] = result
            self.cache_size += sum([self.count_tensor_memory_cost(t) for t in pytree_utils.tree_flatten(result)[0] if isinstance(t, torch.Tensor)])
            _logger.debug(f'cache [{input_hash}], current total cache size is: {self.cache_size / 1024 / 1024 / 1024} GB')
            return result

    def run_call_method(self, target: Target, args: Tuple[Any], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        if self.force_cpu_run('call_method', target):
            return self._run_call_method_on(target, args, kwargs, device='cpu')

        input_hash = self.hash_input('call_method', target, args, kwargs)
        if input_hash in self.cache:
            return self.cache[input_hash]
        else:
            result = super().run_call_method(target, args, kwargs)
            self.cache[input_hash] = result
            self.cache_size += sum([self.count_tensor_memory_cost(t) for t in pytree_utils.tree_flatten(result) if isinstance(t, torch.Tensor)])
            _logger.debug(f'cache [{input_hash}], current total cache size is: {self.cache_size / 1024 / 1024 / 1024} GB')
            return result


    def run_call_module(self, target: Target, args: Tuple[Any], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        # TODO: also add cache for the call_module
        return super().run_call_module(target, args, kwargs)


TRACE_STRATEGY: Dict[str, Type[BaseTraceStrategy]] = {
    CpuStrategy._name: CpuStrategy,
    CudaStrategy._name: CudaStrategy,
    MetaStrategy._name: MetaStrategy,
    CudaRunCpuOffloadStrategy._name: CudaRunCpuOffloadStrategy,
    ReuseCacheStrategy._name: ReuseCacheStrategy,
}


def trace_strategy_class(name: str) -> Type[BaseTraceStrategy]:
    if name not in TRACE_STRATEGY:
        raise ValueError(f"unknown trace strategy '{name}', expected one of {sorted(TRACE_STRATEGY)}")
    return TRACE_STRATEGY[name]


def create_trace_strategy(name: str, tracer: 'ConcreteTracer') -> BaseTraceStrategy:
    return trace_strategy_class(name)(tracer)


def trace_fake(fn: Callable, fake: Any, *args, **kwargs) -> Any:
    """
    ``fn(*args, **kwargs)`` recorded in the graph, with ``fake`` as its value during the trace.

    The same contract as the ``fake_fn`` of a registered operator, for one call site:
    while tracing, ``fn`` is not executed and ``fake`` stands in for its result.
    The graph still records ``fn`` with the original arguments, so the generated code calls the real operator.
    Outside of tracing ``fn`` runs and ``fake`` is ignored.
    Use it for operators that read tensor values (``torch.nonzero``, ``.item()``, ``torch.where(mask)``),
    which cannot run under ``trace_strategy='meta'``.

    ``fake`` must have the shapes, dtypes and ``requires_grad`` of the real result and carry data:
    cpu/cuda tensors, Python values (for ``.item()``, ``.tolist()``) or tuples of them (``torch.where``).
    A callable ``fake`` is called with the same arguments as ``fn`` to produce that value;
    it sees the arguments as the trace does, so under the meta strategy it may use their shapes and dtypes
    but not their values, and it must allocate on cpu.
    Tensor methods are given unbound, e.g. ``trace_fake(torch.Tensor.item, 3, x)``;
    ``bool`` conditions as ``trace_fake(bool, True, x > 0)``.
    ``fn`` must be an operator the tracer records as a node
    (torch function, tensor method, builtin or registered operator), not a plain Python function;
    in-place operators are not supported.
    """
    # Set for the whole forward run of a trace (traces do not nest), so it is the "tracing now" signal.
    tracer = OperatorPatcherContext.ctx_tracer
    if tracer is None:
        return fn(*args, **kwargs)
    # torch functions, tensor methods and builtins are patched while tracing: record the original
    fn = wrap_utils.unwrap(fn)
    if not (fn in tracer.autowrap_leaf_function or fn in tracer.autowrap_leaf_class or is_tensor_method_or_property(fn)):
        # A plain python function would become one opaque IRPyFunc node
        # that cannot be partitioned and that the generated code must import by name;
        # the tracer would normally trace into it.
        raise ValueError(
            f"trace_fake: `{_target_name(fn)}` is not an operator the tracer records as a node (a torch function, "
            "tensor method, builtin or registered operator). Apply trace_fake to the data-dependent operator inside it.")

    with wrap_utils.do_temp_call_origin():
        from .concrete_proxy import ConcreteProxy

        def unwrap(tree):
            return pytree_utils.tree_map_only(ConcreteProxy, lambda p: unwrap(p.value), tree)

        fake = unwrap(fake)   # first: a proxy is callable, the tensor it wraps is not
        if callable(fake):
            # Computed from the arguments as the trace sees them:
            # shapes and dtypes are available, values only when the strategy runs on real data.
            plain_args, plain_kwargs = unwrap((args, kwargs))
            fake = unwrap(fake(*plain_args, **plain_kwargs))

        for leaf in pytree_utils.tree_flatten(fake)[0]:
            if isinstance(leaf, torch.Tensor) and leaf.is_meta:
                raise ValueError('trace_fake: the fake must carry data (a cpu or cuda tensor), got a meta tensor')
        # A private copy: a fake built in forward (`torch.zeros(...)`) is itself a traced value,
        # possibly with autograd history, which must not leak into the result of `fn`.
        fake = pytree_utils.tree_map_only(torch.Tensor, lambda t: t.detach().clone().requires_grad_(t.requires_grad), fake)

    if is_tensor_method_or_property(fn):
        # be consistent with ConcreteAttrProxy.__call__
        return tracer.create_proxy('call_method', fn.__name__, args, kwargs, fake_result=fake)
    return tracer.create_proxy('call_function', fn, args, kwargs, fake_result=fake)
