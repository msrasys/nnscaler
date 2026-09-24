#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Tuple, Dict, Callable, Type

import torch
from torch.fx.node import Target

from . import pytree_utils, metadata

if TYPE_CHECKING:
    from .concrete_tracer import ConcreteTracer

import logging
_logger = logging.getLogger(__name__)


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
        return target(*args, **kwargs), args, kwargs

    def _run_call_method_on(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any], device: str) -> Tuple[Any, Tuple, Dict]:
        if not isinstance(target, str):
            raise ValueError(f'the target of "call_method" should be a string, a bound method name of the first argument, but get target {target}')
        args, kwargs = self._place_tensors_to(args, kwargs, device=device)
        self_obj, *args_tail = args
        func = getattr(self_obj, target)
        return func(*args_tail, **kwargs), args, kwargs

    def _run_call_module_on(self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any], device: str) -> Tuple[Any, Tuple, Dict]:
        if not isinstance(target, str):
            raise ValueError(f'the target of "call_module" should be a string, a name of the nn module, but get target {target}')
        args, kwargs = self._place_tensors_to(args, kwargs, device=device)
        mod = self.tracer.fetch_attr(target)
        return mod(*args, **kwargs), args, kwargs

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

    def run_target(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        """
        Concrete execute the target and return the result.

        Args:
            kind (str) : one of "placeholder", "call_function", "call_method", "call_module", "get_attr", "output".
        """
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
    Meta strategy, run target is on meta, intermediate results are on meta, but note model is placed on cpu for current version.
    """
    _name = 'meta'

    def __init__(self, tracer: 'ConcreteTracer'):
        super().__init__(tracer, 'meta')

    def place_model(self, model: torch.nn.Module) -> torch.nn.Module:
        # TODO: save the original model paramenters/buffers data to somewhere, the concrate value will lose after place the model to meta
        # return self._place_module_to_meta(model)
        return self._place_module_to(model, device='cpu')


class DeferredStrategy(BaseTraceStrategy):
    """Trace metadata where possible; materialize only the next concrete op.

    Parameter recipes stay on the source module, never full parameter values.
    Pure PyTorch expressions run on meta tensors. Custom kernels or reads of
    tensor values evaluate their dependencies with temporary real parameters.
    Returned trace samples are detached to avoid retaining parameter storage
    through autograd, while preserving the observed requires_grad metadata.
    """
    _name = 'deferred'

    def __init__(self, tracer):
        super().__init__(tracer, 'cpu')
        self.expressions = {}
        self.parameters = {}
        self.cache = {}

    def place_model(self, model):
        self.plan = model._nnscaler_init_plan
        self.parameters = {id(p): (p, name) for name, p in model.named_parameters()}
        return model

    @staticmethod
    def _invoke(kind, target, args, kwargs):
        if kind == 'call_method':
            return getattr(args[0], target)(*args[1:], **kwargs)
        return target(*args, **kwargs)

    @staticmethod
    def _detach(value):
        def detach(tensor):
            return tensor.detach().requires_grad_(tensor.requires_grad)
        return pytree_utils.tree_map_only(torch.Tensor, detach, value)

    def _remember(self, value, kind, target, args, kwargs):
        leaves, spec = pytree_utils.tree_flatten(value)
        expression = (kind, target, args, kwargs)
        for index, leaf in enumerate(leaves):
            if isinstance(leaf, torch.Tensor) and leaf.is_meta:
                self.expressions[id(leaf)] = (leaf, expression, index)

    def _materialize(self, tensor, device, cache):
        if not tensor.is_meta:
            return tensor.to(device).requires_grad_(tensor.requires_grad)
        key = id(tensor)
        if key in self.parameters:
            from nnscaler.runtime.initialization import initialize
            _, name = self.parameters[key]
            out = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device=device)
            initialize(self.plan, name, (slice(None),)*tensor.ndim, out=out, _trace=True)
            out.untyped_storage()._nnscaler_trace_temporary = True
            return out.requires_grad_(tensor.requires_grad)
        if key in cache:
            return cache[key]
        if key not in self.expressions:
            raise RuntimeError('Untracked meta tensor during deferred tracing')
        _, (kind, target, args, kwargs), index = self.expressions[key]
        real_args, real_kwargs = pytree_utils.tree_map_only(
            torch.Tensor, lambda t: self._materialize(t, device, cache), (args, kwargs))
        value = self._detach(self._invoke(kind, target, real_args, real_kwargs))
        result = pytree_utils.tree_flatten(value)[0][index]
        result.untyped_storage()._nnscaler_trace_temporary = True
        cache[key] = result
        return result

    def _concrete(self, kind, target, args, kwargs):
        from nnscaler.flags import CompileFlag
        device = 'cpu' if CompileFlag.trace_strategy == 'cpu' else 'cuda'
        cache = {}
        real_args, real_kwargs = pytree_utils.tree_map_only(
            torch.Tensor, lambda t: self._materialize(t, device, cache), (args, kwargs))
        value = self._detach(self._invoke(kind, target, real_args, real_kwargs))
        with torch.no_grad():
            value, = self._place_tensors_to(value, device='cpu')
        # Keep proxies for attributes as meta tensors, so the graph does not
        # accumulate one real CPU/GPU parameter for every operator visited.
        return value, args, kwargs

    def run_target(self, kind, target, args, kwargs):
        from nnscaler.flags import CompileFlag
        # Preserve the existing shape-cache semantics selected by llm-train,
        # without caching concrete parameter arguments/storage along with it.
        cacheable = (CompileFlag.trace_strategy == 'reuse_cache'
                     and kind in ('call_function', 'call_method')
                     and getattr(target, '__module__', '') != 'builtins')
        key = ReuseCacheStrategy.hash_input(kind, target, args, kwargs) if cacheable else None
        if cacheable and key in self.cache:
            return self.cache[key], args, kwargs
        result = self._run_deferred(kind, target, args, kwargs)
        if cacheable:
            self.cache[key] = result[0]
        return result

    def _run_deferred(self, kind, target, args, kwargs):
        if kind not in ('call_function', 'call_method'):
            return super().run_target(kind, target, args, kwargs)
        leaves = pytree_utils.tree_flatten((args, kwargs))[0]
        has_meta = any(isinstance(t, torch.Tensor) and t.is_meta for t in leaves)
        if target is getattr and has_meta and args[1] in ('device', 'is_cuda', 'is_meta'):
            from nnscaler.flags import CompileFlag
            on_cpu = CompileFlag.trace_strategy in ('cpu', 'reuse_cache')
            device = torch.device('cpu') if on_cpu else torch.device('cuda', torch.cuda.current_device())
            value = device if args[1] == 'device' else (device.type == 'cuda' if args[1] == 'is_cuda' else False)
            return value, args, kwargs
        # Registered custom kernels keep their original concrete implementation.
        # Builtin/PyTorch shape propagation can avoid allocating activations too.
        module = getattr(target, '__module__', '') or ''
        standard = kind == 'call_method' or module.startswith(('torch', '_operator', 'operator', 'builtins'))
        from .wrap_utils import is_autograd_apply
        standard = standard and not is_autograd_apply(target)
        if not has_meta:
            if not standard and any(isinstance(t, torch.Tensor) for t in leaves):
                return self._concrete(kind, target, args, kwargs)
            return super().run_target(kind, target, args, kwargs)
        name = target if isinstance(target, str) else getattr(target, '__name__', '')
        # Trailing '_' denotes mutation for tensor methods/PyTorch functions,
        # not Python operators such as operator.is_ (``weight is None``).
        tensor_operation = kind == 'call_method' or module.startswith('torch')
        if tensor_operation and name.endswith('_') and not name.endswith('__'):
            raise NotImplementedError('In-place operations on deferred tensors require an explicit tracing rule')
        if standard:
            def to_meta(t):
                return t.to('meta').requires_grad_(t.requires_grad)
            meta_args, meta_kwargs = pytree_utils.tree_map_only(torch.Tensor, to_meta, (args, kwargs))
            try:
                # Meta kernels use Python shape checks (e.g. type(x) is bool).
                # They must see real builtins, not the tracer's replacements.
                context = self.tracer.patcher.revert() if self.tracer.patcher.patch_mode else nullcontext()
                with context:
                    value = self._invoke(kind, target, meta_args, meta_kwargs)
            except (NotImplementedError, RuntimeError):
                return self._concrete(kind, target, args, kwargs)
            self._remember(value, kind, target, args, kwargs)
            return value, args, kwargs
        return self._concrete(kind, target, args, kwargs)


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
    DeferredStrategy._name: DeferredStrategy,
    CudaRunCpuOffloadStrategy._name: CudaRunCpuOffloadStrategy,
    ReuseCacheStrategy._name: ReuseCacheStrategy,
}
