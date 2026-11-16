#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import builtins
import functools
import time
from typing import TYPE_CHECKING, Any, Tuple, Dict, Callable, Type, List, Optional, Iterator

import torch
from torch.fx.node import Target, Node, map_arg
from torch.overrides import is_tensor_method_or_property

from . import pytree_utils, metadata, orig_func, wrap_utils
from .metadata import EmptyResult
from .operator_patcher import OperatorPatcherContext
from .torch_fx_patcher import side_effectful_inplace_ops
from .frame_utils import get_frame_record

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
    return getattr(target, '__qualname__', getattr(target, '__name__', repr(target)))


def meta_guidance(name: str, replay_disabled: bool = False) -> str:
    replay = (
        "Replaying its inputs from the dummy data is off (trace_strategy='meta' or 'meta:no'); "
        "'meta:cpu' or 'meta:cuda' turns it on. "
        if replay_disabled else "")
    return (
        f"trace_strategy='meta' cannot execute `{name}`: it reads tensor values (or has no meta kernel), "
        f"and meta tensors carry no data. {replay}nnScaler traces a static graph, so either trace with a strategy that "
        "runs on real data ('cpu', 'cuda', 'cuda_run_cpu_offload'), or give the operator a stand-in result: "
        "nnscaler.trace_fake(fn, fake, *args, **kwargs) at the call site, nnscaler.update_op(fn, fake_fn=...) "
        "for a torch function, or nnscaler.register_op(..., fake_fn=...) for the enclosing user function."
    )


# Attribute set on tensors produced during tracing.
# ``(node, path)`` names the fx node whose output, indexed by ``path``, this tensor is;
# replay recomputes the node to recover the concrete value.
_ORIGIN_ATTR = '_nnscaler_meta_origin'
# Attribute telling replay where a meta tensor's data lives. Set in exactly two places:
# on the meta copy `_place_tensors_to` makes of a real (cpu/cuda) tensor, pointing at that tensor;
# and on the meta parameters of a model captured by DeferredInitialization, as a callable materializing them.
# A tensor that is already meta is returned as is by `_place_tensors_to` and never gets one.
_SOURCE_ATTR = '_nnscaler_meta_source'
# node.meta key set on nodes that mutate one of their inputs (see record_node_outputs).
# The value is a tuple of the keys (`_object_key`) of the mutated tensor storages or Python objects.
_INPLACE_META = 'nnscaler_inplace'
# node.meta key: the keys of every object a node's inputs alias, so replay knows which mutations it observes.
_READS_META = 'nnscaler_reads'
# node.meta key holding the fake result given to `trace_fake`; a replay returns it instead of executing the node
FAKE_META = 'nnscaler_trace_fake'


def attach_meta_materializer(module: torch.nn.Module, materialize: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Let meta-trace replay obtain the values of a captured model's parameters and buffers."""
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.is_meta:
            setattr(tensor, _SOURCE_ATTR, functools.partial(materialize, tensor))


def _tensor_leaves_with_path(value: Any, path: Tuple[Any, ...] = ()) -> Iterator[Tuple[Tuple[Any, ...], torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensor_leaves_with_path(item, path + (index,))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_leaves_with_path(item, path + (key,))


def _objects_in(value: Any) -> Iterator[Any]:
    """``value`` and, for list/tuple/dict containers, every object nested in it (the containers included)."""
    yield value
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _objects_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _objects_in(item)


def _index_path(value: Any, path: Tuple[Any, ...]) -> Any:
    for key in path:
        value = value[key]
    return value


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

    def __init__(self, tracer: 'ConcreteTracer', main_device: str, option: str = '') -> None:
        self.check_option(option)
        self.tracer = tracer
        self.main_device = main_device

    @classmethod
    def check_option(cls, option: str) -> None:
        """Validate the ``:option`` suffix of a ``trace_strategy`` string; most strategies take none."""
        if option:
            raise ValueError(f"trace strategy '{cls._name}' takes no ':option', got '{cls._name}:{option}'")

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

    def record_node_outputs(self, node: Node, value: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        """Called by the tracer after a node is created, with its concrete result and the executed args/kwargs."""

    def concretize(self, tensor: torch.Tensor, *, name: str, cause: BaseException) -> torch.Tensor:
        """Return a tensor with real data for ``tensor``; only the meta strategy produces meta values."""
        raise cause

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

    def __init__(self, tracer: 'ConcreteTracer', option: str = ''):
        super().__init__(tracer, 'cpu', option)


class CudaStrategy(BaseTraceStrategy):
    """
    Pure cuda strategy, model is placed on cuda, run target is on cuda, intermediate results are on cuda.
    """
    _name = 'cuda'

    def __init__(self, tracer: 'ConcreteTracer', option: str = ''):
        super().__init__(tracer, 'cuda', option)


class MetaStrategy(BaseTraceStrategy):
    """
    Meta strategy, run target is on meta, intermediate results are on meta.
    A model that is already entirely on meta is traced there; any other model is placed on cpu.

    Operators that read tensor values (``.item()``, ``nonzero``, branch conditions, ...) cannot run on meta.
    By default (``meta`` or ``meta:no``) they raise.
    With ``meta:cpu`` or ``meta:cuda`` they fall back to a replay:
    the traced prefix they depend on is executed on that device from the dummy inputs and parameters,
    the operator runs on those values, and its outputs return to meta.
    nnScaler traces a static graph, so this picks the branch/shape the dummy inputs imply,
    exactly like the cpu strategy, without materializing the whole model.
    """
    _name = 'meta'
    # ':option' -> device the replay runs on; None (the default) raises instead of replaying
    _FALLBACK_DEVICES = {'': None, 'no': None, 'cpu': 'cpu', 'cuda': 'cuda'}
    # On meta there is nothing to move and nothing can be copied out of it (`x.cpu()` raises),
    # so every device argument executes as the strategy device, cuda or not;
    # the graph still records what the user wrote.
    # A replay likewise keeps everything on its fallback device.
    _DEVICE_METHODS = ('cuda', 'cpu')

    @classmethod
    def _is_mapped_device(cls, value: Any) -> bool:
        return isinstance(value, (int, str, torch.device)) and not isinstance(value, bool)

    @classmethod
    def check_option(cls, option: str) -> None:
        if option not in cls._FALLBACK_DEVICES:
            raise ValueError(
                f"trace strategy 'meta' accepts the options 'no' (default), 'cpu' and 'cuda' "
                f"for the data-dependent replay, got 'meta:{option}'")

    def __init__(self, tracer: 'ConcreteTracer', option: str = ''):
        super().__init__(tracer, 'meta', option)
        self.fallback_device: Optional[str] = self._FALLBACK_DEVICES[option]
        # Checked here rather than in `check_option`:
        # a ComputeConfig saying 'meta:cuda' is also loaded by processes that only run the generated code.
        if self.fallback_device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("trace_strategy='meta:cuda' requires CUDA for the replay")
        # Replays seed each node from the trace's initial seed and the node's position,
        # so a random tensor gets the same values in every replay that recomputes it.
        self._seed = torch.initial_seed()
        # real tensor id -> (real tensor, its meta copy). Purpose: `_place_tensors_to` returns the same meta
        # copy every time the same real tensor is placed, so all operators reading it see one storage.
        # Replay tells mutation from read by storage identity (`_object_key`): with a fresh copy per operator,
        # `b.mul_(0)` would mutate one storage and a later `torch.nonzero(b)` read another, and the mutation
        # would not be replayed. It also sets `_SOURCE_ATTR` / `_ORIGIN_ATTR` once per tensor.
        # The real tensor is held so its id cannot be reused by another object; the table lives as long as
        # this strategy, i.e. one trace. Real tensors reach an operator in four ways:
        #   - the dummy inputs behind the placeholders;
        #   - parameters, buffers and plain tensor attributes that are not meta: a module instance given to
        #     `parallelize`, or a tensor that existed before the DeferredInitialization capture
        #     (a module-level constant table, a tensor from a `module_fn` closure);
        #   - cpu results of factory calls without a device (`torch.rand(4)`, `torch.arange(n)`), which run on
        #     cpu like at runtime since only an explicit device is mapped (see `_map_devices_to`);
        #   - the result given to `trace_fake`.
        # Replay never adds to it: its values stay local to `_replay` and return to meta via `empty_strided`.
        self._meta_copies: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def place_model(self, model: torch.nn.Module) -> torch.nn.Module:
        # Meta tensors stay on meta: they come from a model built on meta or captured by DeferredInitialization,
        # and shard initialization re-creates the model at runtime, so nothing is lost.
        # Real tensors are kept on cpu rather than moved to meta because their values are still needed:
        #   - a cpu instance passed to `parallelize` is the source the runtime copies the shard values from;
        #   - in a captured constructor, a tensor that existed before the capture and was registered as it is
        #     (a module-level constant table, a tensor from a `module_fn` closure) stays real
        #     and is the only copy of its data, which the replay of a data-dependent operator may read.
        # Execution does not need them on meta: `_place_tensors_to` makes a meta copy per operator.
        return place_real_tensors_to_cpu(model)

    def _place_tensors_to(self, *args, device: str) -> Tuple[Any]:
        # every caller of this strategy executes on meta; `device` only exists for the base-class signature
        assert device == 'meta', f'the meta strategy places tensors on meta only, got {device!r}'

        def place(x: torch.Tensor) -> torch.Tensor:
            if x.is_meta:
                return x
            entry = self._meta_copies.get(id(x))
            if entry is None:
                moved = x.to('meta').requires_grad_(x.requires_grad)
                # x is the real tensor the meta copy was made from
                setattr(moved, _SOURCE_ATTR, x)
                origin = getattr(x, _ORIGIN_ATTR, None)
                if origin is not None:
                    # A cpu-resident traced value (e.g. `torch.rand(4)` without device) keeps resolving
                    # through its node, so all consumers see the same replayed value.
                    setattr(moved, _ORIGIN_ATTR, origin)
                entry = self._meta_copies[id(x)] = (x, moved)
            return entry[1]

        return pytree_utils.tree_map_only(torch.Tensor, place, args)

    # Methods that mutate a container receiver (`parts.append(x)` on a traced list); their result does not alias it.
    _MUTATING_METHODS = frozenset((
        'append', 'extend', 'insert', 'pop', 'remove', 'clear', 'sort', 'reverse',
        'update', 'setdefault', 'popitem', 'add', 'discard'))

    @staticmethod
    def _object_key(obj: Any) -> int:
        """
        Identity of what a node reads or mutates through ``obj``.
        For a tensor it is its storage: every alias (view, `detach()`, `.data`) shares it, so a mutation
        through one alias is attributed to the storage all of them read. Meta tensors carry no data but do
        have a storage object, and `_place_tensors_to` gives a real tensor one meta copy for the whole trace,
        so two reads of the same parameter or buffer share it too.
        Any other object (an iterator consumed by `next`, a list) is identified by itself.
        """
        return obj.untyped_storage()._cdata if isinstance(obj, torch.Tensor) else id(obj)

    def record_node_outputs(self, node: Node, value: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        """
        Three things are done here, all for replay:
        1. Record the new tensors produced by the node (their origin).
        2. Detect if the node mutates one of its inputs in-place, and which storage.
        3. Record which storages the node reads.
        """
        # Placeholders and attributes resolve through the tracer, not through a node.
        if node.op in ('placeholder', 'get_attr', 'output'):
            return
        input_ids = {id(t) for _, t in _tensor_leaves_with_path((args, kwargs))}
        for path, tensor in _tensor_leaves_with_path(value):
            # An output that is an input keeps its origin or leaf source, even through an in-place op:
            # replay resolves the value from there, and the mutation is reproduced by running the node
            # (marked below) in graph order on that same replayed object, as all its readers alias it.
            if id(tensor) not in input_ids and not hasattr(tensor, _ORIGIN_ATTR):
                setattr(tensor, _ORIGIN_ATTR, (node, path))
        # A node that mutates an input must be replayed when a later node reads the mutated object,
        # even if nothing depends on its output. Which objects it mutates follows from the node's form.
        mutated: Tuple[Any, ...] = ()
        if (node.target in side_effectful_inplace_ops
                or wrap_utils.unwrap(node.target) is builtins.next
                or (node.op == 'call_method' and node.target in self._MUTATING_METHODS
                    and not isinstance(args[0], torch.Tensor))):
            # Case 1: the operator mutates its first operand by definition, tensor or not:
            # `x += y`, `lst += [v]`, `x[i] = v`, `lst[i] = v` (side_effectful_inplace_ops);
            # `next(it)` advancing the iterator of a `for` over a traced tensor;
            # a container method such as `lst.append(v)`.
            # The result says nothing here: setitem is recorded with the container as result, `append` returns None.
            mutated = (args[0],)
        elif self._has_mutating_signature(node):
            # Case 2: an in-place tensor operator (`add_`, `F.relu(inplace=True)`, `out=`) returns the tensors
            # it wrote, i.e. some of its inputs; `out=(values, indices)` returns several of them.
            # Both signs are required: `x.contiguous()` returns its input but has no mutating signature,
            # and a `_`-suffixed name alone does not prove the result aliases an input.
            mutated = tuple(t for _, t in _tensor_leaves_with_path(value) if id(t) in input_ids)
        if mutated:
            node.meta[_INPLACE_META] = tuple(set(map(self._object_key, mutated)))
        node.meta[_READS_META] = tuple({self._object_key(o) for o in (*_objects_in(args), *_objects_in(kwargs))})

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
            return self._run_with_replay(kind, target, args, kwargs, exc)

    def _concrete_values(self, tensors: List[torch.Tensor], name: str, cause: BaseException) -> List[torch.Tensor]:
        """Real values of meta ``tensors`` for the operator ``name`` that failed on meta with ``cause``."""
        if self.fallback_device is None:
            raise RuntimeError(meta_guidance(name, replay_disabled=True)) from cause
        try:
            return self._replay(tensors, name)
        except Exception as exc:
            raise RuntimeError(
                f"{meta_guidance(name)} Replaying its inputs on {self.fallback_device} failed: {exc} "
                f"(on meta it failed with: {cause})") from exc

    def concretize(self, tensor: torch.Tensor, *, name: str, cause: BaseException) -> torch.Tensor:
        return self._concrete_values([tensor], name, cause)[0]

    def _run_with_replay(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                         cause: BaseException) -> Tuple[Any, Tuple, Dict]:
        name = _target_name(target)
        # the same placed arguments the normal path returns, so proxies see meta values either way
        args, kwargs = self._place_tensors_to(args, kwargs, device='meta')
        metas = [t for t in pytree_utils.tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        values = self._concrete_values(metas, name, cause)
        # The replay ran without grad; the operator itself runs with the inputs' grad flags
        # so the outputs' requires_grad is the real one.
        concrete = {id(meta): value.detach().requires_grad_(meta.requires_grad) for meta, value in zip(metas, values)}
        exec_args, exec_kwargs = pytree_utils.tree_map_only(torch.Tensor, lambda t: concrete[id(t)], (args, kwargs))
        # Real tensors now: cpu/cuda kernels run in C++ and do not need the builtins restored (see `run_target`).
        if kind == 'call_function':
            result = self._call_function(target, exec_args, exec_kwargs, self.fallback_device)
        else:
            self_obj, *tail = exec_args
            result = self._call_method(self_obj, target, tuple(tail), exec_kwargs, self.fallback_device)

        # An in-place operator returns its input: hand back the meta input object,
        # so the node is recorded as in-place and proxies keep their object,
        # exactly as when the operator runs on meta itself.
        inputs_of = {id(concrete[id(meta)]): meta for meta in metas}

        def to_meta(t: torch.Tensor) -> torch.Tensor:
            if id(t) in inputs_of:
                return inputs_of[id(t)]
            return torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device='meta').requires_grad_(t.requires_grad)

        return pytree_utils.tree_map_only(torch.Tensor, to_meta, result), args, kwargs

    # ---- replay -------------------------------------------------------------------------------

    def _call_unpatched(self, fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        # Same rule as ConcreteTracer.create_proxy:
        # torch.compile'd and autograd functions must see the original torch functions,
        # so every patch is reverted around them. Anything else runs on real tensors as in a cpu trace.
        if self.tracer.need_revert(fn) or wrap_utils.is_autograd_apply(fn):
            with self.tracer.patcher.revert():
                return self._call_function(fn, args, kwargs, self.fallback_device)
        return self._call_function(fn, args, kwargs, self.fallback_device)

    @staticmethod
    def _has_mutating_signature(node: Node) -> bool:
        name = node.target if node.op == 'call_method' else getattr(node.target, '__name__', '')
        return ((name.endswith('_') and not name.endswith('__'))
                or 'out' in node.kwargs or bool(node.kwargs.get('inplace', False)))

    def _needed_nodes(self, nodes: List[Node], tensors: List[torch.Tensor]) -> set:
        """The nodes a replay must execute to obtain ``tensors``; see `_replay`."""
        position = {node: i for i, node in enumerate(nodes)}
        inplace = [(node, node.meta[_INPLACE_META]) for node in nodes if _INPLACE_META in node.meta]   # node, mutated keys
        # object key -> position of the latest needed node reading it;
        # the operator being resolved comes after every node.
        # Only the latest reader matters:
        # an in-place node must run iff some needed node reads the object it mutates after it.
        latest_reads: Dict[Any, int] = {}

        def record_latest_reads(keys, at: int) -> None:
            """A needed node at position ``at`` reads the objects ``keys``."""
            for key in keys:
                if latest_reads.get(key, -1) < at:
                    latest_reads[key] = at

        record_latest_reads(map(self._object_key, tensors), len(nodes))
        needed = set()
        pending: list[Node] = [getattr(t, _ORIGIN_ATTR)[0] for t in tensors if getattr(t, _ORIGIN_ATTR, None) is not None]
        while True:
            # collect node with input dependencies
            while pending:
                node = pending.pop()
                if node in needed:
                    continue
                needed.add(node)
                if FAKE_META in node.meta:   # a faked node is a leaf of the replay
                    continue
                pending.extend(node.all_input_nodes)
                record_latest_reads(node.meta.get(_READS_META, ()), position[node])

            # collect in-place nodes that are needed
            pending = [node for node, keys in inplace
                       if node not in needed and any(latest_reads.get(key, -1) > position[node] for key in keys)]
            if not pending:
                break

        return needed

    def _replay(self, tensors: List[torch.Tensor], name: str) -> List[torch.Tensor]:
        """
        Concrete cpu values of meta ``tensors``.

        The traced graph is executed in order from the dummy inputs and parameters.
        Nodes that are not ancestors of ``tensors`` are skipped, except in-place nodes whose mutation a
        needed node observes: fx edges do not record mutation, so a node is needed when it mutates a storage
        (`_object_key`) that a needed node reads later in graph order; its ancestors are then needed too,
        up to a fixpoint. Executing in graph order on the cached objects reproduces the aliasing itself.
        """
        nodes = list(self.tracer.graph.nodes)
        needed = self._needed_nodes(nodes, tensors)

        # node -> replayed result, and real tensor -> its single replay copy.
        # Every get_attr node and every placed copy of the same parameter resolve to that one copy,
        # so an in-place node mutating it is seen by all later readers,
        # as in the trace where they alias the same storage.
        cache: Dict[Node, Any] = {}
        leaves: Dict[int, torch.Tensor] = {}
        on_cuda = self.fallback_device == 'cuda'
        rng_devices = [torch.cuda.current_device()] if on_cuda else []
        with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
            for index, node in enumerate(nodes):
                if node in needed:
                    # Not torch.manual_seed: it would also reseed every CUDA device, which fork_rng does not restore.
                    # Random nodes all draw from the global RNG: fx rejects a `generator=` argument (`create_arg`),
                    # so none is recorded in the graph; supporting one would need a private generator per node.
                    seed = (self._seed + index) % (2 ** 63)
                    torch.default_generator.manual_seed(seed)
                    if on_cuda:
                        torch.cuda.manual_seed(seed)
                    try:
                        cache[node] = self._execute(node, cache, leaves)
                    except Exception as exc:
                        raise RuntimeError(f'replaying node `{node.name}` ({_target_name(node.target)}) failed: {exc}') from exc
            values = [self._value_of(tensor, cache, leaves) for tensor in tensors]
        _logger.warning(
            "trace_strategy='meta': `%s` needs tensor data; replayed %d traced nodes on %s. %s",
            name, len(cache), self.fallback_device, get_frame_record())
        return values

    def _leaf_value(self, tensor: torch.Tensor, leaves: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        The replay copy of a placeholder/parameter tensor.
        One private copy per real tensor for the whole replay,
        so in-place nodes mutate it for every reader without touching the original.
        """
        source = getattr(tensor, _SOURCE_ATTR, None) if tensor.is_meta else tensor
        if source is None:
            raise RuntimeError('a meta tensor has no recorded source (not a traced value, input or parameter)')
        if id(source) not in leaves:
            real = source() if callable(source) else source
            # No grad flag: the replay runs under no_grad and only computes values;
            # `_run_with_replay` restores requires_grad on the operator's inputs from their meta tensors.
            leaves[id(source)] = real.detach().to(self.fallback_device, copy=True)
        return leaves[id(source)]

    def _value_of(self, tensor: torch.Tensor, cache: Dict[Node, Any], leaves: Dict[int, torch.Tensor]) -> torch.Tensor:
        origin = getattr(tensor, _ORIGIN_ATTR, None)
        if origin is not None:
            return _index_path(cache[origin[0]], origin[1])
        return self._leaf_value(tensor, leaves)

    def _execute(self, node: Node, cache: Dict[Node, Any], leaves: Dict[int, torch.Tensor]) -> Any:
        if FAKE_META in node.meta:
            # a fresh copy per replay: an in-place node downstream must not change the stored fake
            return pytree_utils.tree_map_only(
                torch.Tensor, lambda t: t.detach().to(self.fallback_device, copy=True), node.meta[FAKE_META])
        leaf = functools.partial(self._leaf_value, leaves=leaves)
        if node.op == 'placeholder':
            return pytree_utils.tree_map_only(torch.Tensor, leaf, self.tracer.placeholder_dict[node.target])
        if node.op == 'get_attr':
            return pytree_utils.tree_map_only(torch.Tensor, leaf, self.tracer.fetch_attr(node.target))
        if node.op == 'output':
            return None
        args = map_arg(node.args, lambda n: cache[n])
        kwargs = map_arg(node.kwargs, lambda n: cache[n])
        if node.op == 'call_function':
            # a registered operator runs its tracing stand-in, like during the trace itself
            info = self.tracer.autowrap_leaf_function.get(node.target)
            fn = info.replacement if info is not None and info.replacement is not None else node.target
            result = self._call_unpatched(fn, args, kwargs)
            # the tracer records the mutated container as the result of setitem (see create_proxy)
            return args[0] if node.target is orig_func.setitem else result
        if node.op == 'call_method':
            self_obj, *tail = args
            info = self.tracer.autowrap_leaf_function.get(wrap_utils.unwrap(getattr(type(self_obj), node.target, None)))
            if info is not None and info.replacement is not None:
                return self._call_unpatched(info.replacement, args, kwargs)
            return self._call_method(self_obj, node.target, tuple(tail), kwargs, self.fallback_device)
        # `call_module` cannot occur: ConcreteTracer.is_leaf_module is always False and the parser rejects it
        raise RuntimeError(f'unexpected node kind {node.op}')


class CudaRunCpuOffloadStrategy(BaseTraceStrategy):
    """
    This is the previous tracer run target logic (nnscaler <= v0.2).

    Model is placed on cpu, run target is on cuda, intermediate results are on cpu.
    If detect OOM during run target, will retry run target on cpu.
    """
    _name = 'cuda_run_cpu_offload'

    def __init__(self, tracer: 'ConcreteTracer', option: str = ''):
        super().__init__(tracer, 'cpu', option)

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

    def __init__(self, tracer: 'ConcreteTracer', option: str = '') -> None:
        super().__init__(tracer, option)
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


def parse_trace_strategy(spec: str) -> Tuple[Type[BaseTraceStrategy], str]:
    """Split and validate ``'name[:option]'``; only ``meta`` takes an option (``cpu``, ``cuda`` or ``no``)."""
    name, _, option = spec.partition(':')
    if name not in TRACE_STRATEGY:
        raise ValueError(f"unknown trace strategy '{name}', expected one of {sorted(TRACE_STRATEGY)}")
    cls = TRACE_STRATEGY[name]
    cls.check_option(option)
    return cls, option


def create_trace_strategy(spec: str, tracer: 'ConcreteTracer') -> BaseTraceStrategy:
    cls, option = parse_trace_strategy(spec)
    return cls(tracer, option)


def trace_fake(fn: Callable, fake: Any, *args, **kwargs) -> Any:
    """
    ``fn(*args, **kwargs)`` recorded in the graph, with ``fake`` as its value during the trace.

    The same contract as the ``fake_fn`` of a registered operator, for one call site:
    while tracing, ``fn`` is not executed and ``fake`` stands in for its result;
    a meta-trace replay likewise returns ``fake`` for this node without executing anything before it.
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
        # A private copy: a fake built in forward (`torch.zeros(...)`) is itself a traced value
        # with an origin and possibly autograd history, which must not leak into the result of `fn`.
        fake = pytree_utils.tree_map_only(torch.Tensor, lambda t: t.detach().clone().requires_grad_(t.requires_grad), fake)

    if is_tensor_method_or_property(fn):
        # be consistent with ConcreteAttrProxy.__call__
        return tracer.create_proxy('call_method', fn.__name__, args, kwargs, fake_result=fake)
    return tracer.create_proxy('call_function', fn, args, kwargs, fake_result=fake)
