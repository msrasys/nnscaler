#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""In-memory initialization of generated modules.

Deferred initialization records pure dispatched operations using meta tensors for
selective CPU replay. Data-dependent tags and meta NotImplementedError trigger a
logged concrete execution during capture; tensor results are then discarded.
This supports scalar control flow but is not a general lazy tensor implementation:
mutations and aliasing operations still require explicit support.

Replay restores the captured default dtype. Supported random operations, including
generically recorded operators tagged as seeded nondeterministic, use independent
per-operation seeds rather than eager PyTorch's random-number stream. Meta execution
does not isolate RNG side effects. Custom-operator replay preserves Python, NumPy,
torch CPU and current-device CUDA RNG states; native random replay uses private
generators or scoped CPU RNG.
Existing real inputs are borrowed and must remain unchanged until
replay. ``torch.tensor`` literals allocate before dispatch, and concrete fallback
temporarily allocates real storage, so these allocations are not deferred.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
import inspect
import logging
import random
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Optional, Type

import numpy as np
import torch
from torch.utils._python_dispatch import (
    TorchDispatchMode, _get_current_dispatch_mode_stack, _pop_mode_temporarily,
)

from nnscaler.utils import get_member_by_name

if TYPE_CHECKING:
    from nnscaler.runtime.module import AttrMeta


_logger = logging.getLogger(__name__)


@contextmanager
def _default_dtype(dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(previous)


@contextmanager
def preserve_rng_state():
    """Preserve Python, NumPy, torch CPU and current-device CUDA RNG states."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def _initialization_rng(seed: Optional[int]):
    with preserve_rng_state() if seed is not None else nullcontext():
        if seed is not None:
            random.seed(seed)
            # Deferred node seeds are 63-bit; NumPy's legacy RNG accepts 32-bit seeds.
            np.random.seed(seed % (2 ** 32))
            # torch.manual_seed also changes other CUDA devices, outside the saved RNG scope.
            torch.random.default_generator.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        yield


def create_init_module(
    module_class: Type[torch.nn.Module],
    module_fn: Optional[Callable[[], torch.nn.Module]],
    module_dtype: Optional[torch.dtype],
    *,
    seed: Optional[int] = None,
) -> torch.nn.Module:
    with _initialization_rng(seed):
        module = module_class() if module_fn is None else module_fn()
        if type(module) is not module_class:
            raise ValueError(f"module_fn should return a {module_class} instance.")
        if module_dtype is not None:
            module.to(dtype=module_dtype)
    return module


def create_shard_init_weights(
    module_class: Type[torch.nn.Module],
    attr_meta_map: Dict[str, "AttrMeta"],
    *,
    seed: int,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield user-provided shards with lazy execution inside the initialization RNG scope."""
    initializer = getattr(module_class, '__shard__init__', None)
    if not callable(initializer):
        raise RuntimeError(
            "A shard initializer must define "
            "__shard__init__(attr_meta_map) as a staticmethod or classmethod."
        )
    with _initialization_rng(seed):
        values = initializer(attr_meta_map)
        if isinstance(values, dict):
            values = values.items()
        yield from values


def _get_init_source_groups(
    module: torch.nn.Module, attr_meta_map: Dict[str, "AttrMeta"],
) -> Dict[int, tuple[torch.Tensor, list[tuple[str, "AttrMeta"]]]]:
    groups = {}
    for attr, meta in attr_meta_map.items():
        source = get_member_by_name(module, meta.orig_name)
        if (
            not isinstance(source, torch.Tensor)
            or tuple(source.shape) != tuple(meta.shape)
            or source.dtype != meta.dtype
        ):
            raise RuntimeError(
                f"Invalid initialization tensor for {meta.orig_name}: "
                f"expected shape {meta.shape} and dtype {meta.dtype}."
            )
        if id(source) not in groups:
            groups[id(source)] = (source, [])
        groups[id(source)][1].append((attr, meta))
    return groups


def capture_init_weights(
    attr_meta_map: Dict[str, "AttrMeta"],
    *,
    module_fn: Optional[Callable[[], torch.nn.Module]] = None,
    module: Optional[torch.nn.Module] = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Implement the __shard__init__ contract using captured construction.

    The bound factory supplies constructor arguments, dtype and initialization seed.
    A supplied instance is used as-is. Yield local views without value-division;
    ParallelModule copies each view before advancing and applies val_chunks.
    Consumers must release each view before requesting the next source tensor.
    """
    if not attr_meta_map:
        return
    capture = None
    if module is None:
        if module_fn is None:
            raise RuntimeError(
                "Independent initialization requires parallelize() with the original "
                "module class/module_fn, or an init_module instance."
            )
        with DeferredInitialization() as capture:
            module = module_fn()
    with torch.no_grad():
        for source, entries in _get_init_source_groups(module, attr_meta_map).values():
            value = source if capture is None else capture.materialize(source)
            for attr, meta in entries:
                yield attr, value[meta.slicers]
            del value


def _unsupported(reason):
    return RuntimeError(
        f"Deferred initialization failed: {reason}. Use other parameter initialization strategies for this constructor."
    )


def _layout(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())


def _map(value, transform):
    if isinstance(value, (tuple, list)):
        return type(value)(_map(item, transform) for item in value)
    if isinstance(value, dict):
        return {key: _map(item, transform) for key, item in value.items()}
    return transform(value)


@dataclass(eq=False)
class PyFunction:
    """A Python callable using the same schema, tags and operation names as dispatch.

    Writes/views use DeferredInitialization's existing supported-name sets.
    Writes preserve metadata and return their first argument; pure functions
    return fresh tensors. Use ``call_with_normalized_args`` at the constructor entry.
    """

    func: Callable[..., Any]
    schema: str
    tags: tuple = ()
    _schema: torch.FunctionSchema = field(init=False, repr=False)
    signature: inspect.Signature = field(init=False, repr=False)

    def __post_init__(self):
        self._schema = torch._C.parse_schema(self.schema)
        self.signature = inspect.signature(self.func)

    def __call__(self, *args, **kwargs):
        # Like dispatcher redispatch, each handler runs with only the lower modes
        # active. Calling func(...) forwards this same PyFunction to the next mode.
        if _get_current_dispatch_mode_stack():
            with _pop_mode_temporarily() as mode:
                return mode.__torch_dispatch__(self, (), args, kwargs)
        # In-place Python initializers need no meta execution, only the write recipe.
        if args and isinstance(args[0], torch.Tensor) and args[0].is_meta:
            # Schema alias_info describes storage aliasing: Tensor(a) shares alias
            # set "a"; Tensor(a!) also marks a write (is_write). Without an alias
            # annotation, alias_info is None. Aliasing alone does not imply mutation.
            # TODO: this may not right for all functions.
            #       we should check its correctness for each specific function.
            alias = self._schema.arguments[0].alias_info
            if alias is not None and alias.is_write:
                return args[0]
        return self.func(*args, **kwargs)

    def call_with_normalized_args(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        # Keep replay-controlled arguments out of the positional recipe.
        options = {name: bound.arguments.pop(name) for name in ("generator", "device")
                   if name in bound.arguments}
        return self(*bound.args, **bound.kwargs, **options)


# Key: (module owning the function, attribute name).
# Value: (operator schema, dispatch tags, predicate evaluated on entry to enable the patch).
_PYTHON_FUNCTION_PATCHES: dict[
    tuple[ModuleType, str], tuple[str, tuple[torch.Tag, ...], Callable[[], bool]]
] = {
    # PyTorch 2.12+ skips meta initialization before dispatch. Patch the helper
    # so existing aliases of the public initializer also capture the whole call.
    (torch.nn.init, "_no_grad_trunc_normal_"): (
        "python::trunc_normal_(Tensor(a!) tensor, float mean, float std, float a, float b, "
        "Generator? generator=None) -> Tensor(a!)",
        (torch.Tag.nondeterministic_seeded,),
        lambda: torch.__version__ >= (2, 12),
    ),
}


@dataclass(eq=False)
class _Node:
    # Meta-only layout template; None for a shared structured/scalar result.
    tensor: Optional[torch.Tensor]
    # Captured operator or PyFunction; None for projection nodes.
    func: Any
    # Positional arguments with tensor dependencies snapshotted; excludes the write destination.
    args: Any
    # Snapshotted keyword arguments; generators are replaced during replay using seed.
    kwargs: Any
    # Per-operation random seed; None when no seeded replay is needed.
    seed: Optional[int] = None
    # Old storage version copied into a fresh buffer before this write.
    # None for full overwrites that do not need the old contents, and for non-write nodes.
    previous: Optional["_Node"] = None
    # Write destination layout (shape, stride, storage_offset) within that fresh buffer.
    # A non-None value identifies a write node.
    target: Any = None
    # Capture-time default dtype, temporarily restored when executing the operator.
    default_dtype: torch.dtype = field(default_factory=torch.get_default_dtype)
    # Shared producer for a projection node; its result is indexed, not copied or modified.
    operation: Optional["_Node"] = None
    # Index/key path from operation's structured result to this tensor; otherwise empty.
    projection: tuple = ()


@dataclass
class _Storage:
    tensor: torch.Tensor
    # Latest allocation or write to this storage.
    node: _Node


@dataclass
class _Read:
    node: _Node
    layout: tuple[tuple[int, ...], tuple[int, ...], int]  # (shape, stride, storage_offset)


@dataclass
class _External:
    # External tensors are those created outside the capture context
    # They are not managed by the deferred initialization system and are assumed to be fully initialized.
    # And during capture, they must not be modified.
    tensor: torch.Tensor
    version: int


class DeferredInitialization(TorchDispatchMode):
    """Capture a model factory, then materialize individual tensors on CPU.

    Example::

        with DeferredInitialization() as capture:
            model = factory()
        weight = capture.materialize(model.weight)

    Pure dispatched operators normally produce meta tensors and dependency
    recipes, including projections of shared multi-output operations. When an
    operator lacks meta support or needs data, capture replays its dependencies
    on CPU to determine output metadata. Tensor results return to the capture
    graph as meta tensors; scalar results can drive constructor control flow.
    Fallback therefore trades capture-time memory for broader constructor support.

    Each materialization evaluates only reachable dependencies with an ephemeral
    cache. Copied weights may require replaying their sources. Fallback operations
    are replayed like ordinary operations, without retaining concrete results
    between calls. Recipes capture the default dtype and random
    seeds; random values are repeatable but need not match eager initialization.

    Empty allocations are replayed without initializing their values, like eager
    PyTorch. Supported writes require a contiguous destination view; these include
    scalar arithmetic, ``erfinv_`` and scalar ``clamp_``. External writes, ``out=``,
    other mutations, tensor-valued arithmetic mutations, unsupported aliasing
    operators and overlapping allocation layouts are rejected. Capture is single-use,
    and public ``materialize`` calls must occur after leaving the context.
    Only initialization inside the context is replayed; later value mutations of
    captured meta tensors are ignored. Do not change their metadata afterwards.
    """

    _views = {
        "detach", "alias", "view", "_unsafe_view", "transpose", "t", "permute",
        "slice", "select", "unsqueeze", "squeeze", "expand", "unbind",
    }
    _arithmetic_writes = {"add_", "sub_", "mul_", "div_"}
    _unary_writes = {"erfinv_", "clamp_"}
    _random_writes = {
        "uniform_", "normal_", "random_", "bernoulli_", "exponential_",
        "geometric_", "log_normal_", "cauchy_", "trunc_normal_",
    }
    _sampling = {"normal", "bernoulli", "poisson", "multinomial"}
    _random_factories = {"rand", "randn", "randint", "randperm"}
    _random_like = {"rand_like", "randn_like", "randint_like"}
    _writes = {"fill_", "zero_", "copy_"} | _arithmetic_writes | _random_writes | _unary_writes
    _random = _random_factories | _random_like | _random_writes | _sampling

    def __init__(self):
        super().__init__()
        # Maps untyped storage pointers to storage metadata.
        # only track tensors created within the capture context
        # external tensors are not tracked
        self._storages: dict[int, _Storage] = {}
        self._calls = 0  # Dispatch count used to derive distinct per-operation random seeds.
        self._used = False  # Permanently set on first entry to prevent capture-context reuse.
        self._active = False  # True inside capture, when materialization is forbidden.

    def __enter__(self):
        if self._used:
            raise _unsupported("a capture context cannot be reused")
        self._used = True
        with ExitStack() as patches:
            for (module, name), (schema, tags, should_patch) in _PYTHON_FUNCTION_PATCHES.items():
                if should_patch():
                    original = getattr(module, name)
                    patches.callback(setattr, module, name, original)
                    setattr(module, name, PyFunction(original, schema, tags).call_with_normalized_args)
            self._patches = patches.pop_all()
        try:
            result = super().__enter__()
        except BaseException:
            self._patches.close()
            raise
        self._active = True
        return result

    def __exit__(self, *args):
        self._active = False
        try:
            return super().__exit__(*args)
        finally:
            self._patches.close()

    def _storage(self, tensor: torch.Tensor):
        if tensor.layout != torch.strided:
            raise _unsupported("only strided tensors are supported")
        return self._storages.get(tensor.untyped_storage())

    def _snapshot(self, value):
        """
        Three types of values:
        1. _Read: Internal tensors created within the capture context
        2. _External: External tensors not created within the capture context
        3. Other Python values (ints, floats, tuples, etc.)
        """
        if not isinstance(value, torch.Tensor):
            return value

        # internal tensors are those created within the capture context. They are tracked by their storage.
        storage = self._storage(value)
        if storage is not None:
            return _Read(storage.node, _layout(value))

        # external tensors are those not created within the capture context. They are tracked
        # by their version to detect modifications after capture.
        if value.is_meta:
            raise _unsupported("an untracked meta tensor has no initialization recipe")
        try:
            return _External(value, value._version)
        except RuntimeError as exc:
            raise _unsupported("external inference tensors cannot be version checked") from exc

    def _register(self, tensor, node):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_meta:
            raise _unsupported("an operation did not produce a meta tensor")
        # Checks whether distinct elements of this tensor share a storage location:
        # 0 = no overlap, 1 = overlap, 2 = too hard to determine (not proof of overlap).
        # Conservatively reject both overlapping and indeterminate allocation layouts.
        if torch._debug_has_internal_overlap(tensor) != 0:
            raise _unsupported("allocations with overlapping or non-dense strides")
        self._storages[tensor.untyped_storage()] = _Storage(tensor, node)
        return tensor

    def __torch_dispatch__(self, func: torch._ops.OpOverload | PyFunction, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        # Schema names are "namespace::operator" (e.g. "aten::add"), without the overload suffix;
        # overload_name is an operator-specific string, not an enum or a fixed set:
        # "" is the unnamed overload (exposed as .default by torch.ops);
        # add's "Tensor"/"Scalar" select a tensor/scalar second operand;
        # add's "out"/"Scalar_out" write into the supplied out tensor;
        # empty's "memory_format" selects the signature accepting memory_format.
        name = func._schema.name.split("::")[-1]
        self._calls += 1
        # out= writes into existing storage outside the supported _write path.
        # Schema flags also cover Scalar_out and outputs named min/max or values/indices.
        if any(arg.is_out for arg in func._schema.arguments):
            raise _unsupported(f"out= overload {func}")

        if name in self._views:
            return self._view(func, args, kwargs)
        if name in self._writes:
            return self._write(func, name, args, kwargs)
        if name == "lift_fresh":
            # Literals already own real storage before dispatch; borrow it and
            # use the ordinary clone recipe for independent materializations.
            return self._op(torch.ops.aten.clone.default, "clone", args, kwargs)

        return self._op(func, name, args, kwargs)

    def _view(self, func, args, kwargs):
        # Views share the producer's recipe with a different layout,
        # so no new initialization node is needed.
        storage = self._storage(args[0])
        if storage is None and args[0].is_meta:
            raise _unsupported(f"view of an untracked meta tensor in {func}")
        result = func(*args, **kwargs)
        views = result if isinstance(result, (tuple, list)) else (result,)
        for view in views:
            if view.untyped_storage() is not args[0].untyped_storage():
                raise _unsupported(f"view {func} unexpectedly allocated new storage")
        return result

    def _op(self, func, name, args, kwargs):
        """Capture factories, copies and other pure operations using meta execution.

        Single tensors use a direct recipe, just like factory allocations.
        Structured outputs project a shared recipe; missing-meta/data-dependent
        kernels execute temporarily to determine output metadata and scalar values.
        """
        # Alias annotations describe storage sharing; is_write marks mutation.
        # Mutation should be handled explicitly in `self._write`
        # Reject remaining writes here rather than record them as pure operations.
        if any(arg.alias_info is not None and arg.alias_info.is_write
               for arg in func._schema.arguments):
            raise _unsupported(f"unsupported mutation {func}")

        # Alias-returning operators (like view/slice) should be handled in `self._view`.
        # note x.to(x.dtype) will be intercepted in higher level
        # and will not go to `__torch_dispatch__`
        if any(ret.alias_info is not None for ret in func._schema.returns):
            raise _unsupported(f"unsupported aliasing operation {func}")

        # Handle generator arguments by moving them to kwargs and truncating args.
        # because some functions accept the generator as a positional argument (like poisson),
        # we move it to kwargs to standardize handling.
        for index, arg in enumerate(func._schema.arguments):
            if arg.name == "generator" and index < len(args):
                kwargs.update((schema.name, value) for schema, value in
                              zip(func._schema.arguments[index:], args[index:]))
                args = args[:index]
                break

        random_op = torch.Tag.nondeterministic_seeded in func.tags
        seed = self._seed(name, kwargs, random_op=random_op)
        recipe_args = _map(args, self._snapshot)
        recipe_kwargs = _map(kwargs, self._snapshot)
        recipe_kwargs.pop("generator", None)
        node = _Node(None, func, recipe_args, recipe_kwargs, seed)

        meta_kwargs = _map(kwargs, self._to_meta)
        if "generator" in meta_kwargs:
            meta_kwargs["generator"] = None
        if any(arg.name == "device" for arg in func._schema.arguments):
            meta_kwargs["device"] = torch.device("meta")
        meta_args = _map(args, self._to_meta)

        reason = None
        if any(tag in func.tags for tag in (
            torch.Tag.data_dependent_output,   # .item
            torch.Tag.dynamic_output_shape,    # .nonzero/unique_dim
        )):
            reason = "operator output requires concrete data"
        else:
            try:
                if name == "randint_like" and isinstance(args[1], torch.Tensor):
                    # PyTorch 2.10's Tensor overload loses strides on meta.
                    # Match eager/replay layout without reading the bound.
                    meta_kwargs.pop("generator", None)
                    result = torch.empty_like(meta_args[0], **meta_kwargs)
                else:
                    result = func(*meta_args, **meta_kwargs)
            except NotImplementedError as exc:
                reason = str(exc)

        if reason is not None:
            _logger.warning("Deferred initialization concrete fallback for %s: %s", func, reason)
            result = _map(self._replay(node), self._to_meta)

        if isinstance(result, torch.Tensor):
            node.tensor = result
            return self._register(result, node)

        def register(value, path=()):
            if isinstance(value, (tuple, list)):
                return type(value)(register(item, path + (index,))
                                   for index, item in enumerate(value))
            if isinstance(value, dict):
                return {key: register(item, path + (key,)) for key, item in value.items()}
            if isinstance(value, torch.Tensor):
                projection = _Node(value, None, (), {}, operation=node, projection=path)
                return self._register(value, projection)
            return value

        return register(result)

    def _to_meta(self, value):
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise _unsupported("only strided inputs are supported")
            if value.is_meta:
                return value
            return torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device="meta")
        return value

    def _seed(self, name, kwargs, *, random_op=False):
        if name not in self._random and not random_op:
            return None
        generator = kwargs.get("generator")
        initial = torch.initial_seed() if generator is None else generator.initial_seed()
        return (initial + self._calls * 0x9E3779B97F4A7C15) % (2 ** 63)

    def _write(self, func, name, args, kwargs):
        tensor = args[0]
        storage = self._storage(tensor)
        if storage is None:
            raise _unsupported(f"mutation of external data in {func}")
        if not tensor.is_contiguous():
            raise _unsupported(f"noncontiguous mutation in {func}")

        # This is a support-scope restriction, not a single-chain limitation:
        # previous and _Read entries in args/kwargs already form a dependency DAG.
        # Tensor arithmetic needs broadcast, dtype, aliasing and historical-value
        # semantics validated before enabling it.
        if name in self._arithmetic_writes | self._unary_writes and any(
            isinstance(value, torch.Tensor) for value in (*args[1:], *kwargs.values())
        ):
            raise _unsupported(f"tensor-valued arithmetic in {func}")

        full = _layout(tensor) == _layout(storage.tensor)
        needs_previous = not full or name in self._arithmetic_writes | self._unary_writes

        # capture the arguments and keyword arguments for the node
        # so we can later replay the operation with the exact same arguments and keyword arguments.
        recipe_args = _map(args[1:], self._snapshot)
        recipe_kwargs = _map(kwargs, self._snapshot)
        seed = self._seed(name, kwargs)
        # remove the generator from the captured keyword arguments
        # replay replaces it with a private CPU generator using node.seed.
        recipe_kwargs.pop("generator", None)
        node = _Node(storage.tensor, func, recipe_args, recipe_kwargs, seed,
                     storage.node if needs_previous else None, _layout(tensor))

        # result is fake
        # so we replace the generator with None to avoid affecting the actual RNG state.
        meta_kwargs = dict(kwargs)
        if "generator" in meta_kwargs:
            meta_kwargs["generator"] = None
        result = func(*args, **meta_kwargs)
        storage.node = node

        return result

    def materialize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Replay just ``tensor`` and its dependencies, returning CPU data without a final copy.

        The captured recipes form a DAG, not just a chain of writes:

        - ``_Node`` directly records a factory, copy, write or single-output pure
          operator with its arguments, capture-time default dtype and random seed.
          Structured outputs instead project one shared recipe node through
          ``operation``/``projection``; that shared node has no tensor template.
          Concrete fallback uses the same recipes and is re-executed as needed.
        - ``_Read`` entries in recipe args/kwargs reference a specific producer
          node and a view layout (shape, stride, storage offset).
        - Empty allocations are ordinary nodes. All tensor arguments reference
          their original producers, even when an operator only needs their layout.
        - A mutation node's ``previous`` references the destination's old value
          when needed; ``target`` identifies the view to modify. Full overwrites
          can omit ``previous``.
        - ``_External`` inputs borrow real tensors and record their versions;
          they are data sources rather than replayable producer nodes.
        - Each ``_Storage.node`` points to that storage's latest captured value.
          Earlier reads keep their original node references across later writes.

        For example, ``x = ones(4); y = full((2,), 7); x[1:3].copy_(y)``
        produces these dependency edges::

            N1: ones(4) -------- previous -----+
                                              +--> N3: copy_ into x[1:3]
            N2: full((2,), 7) -- args[0]._Read -+

        After capture ends, snapshot the requested tensor and recursively resolve
        only its reachable dependencies. ``resolve`` reconstructs reads with
        ``as_strided`` and validates external versions before moving external
        inputs to CPU. ``execute`` resolves operator arguments, creates a private
        CPU generator or scoped CPU RNG state from the saved seed when present,
        and executes the operator under its captured default dtype.
        Mutation nodes allocate a new buffer, copy ``previous`` if required, then
        write through the target view; this leaves older node values unchanged.

        A per-call cache evaluates each shared node at most once. Replay runs
        under ``no_grad`` and clears the cache even on failure; separate calls
        recompute dependencies, including fallback operations.
        Selection is by node, not individual elements:
        requesting a small view may still materialize its full producer. No final
        clone is added, so returned views may retain larger storage and external
        CPU inputs may share storage with the result. The original meta tensor is
        not replaced or populated.
        """
        if self._active:
            raise _unsupported("materialize must be called after leaving capture")
        value = self._snapshot(tensor)
        return self._replay(value)

    def _replay(self, value):
        # Dispatch temporarily removes this mode while invoking its handler;
        # fallback can replay here without disabling other observers/modes.
        cache = {}

        def resolve(item: _Read | _External | Any):
            if isinstance(item, _Read):
                return evaluate(item.node).as_strided(*item.layout)
            if isinstance(item, _External):
                if item.tensor._version != item.version:
                    raise _unsupported("external tensor data changed after capture")
                return item.tensor.detach().to(device="cpu")
            if isinstance(item, _Node):
                return evaluate(item)
            return item

        def evaluate(node: _Node):
            if node in cache:
                return cache[node]
            if node.operation is not None:  # virtual node for a projected result
                result = evaluate(node.operation)
                for index in node.projection:
                    result = result[index]
            elif node.target is not None:  # inplace write node
                # allocate a full tensor for the inplace write (target is the location being written to)
                result = torch.empty_strided(node.tensor.shape, node.tensor.stride(),
                                             dtype=node.tensor.dtype, device="cpu")
                if node.previous is not None:
                    result.copy_(evaluate(node.previous))
                # inplace write to result
                execute(node, result.as_strided(*node.target))
            else:  # normal node
                result = execute(node)
            cache[node] = result
            return result

        def execute(node: _Node, target: Optional[torch.Tensor] = None):
            args = _map(node.args, resolve)
            kwargs = _map(node.kwargs, resolve)
            func = node.func
            name = func._schema.name.split("::")[-1]
            custom = not func._schema.name.startswith("aten::")
            accepts_generator = any(arg.name == "generator" for arg in func._schema.arguments)
            if node.seed is not None and (accepts_generator or name in self._random_factories):
                kwargs["generator"] = torch.Generator(device="cpu").manual_seed(node.seed)
            if target is not None:
                args = (target, *args)
            elif any(arg.name == "device" for arg in func._schema.arguments):
                kwargs["device"] = torch.device("cpu")

            # ATen uses private generators or the CPU RNG scope below. Custom
            # kernels may also use Python/NumPy/CUDA RNG. Avoid allocating RNG
            # snapshots for native deterministic operations and private generators.
            with _default_dtype(node.default_dtype), (
                (_initialization_rng(node.seed) if node.seed is not None else preserve_rng_state())
                if custom else nullcontext()
            ):
                if name in self._random_like:
                    # Use the Python API's overload selection, including tensor
                    # bounds, but support older *_like APIs without generators.
                    kwargs.pop("generator", None)
                    func = getattr(torch, name)
                elif name in self._random_factories:
                    func = getattr(torch, name)
                if not custom and node.seed is not None and (
                    name in self._random_like or "generator" not in kwargs
                ):
                    # use global RNG if the operation does not accept a generator
                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(node.seed)
                        return func(*args, **kwargs)
                return func(*args, **kwargs)

        try:
            with torch.no_grad():
                return resolve(value)
        except Exception as exc:
            raise _unsupported(f"replay failed: {exc}") from exc
        finally:
            cache.clear()
