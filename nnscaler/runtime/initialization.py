#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""In-memory initialization of generated modules.

Deferred initialization captures constructors without storage for selective,
CPU-only replay. It is not a general lazy tensor implementation. Only the operations
listed below are supported; data-dependent constructors must use file or model initialization.
Random operations use independent seeds, not eager PyTorch's random-number stream.
Existing real tensor inputs are borrowed, not copied, and must remain unchanged
until replay. ``torch.tensor`` literals allocate before dispatch; their real
storage is retained by the captured recipes, so these allocations are not deferred.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

if TYPE_CHECKING:
    from nnscaler.runtime.module import AttrMeta


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
            np.random.seed(seed)
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


def create_partial_init_weights(
    module_class: Type[torch.nn.Module],
    attr_meta_map: Dict[str, "AttrMeta"],
    *,
    seed: int,
) -> Dict[str, torch.Tensor]:
    initializer = getattr(module_class, '__partial__init__', None)
    if not callable(initializer):
        raise RuntimeError(
            "custom initialization requires the original module to define "
            "__partial__init__(attr_meta_map) as a staticmethod or classmethod."
        )
    with _initialization_rng(seed):
        return initializer(attr_meta_map)


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
class _Node:
    tensor: torch.Tensor  # Meta-only layout template, never a real parameter cache.
    func: Any
    args: Any
    kwargs: Any
    seed: Optional[int] = None
    mutation: bool = False
    previous: Optional["_Node"] = None
    target: Any = None


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

    Allocating operators produce meta tensors. Recipes retain only meta metadata
    and explicitly supplied external data. Each materialization has an ephemeral
    dependency cache; copied weights may require replaying their source weights,
    but unrelated parameters are never evaluated. Repeated calls are independent.

    Empty allocations are replayed without initializing their values, like eager
    PyTorch. Partial writes require a contiguous destination view. Arbitrary
    computations, overlapping layouts, tensor-valued arithmetic mutations, and
    value-dependent control flow are unsupported. Capture is single-use.
    Only initialization inside the context is replayed; later value mutations of
    captured meta tensors are ignored. Do not change their metadata afterwards.
    """

    _factories = {
        "empty", "empty_strided", "zeros", "ones", "full", "rand", "randn", "arange",
        "empty_like", "zeros_like", "ones_like", "full_like", "new_empty",
        "new_empty_strided", "new_zeros", "new_ones", "new_full",
        "rand_like", "randn_like", "randint", "randint_like", "randperm",
    }
    _views = {
        "detach", "alias", "view", "_unsafe_view", "transpose", "t", "permute",
        "slice", "select", "unsqueeze", "squeeze", "expand",
    }
    # _to_copy is aten name for tensor.to(...) when tensor copying is happening.
    _copies = {"clone", "_to_copy"}
    _arithmetic_writes = {"add_", "sub_", "mul_", "div_"}
    _random_writes = {
        "uniform_", "normal_", "random_", "bernoulli_", "exponential_",
        "geometric_", "log_normal_", "cauchy_",
    }
    _sampling = {"normal", "bernoulli", "poisson", "multinomial"}
    _random_like = {"rand_like", "randn_like", "randint_like"}
    _writes = {"fill_", "zero_", "copy_"} | _arithmetic_writes | _random_writes
    _random = {"rand", "randn", "randint", "randperm"} | _random_like | _random_writes | _sampling

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
        self._used = self._active = True
        return super().__enter__()

    def __exit__(self, *args):
        self._active = False
        return super().__exit__(*args)

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

    def __torch_dispatch__(self, func: torch._ops.OpOverload, types, args=(), kwargs=None):
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

        # Unlike other supported random operators, poisson accepts a positional generator.
        if name == "poisson" and len(args) > 1:
            kwargs["generator"] = args[1]
            args = args[:1]

        if name in self._views:
            return self._view(func, args, kwargs)
        if name in self._writes:
            return self._write(func, name, args, kwargs)
        if name == "lift_fresh":
            return self._lift_fresh(args[0])
        if name in self._factories or name in self._copies or name in self._sampling:
            return self._factory_copy(func, name, args, kwargs)

        raise _unsupported(f"unsupported operation {func}")

    def _view(self, func, args, kwargs):
        # Views share the producer's recipe with a different layout,
        # so no new initialization node is needed.
        storage = self._storage(args[0])
        if storage is None and args[0].is_meta:
            raise _unsupported(f"view of an untracked meta tensor in {func}")
        result = func(*args, **kwargs)
        if result.untyped_storage() is not args[0].untyped_storage():
            raise _unsupported(f"view {func} unexpectedly allocated new storage")
        return result

    def _lift_fresh(self, tensor):
        # torch.tensor literals already have real storage before dispatch.
        # Retain the literal's real data and replay it as a clone.
        source = self._snapshot(tensor)
        result = torch.empty_strided(tensor.shape, tensor.stride(),
                                     dtype=tensor.dtype, device="meta")
        node = _Node(result, torch.ops.aten.clone.default, (source,), {})
        return self._register(result, node)

    def _factory_copy(self, func, name, args, kwargs):
        seed = self._seed(name, kwargs)

        recipe_args = _map(args, self._snapshot)
        recipe_kwargs = _map(kwargs, self._snapshot)

        # Execute on meta for output metadata only; real allocation waits for replay.
        meta_kwargs = _map(kwargs, self._to_meta)
        if any(arg.name == "device" for arg in func._schema.arguments):
            meta_kwargs["device"] = torch.device("meta")
        if "generator" in meta_kwargs:
            # Meta execution must not use the caller's CPU/CUDA generator.
            meta_kwargs["generator"] = None
        meta_args = _map(args, self._to_meta)

        if name == "randint_like" and isinstance(args[1], torch.Tensor):
            # PyTorch 2.10's Tensor overload loses strides on meta. Match eager
            # and replay layout using empty_like without reading the bound.
            meta_kwargs.pop("generator", None)
            result = torch.empty_like(meta_args[0], **meta_kwargs)
        else:
            result = func(*meta_args, **meta_kwargs)

        # Replay creates a private CPU generator from seed, not the mutable original.
        recipe_kwargs.pop("generator", None)
        if any(arg.name == "dtype" for arg in func._schema.arguments):
            # The constructor may restore the default dtype before replay.
            recipe_kwargs["dtype"] = result.dtype
        node = _Node(result, func, recipe_args, recipe_kwargs, seed)
        return self._register(result, node)

    def _to_meta(self, value):
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise _unsupported("only strided inputs are supported")
            if value.is_meta:
                return value
            return torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device="meta")
        return value

    def _seed(self, name, kwargs):
        if name not in self._random:
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
        if name in self._arithmetic_writes and any(
            isinstance(value, torch.Tensor) for value in args[1:]
        ):
            raise _unsupported(f"tensor-valued arithmetic in {func}")

        full = _layout(tensor) == _layout(storage.tensor)
        needs_previous = not full or name in self._arithmetic_writes

        # capture the arguments and keyword arguments for the node
        # so we can later replay the operation with the exact same arguments and keyword arguments.
        recipe_args = _map(args[1:], self._snapshot)
        recipe_kwargs = _map(kwargs, self._snapshot)
        seed = self._seed(name, kwargs)
        # remove the generator from the captured keyword arguments
        # we will replace it with `torch.Generator(device="cpu").manual_seed(node.seed)`
        # when the node is actually evaluated.
        recipe_kwargs.pop("generator", None)
        node = _Node(storage.tensor, func, recipe_args, recipe_kwargs, seed, True,
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

        - ``_Node`` records an operator, its arguments, a meta output template,
          and an optional per-operation random seed.
        - ``_Read`` entries in node args/kwargs reference a specific producer
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
        inputs to CPU. ``evaluate`` resolves operator arguments, creates a private
        CPU generator from the saved seed when present, and executes the operator.
        Mutation nodes allocate a new buffer, copy ``previous`` if required, then
        write through the target view; this leaves older node values unchanged.

        A per-call cache evaluates each shared node at most once. Replay runs
        under ``no_grad`` and clears the cache even on failure; separate calls
        recompute dependencies. Selection is by node, not individual elements:
        requesting a small view may still materialize its full producer. No final
        clone is added, so returned views may retain larger storage and external
        CPU inputs may share storage with the result. The original meta tensor is
        not replaced or populated.
        """
        if self._active:
            raise _unsupported("materialize must be called after leaving capture")
        value = self._snapshot(tensor)
        cache = {}

        def resolve(item: _Read | _External | Any):
            if isinstance(item, _Read):
                return evaluate(item.node).as_strided(*item.layout)
            if isinstance(item, _External):
                if item.tensor._version != item.version:
                    raise _unsupported("external tensor data changed after capture")
                return item.tensor.detach().to(device="cpu")
            return item

        def evaluate(node: _Node):
            if node in cache:
                return cache[node]

            args = _map(node.args, resolve)
            kwargs = _map(node.kwargs, resolve)
            if node.seed is not None:
                kwargs["generator"] = torch.Generator(device="cpu").manual_seed(node.seed)

            if node.mutation:
                result = torch.empty_strided(node.tensor.shape, node.tensor.stride(),
                                             dtype=node.tensor.dtype, device="cpu")
                if node.previous is not None:
                    result.copy_(evaluate(node.previous))
                node.func(result.as_strided(*node.target), *args, **kwargs)
            else:
                name = node.func._schema.name.split("::")[-1]
                if any(arg.name == "device" for arg in node.func._schema.arguments):
                    kwargs["device"] = torch.device("cpu")

                # many random-like aten functions doesn't accept a generator directly
                if name in self._random_like:
                    # Older *_like APIs do not accept a generator.
                    kwargs.pop("generator")
                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(node.seed)
                        result = getattr(torch, name)(*args, **kwargs)
                elif name in {"rand", "randn", "randint", "randperm"}:
                    result = getattr(torch, name)(*args, **kwargs)
                else:
                    result = node.func(*args, **kwargs)
            cache[node] = result
            return result

        try:
            with torch.no_grad():
                return resolve(value)
        except Exception as exc:
            raise _unsupported(f"replay failed: {exc}") from exc
        finally:
            cache.clear()
