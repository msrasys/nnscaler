#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import gc
import random
import weakref
from contextlib import ExitStack

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack

from nnscaler.runtime.initialization import DeferredInitialization, PyFunction, preserve_rng_state


class AllocationRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.outputs = []
        self.random_shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if isinstance(result, torch.Tensor):
            self.outputs.append((result.device.type, weakref.ref(result)))
            if func in (torch.ops.aten.uniform_.default, torch.ops.aten.normal_.default):
                self.random_shapes.append(tuple(result.shape))
        return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("device", [None, "cuda", "cpu", "cuda:0"])
def test_materialize_device_and_rng(device):
    external = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    kwargs = {} if device is None else {"device": torch.device(device)}
    with DeferredInitialization(**kwargs) as capture:
        tensor = torch.empty(3, 4)
        tensor.normal_()
        tensor[:, ::2].add_(external[:, ::2])
        result = tensor + torch.rand_like(tensor)
        orthogonal = torch.empty(4, 4)
        torch.nn.init.orthogonal_(orthogonal)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    actual = capture.materialize(result)
    expected_device = torch.device("cpu") if device in (None, "cpu") else torch.device("cuda", torch.cuda.current_device())
    assert actual.device == expected_device
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, capture.materialize(result))
    matrix = capture.materialize(orthogonal)
    torch.testing.assert_close(matrix @ matrix.t(), torch.eye(4, device=matrix.device), atol=1e-5, rtol=1e-5)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert random.getstate() == python_rng
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_rng[0]
    assert np.array_equal(numpy_after[1], numpy_rng[1])
    assert numpy_after[2:] == numpy_rng[2:]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_materialize_explicit_cuda_device_restores_current_device():
    with torch.cuda.device(0):
        with DeferredInitialization(device="cuda:1") as capture:
            result = torch.rand_like(torch.ones(8))
        states = torch.cuda.get_rng_state_all()
        first = capture.materialize(result)
        second = capture.materialize(result)
        assert first.device == torch.device("cuda:1")
        assert torch.equal(first, second)
        assert torch.cuda.current_device() == 0
        assert all(torch.equal(before, after) for before, after in zip(states, torch.cuda.get_rng_state_all()))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_capture_pins_current_cuda_device():
    with torch.cuda.device(0):
        capture = DeferredInitialization(device="cuda")
    with torch.cuda.device(1):
        with capture:
            tensor = torch.randn(32)
            total = tensor.sum().item()
        actual = capture.materialize(tensor)
        assert actual.device == torch.device("cuda:0")
        assert actual.sum().item() == total
        assert torch.cuda.current_device() == 1


def test_capture_rejects_meta_device():
    with pytest.raises(RuntimeError, match="replay device must be CPU or CUDA"):
        DeferredInitialization(device="meta")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_materialize_after_capture_fallback(caplog):
    with DeferredInitialization(device="cuda") as capture:
        tensor = torch.randn(64)
        indices = (tensor > 0).nonzero()
    assert "concrete fallback" in caplog.text
    actual = capture.materialize(indices)
    assert actual.is_cuda
    assert actual.shape == indices.shape
    assert torch.equal(actual, (capture.materialize(tensor) > 0).nonzero())


def test_standard_constructors_are_meta_and_selective():
    with AllocationRecorder() as construction, DeferredInitialization() as capture:
        model = torch.nn.ModuleDict({
            "linear": torch.nn.Linear(7, 11),
            "norm": torch.nn.LayerNorm(11),
            "embedding": torch.nn.Embedding(19, 13, padding_idx=2),
        })
    assert construction.outputs
    assert all(device == "meta" for device, _ in construction.outputs)
    assert all(parameter.is_meta for parameter in model.parameters())
    with AllocationRecorder() as replay:
        weight = capture.materialize(model["linear"].weight)
    assert replay.random_shapes == [(11, 7)]
    assert weight.device.type == "cpu"
    assert weight.shape == (11, 7)
    assert (weight.abs() <= 7 ** -0.5).all()
    assert torch.equal(capture.materialize(model["norm"].weight), torch.ones(11))
    assert torch.equal(capture.materialize(model["norm"].bias), torch.zeros(11))
    embedding = capture.materialize(model["embedding"].weight)
    assert torch.equal(embedding[2], torch.zeros(13))
    assert embedding[3].abs().sum() > 0


def test_no_retained_real_cache_or_shared_results():
    with DeferredInitialization() as capture:
        parameter = torch.nn.Parameter(torch.ones(8))
    with AllocationRecorder() as recorder:
        first = capture.materialize(parameter)
    first.fill_(12)
    second = capture.materialize(parameter)
    assert torch.equal(second, torch.ones(8))
    reference = weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None
    assert all(reference() is None for _, reference in recorder.outputs)
    assert all(storage.tensor.is_meta for storage in capture._storages.values())


def test_previous_full_parameter_storage_expires_before_next_call():
    from torch.multiprocessing.reductions import StorageWeakRef

    with DeferredInitialization() as capture:
        model = torch.nn.Sequential(*[
            torch.nn.Linear(16, 16) for _ in range(3)
        ]).double()
        snapshots = [parameter.detach().clone() for parameter in model.parameters()]
    tensors = list(model.parameters()) + snapshots
    assert all(tensor.is_meta for tensor in tensors)
    for tensor in tensors:
        value = capture.materialize(tensor)
        storage = StorageWeakRef(value.untyped_storage())
        assert not storage.expired()
        del value
        # Refcounting must release storage immediately, without a gc.collect().
        assert storage.expired()
    assert all(tensor.is_meta for tensor in tensors)


@pytest.mark.parametrize("factory", [
    pytest.param(lambda: torch.ones(3, 4), id="factory"),
    pytest.param(lambda: torch.tensor([[1., 2.], [3., 4.]]), id="literal"),
])
def test_versions_shared_parameters_and_partial_writes(factory):
    initial = factory()
    with DeferredInitialization() as capture:
        module = torch.nn.Module()
        module.first = torch.nn.Parameter(factory())
        module.second = module.first
        with torch.no_grad():
            copied_before = module.first.clone()
            module.first[1].fill_(5)
            copied_after = torch.empty_like(module.first)
            copied_after.copy_(module.first)
            module.second.mul_(2)
            module.first.add_(3)
    assert module.first is module.second
    expected = initial.clone()
    expected[1] = 5
    assert torch.equal(capture.materialize(copied_before), initial)
    assert torch.equal(capture.materialize(copied_after), expected)
    assert torch.equal(capture.materialize(module.first), expected * 2 + 3)
    assert torch.equal(capture.materialize(module.second), expected * 2 + 3)


@pytest.mark.parametrize("operation,kwargs,dtype,partial", [
    pytest.param(operation, kwargs, dtype, partial, id=f"{operation}-{kwargs}-{dtype}-{partial}")
    for operation, kwargs, dtypes in [
        ("add_", {"alpha": 3}, [torch.float16, torch.float32, torch.float64]),
        ("sub_", {"alpha": 3}, [torch.float16, torch.float32, torch.float64]),
        ("mul_", {}, [torch.float16, torch.float32, torch.float64]),
        ("div_", {}, [torch.float16, torch.float32, torch.float64]),
        ("div_", {"rounding_mode": "floor"}, [torch.int64, torch.float32]),
        ("div_", {"rounding_mode": "trunc"}, [torch.int64, torch.float32]),
    ]
    for dtype in dtypes
    for partial in ([False] if "rounding_mode" in kwargs else [False, True])
])
def test_scalar_arithmetic_preserves_previous_values(operation, kwargs, dtype, partial):
    def initialize():
        tensor = torch.arange(-5, 5, dtype=dtype)
        before = tensor.clone()
        target = tensor[1:6] if partial else tensor
        getattr(target, operation)(2, **kwargs)
        after = target.clone()
        return tensor, before, after

    expected = initialize()
    with DeferredInitialization() as capture:
        actual = initialize()
    for tensor, reference in zip(actual, expected):
        torch.testing.assert_close(capture.materialize(tensor), reference, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["add_", "sub_", "mul_", "div_", "clamp_"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("operand", ["broadcast", "transposed", "self", "same-view", "disjoint", "empty"])
def test_tensor_arithmetic_preserves_aliases_and_history(operation, dtype, operand):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(1, 17, dtype=dtype).reshape(4, 4).clone())
            with torch.no_grad():
                source = torch.linspace(0.5, 2.5, 8, dtype=torch.float32).reshape(4, 2)
                target = self.weight[1:3]
                if operand == "broadcast":
                    rhs = source[:1].t()
                elif operand == "transposed":
                    rhs = source.t()
                elif operand == "self":
                    rhs = target
                elif operand == "same-view":
                    rhs = self.weight[1:3].view(2, 4)
                elif operand == "disjoint":
                    target, rhs = self.weight[:2], self.weight[2:]
                else:
                    target, rhs = self.weight[:0], self.weight[:0]
                self.before = self.weight.clone()
                if operation == "clamp_":
                    target.clamp_(min=rhs, max=torch.full((), 12.))
                else:
                    kwargs = {"alpha": 0.7} if operation in ("add_", "sub_") else {}
                    getattr(target, operation)(rhs, **kwargs)
                self.after = self.weight.clone()
                source.fill_(99)
                self.weight.add_(1)
                self.rhs = rhs

    expected = Model()
    with DeferredInitialization() as capture:
        actual = Model()
    for name in ("weight", "before", "after", "rhs"):
        torch.testing.assert_close(
            capture.materialize(getattr(actual, name)), getattr(expected, name), rtol=0, atol=0,
        )


@pytest.mark.parametrize("operation", ["add_", "sub_", "mul_", "div_", "clamp_"])
@pytest.mark.parametrize("overlap", ["partial", "strided", "broadcast", "transpose", "stride-gaps"])
def test_tensor_arithmetic_rejects_overlapping_operands(operation, overlap):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(1., 17.).reshape(4, 4).clone())
            with torch.no_grad():
                if overlap == "partial":
                    target, rhs = self.weight[1:], self.weight[:-1]
                elif overlap == "strided":
                    target, rhs = self.weight.flatten()[2:10], self.weight.flatten()[::2]
                elif overlap == "broadcast":
                    target, rhs = self.weight, self.weight[:1]
                elif overlap == "transpose":
                    target, rhs = self.weight, self.weight.t()
                else:
                    target, rhs = self.weight.flatten()[1:3], self.weight.flatten()[::8]
                if operation == "clamp_":
                    target.clamp_(max=rhs)
                else:
                    getattr(target, operation)(rhs)

    with pytest.raises(RuntimeError, match="overlapping tensor operand"):
        with DeferredInitialization():
            Model()


@pytest.mark.parametrize("operation", ["add_", "sub_", "mul_", "div_"])
@pytest.mark.parametrize("invalid", ["broadcast", "dtype"])
def test_tensor_arithmetic_rejects_invalid_inputs(operation, invalid):
    def initialize():
        target = torch.ones(3, dtype=torch.int64 if invalid == "dtype" else torch.float32)
        rhs = torch.full((3 if invalid == "dtype" else 4,), 0.5)
        getattr(target, operation)(rhs)
        return target

    with pytest.raises(RuntimeError):
        initialize()
    with pytest.raises(RuntimeError):
        with DeferredInitialization() as capture:
            target = initialize()
        capture.materialize(target)


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_tensor_integer_division(rounding_mode):
    def initialize():
        target = torch.arange(-5, 5)
        target.div_(torch.full((1,), 2), rounding_mode=rounding_mode)
        return target

    expected = initialize()
    with DeferredInitialization() as capture:
        actual = initialize()
    torch.testing.assert_close(capture.materialize(actual), expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["add_", "sub_", "mul_", "div_"])
def test_arithmetic_on_empty_preserves_allocation_dependency(operation):
    with DeferredInitialization() as capture:
        tensor = torch.empty(3)
        allocation = capture._storage(tensor).node
        getattr(tensor, operation)(2)
    assert capture._storage(tensor).node.previous is allocation
    actual = capture.materialize(tensor)
    assert actual.shape == (3,)
    assert actual.dtype == tensor.dtype
    assert actual.device.type == "cpu"


def test_noncontiguous_read_and_post_copy_source_mutation():
    with DeferredInitialization() as capture:
        source = torch.arange(12, dtype=torch.float32).view(3, 4)
        transposed = source.t()
        destination = torch.empty(4, 3)
        destination.copy_(transposed)
        sliced = source[:, ::2].clone()
        source.fill_(99)
    expected = torch.arange(12, dtype=torch.float32).view(3, 4)
    assert capture._storages[source.untyped_storage()] is capture._storages[transposed.untyped_storage()]
    assert capture._storages[destination.untyped_storage()] is not capture._storages[source.untyped_storage()]
    assert torch.equal(capture.materialize(destination), expected.t())
    assert torch.equal(capture.materialize(sliced), expected[:, ::2])
    assert torch.equal(capture.materialize(transposed), torch.full((4, 3), 99.0))


# Each case specifies the initial shape and its metadata-only mutation.
@pytest.mark.parametrize("shape, operation", [
    pytest.param((3, 4), lambda x: x.transpose_(0, 1), id="transpose"),
    pytest.param((3, 4), lambda x: x.t_(), id="t"),
    pytest.param((3, 4), lambda x: x.swapaxes_(0, 1), id="swapaxes"),
    pytest.param((3, 4), lambda x: x.swapdims_(0, 1), id="swapdims"),
    pytest.param((3, 1, 4), lambda x: x.squeeze_(), id="squeeze"),
    pytest.param((3, 1, 4), lambda x: x.squeeze_(1), id="squeeze-dim"),
    pytest.param((1, 3, 1, 4), lambda x: x.squeeze_((0, 2)), id="squeeze-dims"),
    pytest.param((3, 4), lambda x: x.unsqueeze_(1), id="unsqueeze"),
    pytest.param((3, 4), lambda x: x.as_strided_((2, 3), (4, 1), 1), id="as-strided"),
])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
])
def test_inplace_views_preserve_history_and_aliases(shape, operation, device):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # clone() makes the captured producer also be the tensor whose metadata is mutated.
            source = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device)
            self.register_buffer("weight", source.reshape(shape).clone())
            self.weight.add_(1)
            self.alias = self.weight.detach()
            self.before = self.weight.clone()
            assert operation(self.weight) is self.weight
            self.after_view = self.weight.clone()
            self.weight[0].add_(10)
            self.after_write = self.weight.clone()
            self.alias[-1].fill_(-5)

    expected = Model()
    with DeferredInitialization(device) as capture:
        model = Model()
    for name in ("weight", "alias", "before", "after_view", "after_write"):
        reference = getattr(expected, name)
        for _ in range(2):
            actual = capture.materialize(getattr(model, name))
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert actual.stride() == reference.stride()
            assert actual.storage_offset() == reference.storage_offset()

    external = torch.empty(shape, device=device)
    version = external._version
    with DeferredInitialization(device), pytest.raises(RuntimeError, match="mutation of external data"):
        operation(external)
    assert external.shape == shape
    assert external._version == version


# Each case gives the mutation and the first newly allocated element to initialize, if any.
@pytest.mark.parametrize("operation, initialize_from", [
    pytest.param(lambda x, y: x.resize_(3, 2), None, id="resize-shape"),
    pytest.param(lambda x, y: x.resize_(3, 4), 6, id="resize-grow"),
    pytest.param(lambda x, y: x.resize_(3), None, id="resize-shrink"),
    pytest.param(lambda x, y: x.resize_(0).resize_(3, 4), 6, id="resize-regrow"),
    pytest.param(lambda x, y: x.resize_as_(y), 6, id="resize-as-grow"),
    pytest.param(lambda x, y: x.resize_as_(y[:, :2]), None, id="resize-as-shape"),
    pytest.param(lambda x, y: x[1:].resize_(3, 3), 3, id="resize-offset"),
    pytest.param(lambda x, y: x.t_().resize_(3, 4), 6, id="resize-transposed"),
    pytest.param(lambda x, y: x.set_(source=y), None, id="set-tensor"),
    pytest.param(lambda x, y: x.set_(y.t()), None, id="set-view"),
    pytest.param(lambda x, y: x.set_(y, 1, (2, 3), (4, 1)), None, id="set-tensor-offset"),
    pytest.param(lambda x, y: x.set_(y.untyped_storage()), None, id="set-storage"),
    pytest.param(lambda x, y: x.set_(y.untyped_storage(), 1, (2, 3), (4, 1)), None, id="set-storage-offset"),
    pytest.param(lambda x, y: x.set_(y.untyped_storage(), 4, (4, 4), (4, 1)), 8, id="set-grow"),
    pytest.param(lambda x, y: x.set_(), None, id="set-empty"),
    pytest.param(lambda x, y: x.set_().resize_as_(y), 0, id="set-empty-grow"),
    pytest.param(lambda x, y: x.set_(x), None, id="set-self"),
    pytest.param(
        lambda x, y: torch.empty(0, dtype=torch.int64, device=x.device).set_(
            y.untyped_storage(), 0, (8,), (1,),
        ),
        6, id="set-reinterpret-grow",
    ),
])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
])
def test_storage_mutations_preserve_history_and_aliases(operation, initialize_from, device):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.arange(6., device=device).reshape(2, 3).clone())
            self.register_buffer("source", (torch.arange(12., device=device) + 30).reshape(3, 4).clone())
            self.weight.add_(1)
            self.old_alias = self.weight.detach()
            self.before = self.weight.clone()
            self.source_before = self.source.clone()
            self.weight = operation(self.weight, self.source)
            if initialize_from is not None:
                self.weight.view(-1)[initialize_from:].fill_(7)
            if self.weight.numel():
                self.weight[0].fill_(11)
            self.after = self.weight.clone()
            self.old_alias[0].fill_(13)
            self.source[0].fill_(17)

    expected = Model()
    with DeferredInitialization(device) as capture:
        model = Model()
    for name in ("weight", "source", "old_alias", "before", "source_before", "after"):
        reference = getattr(expected, name)
        for _ in range(2):
            actual = capture.materialize(getattr(model, name))
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert actual.stride() == reference.stride()
            assert actual.storage_offset() == reference.storage_offset()


def test_set_rejects_uncaptured_storage():
    external = torch.arange(6.)
    untracked_meta = torch.empty(6, device="meta")
    version = external._version
    with DeferredInitialization() as capture:
        target = torch.ones(3)
        for source in (external, external.untyped_storage(), untracked_meta):
            with pytest.raises(RuntimeError, match="set_ source must have captured storage"):
                target.set_(source)
        with pytest.raises(RuntimeError, match="mutation of external data"):
            external.set_(target)
    assert external._version == version
    assert torch.equal(external, torch.arange(6.))
    assert torch.equal(capture.materialize(target), torch.ones(3))


@pytest.mark.parametrize("shape,dim", [
    ((3, 4), 0),
    ((3, 4), 1),
    ((3, 4), -1),
    ((4,), 0),
    ((0, 4), 0),
    ((3, 0), 0),
])
def test_unbind_views_share_recipe(shape, dim):
    expected = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    with DeferredInitialization() as capture:
        source = torch.arange(expected.numel(), dtype=expected.dtype).reshape(shape)
        node = capture._storage(source).node
        parts = source.unbind(dim=dim)
        assert capture._storage(source).node is node
        assert all(part.is_meta and capture._storage(part) is capture._storage(source)
                   for part in parts)
    references = expected.unbind(dim)
    assert len(parts) == len(references)
    for part, reference in zip(parts, references):
        torch.testing.assert_close(capture.materialize(part), reference, rtol=0, atol=0)


def test_unbind_views_preserve_mutation_history():
    def initialize():
        source = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        first, second, third = source[1:4].unbind()
        before = first.clone()
        first.fill_(7)
        after = source.clone()
        second.copy_(third)
        source.add_(2)
        return source, first, second, third, before, after

    expected = initialize()
    with DeferredInitialization() as capture:
        actual = initialize()
    for tensor, reference in zip(actual, expected):
        torch.testing.assert_close(capture.materialize(tensor), reference, rtol=0, atol=0)


def test_unbind_noncontiguous_writes_preserve_history():
    with DeferredInitialization() as capture:
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        column = source.unbind(1)[2]
        before = column.clone()
        column.fill_(7)
        after = source.clone()
        source.fill_(9)
    assert torch.equal(capture.materialize(before), torch.tensor([2., 6., 10.]))
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected[:, 2] = 7
    assert torch.equal(capture.materialize(after), expected)
    assert torch.equal(capture.materialize(column), torch.full((3,), 9.))


@pytest.mark.parametrize("layout", ["transpose", "column", "strided", "empty"])
@pytest.mark.parametrize("operation", ["fill_", "copy_", "add_", "clamp_"])
def test_noncontiguous_writes_match_eager(layout, operation):
    def initialize():
        tensor = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        if layout == "transpose":
            target = tensor.transpose(0, 1)
        elif layout == "column":
            target = tensor[:, :, 0]
        elif layout == "strided":
            target = tensor[:, ::2, :]
        else:
            target = tensor[:0].transpose(0, 1)
        before = tensor.clone()
        if operation == "fill_":
            target.fill_(7)
        elif operation == "clamp_":
            target.clamp_(max=torch.full((), 7.))
        else:
            getattr(target, operation)(torch.full(target.shape, 7.))
        return tensor, before

    expected = initialize()
    with DeferredInitialization() as capture:
        actual = initialize()
    for tensor, reference in zip(actual, expected):
        torch.testing.assert_close(capture.materialize(tensor), reference, rtol=0, atol=0)


def test_unbind_external_views_preserve_snapshot():
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    with DeferredInitialization() as capture:
        parts = source.unbind(1)
        copied = parts[2].clone()
    assert all(part.untyped_storage() is source.untyped_storage() for part in parts)
    assert torch.equal(capture.materialize(copied), source[:, 2])
    source.add_(1)
    with pytest.raises(RuntimeError, match="external tensor data changed"):
        capture.materialize(copied)


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
])
@pytest.mark.parametrize("stagewise", [False, True])
def test_unbind_tensor_iteration_in_constructor(device, stagewise):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            rates = torch.linspace(0, 0.2, 3, device="cpu")
            if stagewise:
                schedule = [value for stage in rates.split([1, 2]) for value in stage.tolist()]
            else:
                schedule = [value.item() for value in rates]
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in schedule])
            with torch.no_grad():
                for layer, value in zip(self.layers, schedule):
                    layer.weight.fill_(value)

    expected = Model()
    with DeferredInitialization(device=device) as capture:
        actual = Model()
    for layer, reference in zip(actual.layers, expected.layers):
        assert torch.equal(capture.materialize(layer.weight).cpu(), reference.weight)


@pytest.mark.parametrize("sections", [2, [2, 3], [0, 5, 0]])
def test_split_views_preserve_mutation_history(sections):
    with DeferredInitialization() as capture:
        source = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        parts = source.split(sections)
        before = [part.clone() for part in parts]
        parts[1].fill_(7)
    expected = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    for part, reference in zip(before, expected.split(sections)):
        assert torch.equal(capture.materialize(part), reference)
    expected.split(sections)[1].fill_(7)
    assert torch.equal(capture.materialize(source), expected)


@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
@pytest.mark.parametrize("fail", [False, True])
def test_tolist_fallback_and_restore(shape, fail):
    original = torch.Tensor.tolist
    try:
        with DeferredInitialization() as capture:
            value = torch.randn(shape)
            result = value.tolist()
            assert torch.Tensor.tolist is not original
            enclosing = torch.Tensor.tolist
            with DeferredInitialization():
                pass
            assert torch.Tensor.tolist is enclosing
            if fail:
                raise ValueError("exit capture")
    except ValueError as exc:
        assert fail and str(exc) == "exit capture"
    assert torch.Tensor.tolist is original
    assert result == capture.materialize(value).tolist()


@pytest.mark.parametrize("factory", [
    pytest.param(lambda x: torch.zeros_like(x), id="zeros-like"),
    pytest.param(lambda x: torch.rand_like(x), id="rand-like"),
    pytest.param(lambda x: torch.randint_like(x, 7), id="randint-like"),
    pytest.param(lambda x: torch.ops.aten.bernoulli.p(x, 0.3), id="bernoulli-p"),
])
@pytest.mark.parametrize("source", ["empty", "initialized", "external"])
def test_factory_inputs_replay_original_producers(factory, source):
    external = torch.ones(4, 8).t()
    with AllocationRecorder() as construction, DeferredInitialization() as capture:
        if source == "empty":
            template = torch.empty(4, 8).t()
        elif source == "initialized":
            template = torch.empty(4, 8).normal_().t()
        else:
            template = external
        tensor = factory(template)
    assert all(device == "meta" for device, _ in construction.outputs)
    template_read = capture._storage(tensor).node.args[0]
    if source == "external":
        assert template_read.tensor is external
    else:
        assert template_read.node is capture._storage(template).node
        assert template_read.node.tensor.is_meta
    with AllocationRecorder() as replay:
        actual = capture.materialize(tensor)
    assert replay.random_shapes == ([(4, 8)] if source == "initialized" else [])
    assert all(device == "cpu" for device, _ in replay.outputs)
    assert actual.stride() == tensor.stride()
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, capture.materialize(tensor))
    del actual
    gc.collect()
    assert all(reference() is None for _, reference in replay.outputs)
    if source == "external":
        external.fill_(float("nan"))
        with pytest.raises(RuntimeError, match="external tensor data changed"):
            capture.materialize(tensor)


def test_copy_dependency_is_not_retroactively_reinitialized():
    with DeferredInitialization() as capture:
        source = torch.empty(6)
        source.uniform_(-2, 3)
        before = source.clone()
        destination = torch.empty(6)
        destination.copy_(source)
        source.normal_(7, 0.2)
    assert torch.equal(capture.materialize(destination), capture.materialize(before))
    assert not torch.equal(capture.materialize(destination), capture.materialize(source))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
def test_random_distribution_parameters_and_dtype(dtype):
    with DeferredInitialization() as capture:
        uniform = torch.empty(20000, dtype=dtype).uniform_(2, 5)
        normal = torch.empty(20000, dtype=dtype).normal_(7, 2)
    actual_uniform = capture.materialize(uniform)
    actual_normal = capture.materialize(normal)
    assert actual_uniform.dtype == actual_normal.dtype == dtype
    assert actual_uniform.min() >= 2
    assert actual_uniform.max() <= 5
    assert abs(actual_normal.float().mean().item() - 7) < 0.1
    assert abs(actual_normal.float().std().item() - 2) < 0.1


@pytest.mark.parametrize("factory", [
    lambda: torch.zeros(3, 4),
    lambda: torch.ones(3, 4, dtype=torch.int64),
    lambda: torch.full((3, 4), 2.5, dtype=torch.float64),
    lambda: torch.arange(2, 11, 2, dtype=torch.float64),
    lambda: torch.tensor(2.5).to(torch.float64),
    lambda: torch.empty(3, 4).fill_(2),
    lambda: torch.empty(3, 4).zero_(),
    lambda: torch.ones(3, 4).to(torch.float64),
    lambda: torch.ones(3, 4).t().clone(),
    lambda: torch.zeros_like(torch.empty(3, 4)),
    lambda: torch.ones_like(torch.empty(3, 4), dtype=torch.float64),
    lambda: torch.full_like(torch.empty(3, 4), 3),
    lambda: torch.empty(1).new_full((3, 4), 3),
])
def test_factories_and_conversions(factory):
    expected = factory()
    with DeferredInitialization() as capture:
        actual = factory()
    assert actual.is_meta
    assert capture._storage(actual).node.operation is None
    materialized = capture.materialize(actual)
    assert materialized.dtype == expected.dtype
    assert torch.equal(materialized, expected)


@pytest.mark.parametrize("factory", [
    lambda: torch.full((4,), 1.00000001),
    lambda: torch.empty(4).fill_(1.00000001),
    lambda: torch.arange(0, 1, 0.1),
    lambda: torch.full((4,), 1),
    lambda: torch.full((4,), 1 + 2j),
    lambda: torch.full((4,), 1.00000001, dtype=torch.float32),
    lambda: torch.full_like(torch.empty(4, dtype=torch.float32), 1.00000001),
    pytest.param(lambda: torch.arange(4).sin(), id="generic-sin"),
])
def test_factory_dtype_is_independent_of_replay_default(factory):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        expected = factory()
        with DeferredInitialization() as capture:
            tensor = factory()
        torch.set_default_dtype(torch.float32)
        actual = capture.materialize(tensor)
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("data", [2, [], [1], [1, 2, 3], [[1, 2], [3, 4]]])
@pytest.mark.parametrize("dtype", [
    torch.bool, torch.int64, torch.float32, torch.float64, torch.complex64,
])
def test_tensor_literals(data, dtype):
    expected = torch.tensor(data, dtype=dtype)
    with DeferredInitialization() as capture:
        tensor = torch.tensor(data, dtype=dtype)
    assert tensor.is_meta
    first = capture.materialize(tensor)
    assert first.device.type == "cpu"
    assert first.shape == expected.shape
    assert first.dtype == expected.dtype
    assert torch.equal(first, expected)
    first.zero_()
    assert torch.equal(capture.materialize(tensor), expected)


@pytest.mark.parametrize("explicit_generator", [False, True])
def test_rng_determinism_private_generators_and_capture_state(explicit_generator):
    generator = torch.Generator() if explicit_generator else None
    rng = generator if generator is not None else torch.random.default_generator

    def factory():
        return [
            torch.rand(7, generator=generator),
            torch.randn(7, generator=generator),
            torch.empty(7).normal_(2, 3, generator=generator),
            torch.empty(7).uniform_(generator=generator),
        ]

    with torch.random.fork_rng(devices=[]):
        rng.manual_seed(923)
        state = rng.get_state()
        global_state = torch.get_rng_state()
        with DeferredInitialization() as first_capture:
            first = factory()
        assert torch.equal(rng.get_state(), state)
        with DeferredInitialization() as second_capture:
            second = factory()
        for a, b in zip(reversed(first), reversed(second)):
            expected = first_capture.materialize(a)
            assert torch.equal(rng.get_state(), state)
            assert torch.equal(second_capture.materialize(b), expected)
        assert not torch.equal(first_capture.materialize(first[0]), first_capture.materialize(first[-1]))
        assert torch.equal(torch.get_rng_state(), global_state)
        rng.manual_seed(924)
        with DeferredInitialization() as third_capture:
            third = factory()
        # The captures use different RNG seeds (923 vs 924), so replayed random values should differ.
        assert not torch.equal(first_capture.materialize(first[0]), third_capture.materialize(third[0]))


@pytest.mark.parametrize("factory", [
    pytest.param(lambda g: torch.randint(7, (32,), generator=g), id="randint-high"),
    pytest.param(lambda g: torch.randint(-4, 7, (32,), generator=g), id="randint-low-high"),
    pytest.param(lambda g: torch.randperm(32, generator=g), id="randperm"),
    pytest.param(lambda g: torch.normal(2., 3., (32,), generator=g), id="normal-scalars"),
    pytest.param(lambda g: torch.normal(torch.full((32,), 2.), 3., generator=g), id="normal-mean"),
    pytest.param(lambda g: torch.normal(2., torch.full((32,), 3.), generator=g), id="normal-std"),
    pytest.param(lambda g: torch.normal(torch.full((32,), 2.), torch.full((32,), 3.),
                                       generator=g), id="normal-tensors"),
    pytest.param(lambda g: torch.bernoulli(torch.full((32,), 0.3), generator=g), id="bernoulli"),
    pytest.param(lambda g: torch.ops.aten.bernoulli.p(torch.empty(32), 0.3, generator=g),
                 id="bernoulli-scalar"),
    pytest.param(lambda g: torch.poisson(torch.full((32,), 2.), g), id="poisson-positional"),
    pytest.param(lambda g: torch.multinomial(torch.ones(10), 20, True, generator=g),
                 id="multinomial-replacement"),
    pytest.param(lambda g: torch.multinomial(torch.ones(10), 5, False, generator=g),
                 id="multinomial-no-replacement"),
    pytest.param(lambda g: torch.empty(32, dtype=torch.int64).random_(generator=g), id="random"),
    pytest.param(lambda g: torch.empty(32).random_(7, generator=g), id="random-high"),
    pytest.param(lambda g: torch.empty(32).random_(-4, 7, generator=g), id="random-low-high"),
    pytest.param(lambda g: torch.empty(32).bernoulli_(0.3, generator=g), id="bernoulli-inplace"),
    pytest.param(lambda g: torch.empty(32).bernoulli_(torch.full((32,), 0.3), generator=g),
                 id="bernoulli-inplace-tensor"),
    pytest.param(lambda g: torch.empty(32).exponential_(2., generator=g), id="exponential"),
    pytest.param(lambda g: torch.empty(32).geometric_(0.3, generator=g), id="geometric"),
    pytest.param(lambda g: torch.empty(32).log_normal_(0.2, 0.5, generator=g), id="log-normal"),
    pytest.param(lambda g: torch.empty(32).cauchy_(1., 2., generator=g), id="cauchy"),
])
@pytest.mark.parametrize("explicit_generator", [False, True])
def test_extended_random_replay(factory, explicit_generator):
    generator = torch.Generator().manual_seed(132) if explicit_generator else None
    before = torch.get_rng_state()
    generator_state = generator.get_state() if generator is not None else None
    with DeferredInitialization() as capture:
        tensor = factory(generator)
    assert tensor.is_meta
    seed = capture._storage(tensor).node.seed
    expected = factory(torch.Generator().manual_seed(seed))
    first = capture.materialize(tensor)
    assert torch.equal(first, expected)
    assert torch.equal(capture.materialize(tensor), expected)
    assert first.dtype == expected.dtype
    assert first.shape == expected.shape
    assert first.device.type == "cpu"
    assert torch.equal(torch.get_rng_state(), before)
    if generator is not None:
        assert torch.equal(generator.get_state(), generator_state)


@pytest.mark.parametrize("factory,dtype", [
    pytest.param(factory, dtype, id=f"{name}-{dtype}")
    for name, factory, dtypes in [
        ("rand-like", torch.rand_like, [torch.float32, torch.float64, torch.complex64]),
        ("randn-like", torch.randn_like, [torch.float32, torch.float64, torch.complex64]),
        ("randint-like-high", lambda x: torch.randint_like(x, 7), [torch.float32, torch.float64]),
        ("randint-like-low-high", lambda x: torch.randint_like(x, -4, 7), [torch.float32, torch.float64]),
    ]
    for dtype in dtypes
])
def test_random_like_accepts_empty_template(factory, dtype):
    with DeferredInitialization() as capture:
        tensor = factory(torch.empty(4, 8, dtype=dtype).t())
    seed = capture._storage(tensor).node.seed
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        expected = factory(torch.empty(4, 8, dtype=dtype).t())
    state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    actual = capture.materialize(tensor)
    assert actual.stride() == expected.stride()
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert torch.equal(capture.materialize(tensor), expected)
    assert torch.equal(torch.get_rng_state(), state)
    if cuda_states:
        assert all(torch.equal(before, after) for before, after in
                   zip(cuda_states, torch.cuda.get_rng_state_all()))


@pytest.mark.parametrize("name", ["rand_like", "randn_like", "randint_like"])
def test_random_like_replay_failure_restores_rng(name, monkeypatch):
    factory = getattr(torch, name)
    with DeferredInitialization() as capture:
        template = torch.empty(8)
        tensor = factory(template, 7) if name == "randint_like" else factory(template)
    state = torch.get_rng_state()

    def fail_after_sampling(*args, **kwargs):
        assert "generator" not in kwargs
        factory(*args, **kwargs)
        raise RuntimeError("random-like replay failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch, name, fail_after_sampling)
        with pytest.raises(RuntimeError, match="random-like replay failure"):
            capture.materialize(tensor)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(capture.materialize(tensor), capture.materialize(tensor))
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.skipif(
    not hasattr(torch.ops.aten.randint_like, "Tensor"),
    reason="requires the tensor upper-bound overload",
)
@pytest.mark.parametrize("source,explicit_generator", [
    (source, explicit_generator)
    for source in ["literal", "computed", "external"]
    for explicit_generator in [False, True]
] + [("partial", False)])
def test_randint_like_tensor_bound(source, explicit_generator):
    external = torch.tensor(7)
    generator = torch.Generator().manual_seed(132) if explicit_generator else None
    state = generator.get_state() if generator is not None else torch.get_rng_state()
    with DeferredInitialization() as capture:
        if source == "literal":
            high = torch.tensor(7)
        elif source == "computed":
            high = torch.full((), 3, dtype=torch.int64).add_(4)
        elif source == "partial":
            bounds = torch.empty(2, dtype=torch.int64)
            bounds[0].fill_(7)
            high = bounds[0]
        else:
            high = external
        template = torch.empty(4, 8, dtype=torch.int64).t()
        sample = torch.randint_like(template, high, generator=generator)
        if source != "external":
            high.fill_(2)
    seed = capture._storage(sample).node.seed
    expected = torch.randint_like(
        torch.empty(4, 8, dtype=torch.int64).t(), torch.tensor(7),
        generator=torch.Generator().manual_seed(seed),
    )
    actual = capture.materialize(sample)
    assert actual.device.type == "cpu"
    assert ((actual >= 0) & (actual < 7)).all()
    assert torch.equal(actual, expected)
    assert actual.stride() == expected.stride()
    assert torch.equal(capture.materialize(sample), expected)
    assert torch.equal(generator.get_state() if generator is not None else torch.get_rng_state(), state)
    if source == "external":
        external.fill_(2)
        with pytest.raises(RuntimeError, match="external tensor data changed"):
            capture.materialize(sample)


@pytest.mark.parametrize("sample", [
    pytest.param(torch.bernoulli, id="probability-input"),
    pytest.param(
        lambda probabilities, **kwargs: torch.ops.aten.bernoulli.Tensor(
            torch.empty(4, 8), probabilities, **kwargs,
        ),
        id="empty-template",
    ),
])
def test_sampling_dependencies_and_partial_random_writes(sample):
    with DeferredInitialization() as capture:
        probabilities = torch.full((4, 8), 0.25)
        sampled = sample(probabilities)
        mean = torch.full((4, 8), 3.)
        normal = torch.normal(mean, 0.5)
        destination = torch.ones(4, 8)
        before = destination.clone()
        destination[1].bernoulli_(probabilities[1])
        probabilities.zero_()
        mean.fill_(20.)
    assert torch.equal(capture.materialize(before), torch.ones(4, 8))
    sample_seed = capture._storage(sampled).node.seed
    expected = sample(
        torch.full((4, 8), 0.25), generator=torch.Generator().manual_seed(sample_seed),
    )
    assert torch.equal(capture.materialize(sampled), expected)
    normal_seed = capture._storage(normal).node.seed
    expected_normal = torch.normal(
        torch.full((4, 8), 3.), 0.5, generator=torch.Generator().manual_seed(normal_seed),
    )
    assert torch.equal(capture.materialize(normal), expected_normal)
    write_seed = capture._storage(destination).node.seed
    expected_destination = torch.ones(4, 8)
    expected_destination[1].bernoulli_(
        torch.full((8,), 0.25), generator=torch.Generator().manual_seed(write_seed),
    )
    assert torch.equal(capture.materialize(destination), expected_destination)


def test_randint_like_dtype_and_memory_format():
    with DeferredInitialization() as capture:
        tensor = torch.randint_like(
            torch.empty(2, 3, 4, 5), -4, 7,
            dtype=torch.int64, memory_format=torch.channels_last,
        )
    result = capture.materialize(tensor)
    assert result.dtype == torch.int64
    assert result.is_contiguous(memory_format=torch.channels_last)
    assert result.min() >= -4
    assert result.max() < 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sampling_external_cuda_input_and_generator():
    probabilities = torch.full((32,), 0.25, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(132)
    state = generator.get_state()
    with DeferredInitialization() as capture:
        tensor = torch.bernoulli(probabilities, generator=generator)
    assert tensor.is_meta
    expected = torch.bernoulli(
        probabilities.cpu(),
        generator=torch.Generator().manual_seed(capture._storage(tensor).node.seed),
    )
    assert torch.equal(capture.materialize(tensor), expected)
    assert torch.equal(generator.get_state(), state)
    probabilities.fill_(0.75)
    with pytest.raises(RuntimeError, match="external tensor data changed"):
        capture.materialize(tensor)


@pytest.mark.parametrize("copy", [
    pytest.param(lambda x: x.clone(), id="clone"),
    pytest.param(lambda x: x.to(torch.float64), id="to-copy"),
])
@pytest.mark.parametrize("view", [
    pytest.param(lambda x: x, id="contiguous"),
    pytest.param(lambda x: x.t(), id="transpose"),
    pytest.param(lambda x: x[:, ::2], id="slice"),
])
def test_external_copy_uses_meta_inputs(copy, view):
    external = view(torch.arange(24, dtype=torch.float32).view(4, 6))
    expected = copy(external)
    with DeferredInitialization() as capture:
        copied = copy(external)
    assert copied.is_meta
    assert copied.stride() == expected.stride()
    actual = capture.materialize(copied)
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert torch.equal(actual, expected)
    actual.zero_()
    assert torch.equal(external, expected)
    assert torch.equal(capture.materialize(copied), expected)
    external.add_(1)
    with pytest.raises(RuntimeError, match="external tensor data changed"):
        capture.materialize(copied)


def test_external_data_and_replay_failure_cleanup():
    external = torch.arange(6).view(2, 3)
    with DeferredInitialization() as capture:
        independent = torch.ones(7)
        borrowed = torch.empty(3, 2, dtype=torch.int64)
        borrowed.copy_(external.t())
    assert torch.equal(capture.materialize(borrowed), external.t())
    external.add_(1)
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="external tensor data changed after capture"):
        capture.materialize(borrowed)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(capture.materialize(independent), torch.ones(7))


@pytest.mark.parametrize("operation,dtype", [
    pytest.param(lambda x, out: torch.ones(3, out=out), torch.float32, id="factory-out"),
    pytest.param(lambda x, out: torch.add(x, 2, out=out), torch.float32, id="add-out"),
    pytest.param(
        lambda x, out: torch.ops.aten.add.Scalar_out(x, 2, out=out), torch.float32,
        id="scalar-out",
    ),
    pytest.param(
        lambda x, out: torch.aminmax(x, out=(out[0], out[1])), torch.float32,
        id="min-max-outputs",
    ),
    pytest.param(
        lambda x, out: torch.sort(x, out=(out, torch.empty(3, dtype=torch.int64))), torch.float32,
        id="values-indices-outputs",
    ),
    pytest.param(lambda x, out: torch.randint(5, (3,), out=out), torch.int64, id="randint-out"),
    pytest.param(lambda x, out: torch.randperm(3, out=out), torch.int64, id="randperm-out"),
    pytest.param(
        lambda x, out: torch.multinomial(x, 3, out=out), torch.int64,
        id="multinomial-out",
    ),
])
def test_out_overloads_are_rejected(operation, dtype):
    with DeferredInitialization():
        source = torch.ones(3)
        destination = torch.empty(3, dtype=dtype)
        with pytest.raises(RuntimeError, match="out= overload"):
            operation(source, destination)


@pytest.mark.parametrize("factory, message", [
    (lambda: torch.ones(2, 2).addmm_(torch.eye(2), torch.eye(2)), "unsupported mutation"),
    (lambda: torch._foreach_add_([torch.ones(3)], 1), "unsupported mutation"),
    (lambda: torch.ones(1, 4).expand(3, 4).fill_(1), "overlapping mutation"),
    (lambda: torch.empty_strided((3, 4), (0, 1)), "overlapping"),
])
def test_unsupported_constructors_fail_explicitly(factory, message):
    with pytest.raises(RuntimeError, match=message):
        with DeferredInitialization():
            factory()


@pytest.mark.parametrize("factory", [
    lambda: torch.empty(3, 4),
    lambda: torch.empty_strided((3, 4), (1, 3)),
    lambda: torch.empty_like(torch.ones(3, 4)),
    lambda: torch.ones(1).new_empty(3, 4),
    lambda: torch.ones(1).new_empty_strided((3, 4), (1, 3)),
    lambda: torch.empty(3, 4).clone(),
])
def test_empty_allocation_materialization(factory):
    with DeferredInitialization() as capture:
        empty = factory()
    actual = capture.materialize(empty)
    assert actual.device.type == "cpu"
    assert actual.shape == empty.shape
    assert actual.stride() == empty.stride()
    assert actual.dtype == empty.dtype
    assert empty.is_meta
    assert actual.untyped_storage() is not capture.materialize(empty).untyped_storage()


def test_partial_writes_initialize_empty_and_preserve_snapshots():
    with DeferredInitialization() as capture:
        tensor = torch.empty(3, 4)
        tensor[0].fill_(2)
        before = tensor[0].clone()
        tensor[1:].fill_(3)
        tensor[0].add_(5)
    assert torch.equal(capture.materialize(before), torch.full((4,), 2.))
    expected = torch.full((3, 4), 3.)
    expected[0].fill_(7)
    assert torch.equal(capture.materialize(tensor), expected)
    assert torch.equal(capture.materialize(tensor), expected)


def test_capture_lifecycle_and_usage_errors():
    capture = DeferredInitialization()
    with pytest.raises(RuntimeError, match="unsupported mutation.*addmm_"):
        with capture:
            torch.ones(2, 2).addmm_(torch.eye(2), torch.eye(2))
    assert not torch.ones(1).is_meta
    with pytest.raises(RuntimeError, match="cannot be reused"):
        with capture:
            pass
    with DeferredInitialization() as another:
        value = torch.ones(1)
        with pytest.raises(RuntimeError, match="after leaving capture"):
            another.materialize(value)
    with pytest.raises(RuntimeError, match="untracked meta"):
        another.materialize(torch.empty(3, device="meta"))


@pytest.mark.parametrize("mutation", ["parameter", "view", "reset_parameters"])
def test_post_capture_value_mutation_does_not_change_replay(mutation):
    with DeferredInitialization() as capture:
        model = torch.nn.Linear(3, 4)
    expected = capture.materialize(model.weight)
    with torch.no_grad():
        if mutation == "parameter":
            model.weight.zero_()
        elif mutation == "view":
            model.weight[0].fill_(7)
        else:
            model.reset_parameters()
    assert torch.equal(capture.materialize(model.weight), expected)


def test_replay_error_restores_rng_and_releases_dependency_cache():
    class FailingCopy(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.copy_.default:
                raise RuntimeError("injected replay failure")
            return func(*args, **(kwargs or {}))

    with DeferredInitialization() as capture:
        source = torch.empty(3).uniform_(1, 2)
        destination = torch.empty(3)
        destination.copy_(source)
    before = torch.get_rng_state()
    with AllocationRecorder() as recorder:
        with pytest.raises(RuntimeError, match="replay failed: injected replay failure"), FailingCopy():
            capture.materialize(destination)
    gc.collect()
    assert all(reference() is None for _, reference in recorder.outputs)
    assert torch.equal(torch.get_rng_state(), before)
    assert capture.materialize(source).min() >= 1


def test_strided_buffer_and_requested_view_preserve_layout():
    with DeferredInitialization() as capture:
        model = torch.nn.Module()
        model.register_buffer("strided", torch.ones(3, 4).t())
        model.register_buffer("sliced", torch.arange(32)[3:11:2])
    strided = capture.materialize(model.strided)
    assert strided.stride() == model.strided.stride()
    assert torch.equal(strided, torch.ones(4, 3))
    sliced = capture.materialize(model.sliced)
    assert torch.equal(sliced, torch.arange(32)[3:11:2])
    assert sliced.stride() == model.sliced.stride()
    assert sliced.storage_offset() == model.sliced.storage_offset()
    assert sliced.untyped_storage().nbytes() == 32 * sliced.element_size()
    from torch.multiprocessing.reductions import StorageWeakRef

    storage = StorageWeakRef(sliced.untyped_storage())
    del sliced
    assert storage.expired()


def test_materialize_external_cpu_view_does_not_copy():
    external = torch.arange(16)[2:10:2]
    with DeferredInitialization() as capture:
        pass
    result = capture.materialize(external)
    assert torch.equal(result, external)
    assert result.untyped_storage() is external.untyped_storage()
    assert result.stride() == external.stride()
    assert result.storage_offset() == external.storage_offset()


def test_registered_buffers_snapshot_clones_and_module_dtype_conversion():
    class SnapshotModel(torch.nn.Module):
        initial_tensors = {}

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(8, 8, bias=False) for _ in range(3)
            ])
            self.shared = self.layers[0]
            self.register_buffer("scale", torch.full((8,), random.random() + np.random.rand()))
            self.register_buffer("offset", torch.rand(8), persistent=False)
            self.register_buffer("scalar", torch.tensor([0.125], dtype=torch.float64))
            self.scalar.add_(2 ** -45)
            type(self).initial_tensors = {
                name: value.detach().clone()
                for name, value in list(self.named_parameters(remove_duplicate=False))
                + list(self.named_buffers(remove_duplicate=False))
            }

    with DeferredInitialization() as capture:
        model = SnapshotModel().double()
    assert all(t.is_meta for t in SnapshotModel.initial_tensors.values())
    assert model.shared is model.layers[0]
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        actual = capture.materialize(value)
        initial = capture.materialize(SnapshotModel.initial_tensors[name])
        assert actual.dtype == torch.float64
        assert torch.equal(actual, initial.double())
    scalar = capture.materialize(model.scalar)
    assert scalar.item() == 0.125 + 2 ** -45
    assert scalar.item() != scalar.float().item()


def test_actual_cpu_allocation_events_exclude_unselected_weights():
    from torch.profiler import ProfilerActivity, profile

    def factory():
        model = torch.nn.Sequential(*[
            torch.nn.Linear(128, 128, bias=False) for _ in range(4)
        ])
        snapshots = [weight.detach().clone() for weight in model.parameters()]
        return model, snapshots

    def allocated_bytes(profiler):
        return sum(max(event.self_cpu_memory_usage, 0) for event in profiler.events())

    # CPU profiler memory events come from the allocator, independently of the
    # tensors' device labels observed by a Python dispatch recorder.
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, acc_events=True) as eager:
        eager_model, eager_snapshots = factory()
    expected_bytes = sum(weight.numel() * weight.element_size() for weight in eager_model.parameters())
    # original weights + their clones
    assert allocated_bytes(eager) >= 2 * expected_bytes
    del eager_model, eager_snapshots

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, acc_events=True) as deferred:
        with DeferredInitialization() as capture:
            model, snapshots = factory()
    assert allocated_bytes(deferred) == 0
    assert all(snapshot.is_meta for snapshot in snapshots)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, acc_events=True) as replay:
        selected = capture.materialize(model[0].weight)
    assert allocated_bytes(replay) == selected.numel() * selected.element_size()


def test_generic_pure_operations_and_historical_inputs():
    def factory():
        x = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        y = torch.sin(x) + torch.cos(x)
        z = torch.tril((y / 2).pow(2))
        result = z.sum(1) + z.mean(1)
        x.fill_(99)
        return result

    expected = factory()
    with DeferredInitialization() as capture:
        result = factory()
    assert result.is_meta
    assert capture._storage(result).node.operation is None
    torch.testing.assert_close(capture.materialize(result), expected)


@pytest.mark.parametrize("dtype", [None, torch.complex128])
def test_generic_complex_reduction_preserves_computation_dtype(dtype):
    def factory():
        return torch.linalg.vector_norm(torch.ones(4, dtype=torch.complex64), dtype=dtype)

    expected = factory()
    with DeferredInitialization() as capture:
        result = factory()
    torch.testing.assert_close(capture.materialize(result), expected, rtol=0, atol=0)


def test_generic_multioutput_shared_operation_and_mutation():
    with DeferredInitialization() as capture:
        source = torch.arange(12, dtype=torch.float64).reshape(4, 3)
        q, r = torch.linalg.qr(source)
        assert capture._storage(q).node.operation is capture._storage(r).node.operation
        assert capture._storage(q).node.operation is not None
        before = q.clone()
        combined = q @ r
        q.t()[0].fill_(2)
        maxima, indices = torch.max(combined, dim=1)
    expected_q, _ = torch.linalg.qr(torch.arange(12, dtype=torch.float64).reshape(4, 3))
    torch.testing.assert_close(capture.materialize(before), expected_q)
    expected_q[:, 0].fill_(2)
    torch.testing.assert_close(capture.materialize(q), expected_q)
    torch.testing.assert_close(capture.materialize(maxima), torch.arange(2, 12, 3, dtype=torch.float64))
    assert torch.equal(capture.materialize(indices), torch.full((4,), 2, dtype=torch.int64))

    class QRRecorder(TorchDispatchMode):
        calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.linalg_qr.default:
                self.calls += 1
            return func(*args, **(kwargs or {}))

    with QRRecorder() as recorder:
        capture.materialize(combined)
    assert recorder.calls == 1


def test_data_dependent_fallback_control_flow_and_later_operations(caplog):
    with DeferredInitialization() as capture:
        x = torch.arange(6)
        positions = torch.nonzero(x > 2).flatten()
        count = positions.sum().item()
        assert count == 12
        result = positions.sin() if count > 10 else positions.cos()
        positions[0].fill_(5)
        changed = positions + 1
        x.fill_(99)
    torch.testing.assert_close(capture.materialize(result), torch.tensor([3, 4, 5]).sin())
    assert torch.equal(capture.materialize(changed), torch.tensor([6, 5, 6]))
    assert "fallback" in caplog.text.lower()
    first = capture.materialize(positions)
    first.fill_(42)
    assert torch.equal(capture.materialize(positions), torch.tensor([5, 4, 5]))


def test_generic_tagged_random_is_stable_and_preserves_rng():
    state = torch.get_rng_state()
    generator = torch.Generator().manual_seed(14)
    generator_state = generator.get_state()
    with DeferredInitialization() as capture:
        gamma = torch.ops.aten._standard_gamma.default(torch.ones(64), generator=generator)
        values, mask = torch.ops.aten.native_dropout.default(torch.ones(64), 0.3, True)
    for result in (gamma, values, mask):
        assert torch.equal(capture.materialize(result), capture.materialize(result))
    assert torch.equal(capture.materialize(values), capture.materialize(mask) / 0.7)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(generator.get_state(), generator_state)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_control_flow_uses_same_values_as_replay(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    with DeferredInitialization(device=device) as capture:
        x = torch.randn(32)
        total = x.sum().item()
        factor = 2 if total > 0 else 3
        result = x * factor
    with torch.random.fork_rng(devices=[]):
        torch.rand(100)
        materialized = capture.materialize(x)
        assert materialized.sum().item() == total
        assert torch.equal(capture.materialize(result), materialized * factor)
    assert torch.equal(torch.get_rng_state(), state)
    if cuda_state is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_state)


def test_trunc_normal_and_partial_unary_writes():
    with DeferredInitialization() as capture:
        x = torch.empty(256)
        torch.nn.init.trunc_normal_(x, mean=0.2, std=0.4, a=-0.5, b=0.8)
        # PyTorch <=2.11 decomposes this initializer into several ATen nodes.
        random_node = capture._storage(x).node
        while random_node.seed is None:
            random_node = random_node.previous
        seed = random_node.seed
        before = x.clone()
        x[3:8].clamp_(-0.1, 0.1)
    original = capture.materialize(before)
    assert original.min() >= -0.5
    assert original.max() <= 0.8
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = torch.nn.init.trunc_normal_(
            torch.empty(256), mean=0.2, std=0.4, a=-0.5, b=0.8,
        )
    torch.testing.assert_close(original, reference, rtol=0, atol=0)
    expected = original.clone()
    expected[3:8].clamp_(-0.1, 0.1)
    torch.testing.assert_close(capture.materialize(x), expected)


@pytest.mark.parametrize('fail', [False, True])
def test_python_function_observer_order(fail):
    """Redispatch one PyFunction through every observer, restoring the stack on errors."""
    events = []

    def function(x):
        events.append('body')
        assert not _get_current_dispatch_mode_stack()
        if fail:
            raise ValueError('body failed')
        return x + 1

    operation = PyFunction(function, 'python::increment(Tensor x) -> Tensor')

    class Observer(TorchDispatchMode):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if not isinstance(func, PyFunction):
                return func(*args, **(kwargs or {}))
            assert func is operation
            events.append(f'{self.name}:enter')
            try:
                return func(*args, **(kwargs or {}))
            finally:
                events.append(f'{self.name}:exit')

    x = torch.arange(4)
    with Observer('outer'), DeferredInitialization() as capture, Observer('inner'):
        before = _get_current_dispatch_mode_stack()
        if fail:
            with pytest.raises(ValueError, match='body failed'):
                operation.call_with_normalized_args(x)
        else:
            result = operation.call_with_normalized_args(x)
        assert _get_current_dispatch_mode_stack() == before
    assert events == ['inner:enter', 'outer:enter', 'body', 'outer:exit', 'inner:exit']
    if not fail:
        assert torch.equal(capture.materialize(result), x + 1)


def test_python_function_fallback_observers(caplog):
    """Lower observers see concrete fallback; upper observers see the original call."""
    seen = {'lower': [], 'upper': []}
    operation = PyFunction(lambda x: torch.nonzero(x), 'python::nonzero(Tensor x) -> Tensor')

    class Observer(TorchDispatchMode):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if isinstance(func, PyFunction):
                assert func is operation
                seen[self.name].append(args[0].device.type)
            return func(*args, **(kwargs or {}))

    with Observer('lower'), DeferredInitialization() as capture, Observer('upper'):
        result = operation.call_with_normalized_args(torch.arange(4))
    assert seen == {'lower': ['meta', 'cpu'], 'upper': ['meta']}
    with Observer('lower'):
        actual = capture.materialize(result)
    assert seen['lower'] == ['meta', 'cpu', 'cpu']
    assert torch.equal(actual, torch.tensor([[1], [2], [3]]))
    assert 'fallback' in caplog.text.lower()


def test_python_function_seeded_factory_restores_rng():
    """Reuse tagged-op replay and normalize positional generator/device arguments."""
    def factory(shape, generator=None, device='cpu', scale=2):
        return torch.rand(shape, device=device, generator=generator) * scale

    operation = PyFunction(
        factory,
        'python::factory(int[] shape, Generator? generator=None, Device device="cpu", int scale=2) -> Tensor',
        tags=(torch.Tag.nondeterministic_seeded,),
    )
    generator = torch.Generator().manual_seed(17)
    generator_state = generator.get_state()
    with preserve_rng_state():
        torch_state = torch.get_rng_state()
        with DeferredInitialization() as capture:
            result = operation.call_with_normalized_args((8,), generator, 'cpu', 2)
        seed = capture._storage(result).node.seed
        actual = capture.materialize(result)
        assert torch.equal(actual, capture.materialize(result))
        assert torch.equal(torch.get_rng_state(), torch_state)
        assert torch.equal(generator.get_state(), generator_state)
        expected = factory((8,), torch.Generator().manual_seed(seed))
        assert torch.equal(actual, expected)


@pytest.fixture
def python_trunc_normal(monkeypatch):
    original = torch.nn.init._no_grad_trunc_normal_

    def initializer(tensor, mean, std, a, b, generator=None):
        # Reproduce PyTorch 2.12+'s meta early return on older test runtimes.
        if tensor.is_meta:
            return tensor
        return original(tensor, mean, std, a, b, generator=generator)

    monkeypatch.setattr(torch.nn.init, '_no_grad_trunc_normal_', initializer)
    monkeypatch.setattr(torch, '__version__', torch.torch_version.TorchVersion('2.12.0'))
    # Emulate the routing registered at import time on PyTorch 2.12+.
    monkeypatch.setattr(DeferredInitialization, '_writes', DeferredInitialization._writes | {'trunc_normal_'})
    monkeypatch.setattr(DeferredInitialization, '_random', DeferredInitialization._random | {'trunc_normal_'})
    return initializer


@pytest.mark.parametrize('version, enabled', [('2.11.0', False), ('2.12.0', True)])
def test_python_initializer_patch_predicate(monkeypatch, version, enabled):
    """Evaluate each patch's version predicate when entering, not when importing."""
    original = torch.nn.init._no_grad_trunc_normal_
    monkeypatch.setattr(torch, '__version__', torch.torch_version.TorchVersion(version))
    with DeferredInitialization():
        assert (torch.nn.init._no_grad_trunc_normal_ is not original) == enabled
    assert torch.nn.init._no_grad_trunc_normal_ is original


def test_python_initializer_patch_lifetime(python_trunc_normal):
    """Restore each enclosing helper on normal/exceptional exit, including nested captures."""
    original = torch.nn.init._no_grad_trunc_normal_
    with DeferredInitialization():
        outer_helper = torch.nn.init._no_grad_trunc_normal_
        assert outer_helper is not original
        with DeferredInitialization():
            assert torch.nn.init._no_grad_trunc_normal_ is not outer_helper
        assert torch.nn.init._no_grad_trunc_normal_ is outer_helper
        with pytest.raises(ValueError, match='constructor failed'):
            with DeferredInitialization():
                assert torch.nn.init._no_grad_trunc_normal_ is not outer_helper
                raise ValueError('constructor failed')
        assert torch.nn.init._no_grad_trunc_normal_ is outer_helper
    assert torch.nn.init._no_grad_trunc_normal_ is original


@pytest.mark.parametrize('stage', ['transfer', 'mode'])
def test_python_initializer_patch_entry_failure(python_trunc_normal, monkeypatch, stage):
    """Patch transfer and mode-entry failures restore patches without leaking a mode."""
    original = torch.nn.init._no_grad_trunc_normal_
    original_modes = _get_current_dispatch_mode_stack()

    def fail_enter(self):
        raise RuntimeError('entry failed')

    target, name = (ExitStack, 'pop_all') if stage == 'transfer' else (TorchDispatchMode, '__enter__')
    monkeypatch.setattr(target, name, fail_enter)
    capture = DeferredInitialization()
    with pytest.raises(RuntimeError, match='entry failed'):
        with capture:
            pytest.fail('entry should have failed')
    assert torch.nn.init._no_grad_trunc_normal_ is original
    assert _get_current_dispatch_mode_stack() == original_modes
    assert not capture._active


@pytest.mark.parametrize('partial', [False, True])
def test_python_trunc_normal_preserves_layout_and_rng(python_trunc_normal, partial):
    """Preserve snapshots and untouched elements; full overwrites need no old storage value."""
    generator = torch.Generator().manual_seed(17)
    generator_state = generator.get_state()
    rng_state = torch.get_rng_state()
    region = slice(3, 12) if partial else slice(None)
    with DeferredInitialization() as capture:
        x = torch.full((16,), 0.25)
        before = x.clone()
        target = x[region]
        assert torch.nn.init.trunc_normal_(
            target, mean=0.2, std=0.4, a=-0.5, b=0.8, generator=generator,
        ) is target
        node = capture._storage(x).node
        assert node.func.func is python_trunc_normal
        assert (node.previous is not None) == partial
    expected = torch.full((16,), 0.25)
    torch.nn.init.trunc_normal_(
        expected[region], mean=0.2, std=0.4, a=-0.5, b=0.8,
        generator=torch.Generator().manual_seed(node.seed),
    )
    assert torch.equal(capture.materialize(before), torch.full((16,), 0.25))
    assert torch.equal(capture.materialize(x), expected)
    assert torch.equal(generator.get_state(), generator_state)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_python_matrix_initializers():
    """Replay eye_, orthogonal_ and sparse_ through their public initialization APIs."""
    with DeferredInitialization() as capture:
        eye = torch.nn.init.eye_(torch.empty(4, 4))
        orthogonal = torch.nn.init.orthogonal_(torch.empty(4, 4))
        sparse = torch.nn.init.sparse_(torch.empty(4, 4), sparsity=0.5)

    assert torch.equal(capture.materialize(eye), torch.eye(4))
    q = capture.materialize(orthogonal)
    torch.testing.assert_close(q.T @ q, torch.eye(4), rtol=1e-5, atol=1e-6)
    values = capture.materialize(sparse)
    assert ((values == 0).sum(dim=0) >= 2).all()
    assert torch.equal(values, capture.materialize(sparse))


@pytest.mark.skipif(torch.__version__ < (2, 2), reason='Library.define tags require PyTorch 2.2+')
def test_python_initializer_inside_concrete_fallback(python_trunc_normal, caplog):
    """Execute, rather than recapture, an initializer inside a real custom-op fallback."""
    lib = torch.library.Library('deferred_python_initializer_test', 'DEF')
    lib.define('initialize(Tensor template) -> Tensor', tags=(torch.Tag.nondeterministic_seeded,))
    lib.impl(
        'initialize',
        lambda template: torch.nn.init.trunc_normal_(torch.empty_like(template), a=-0.5, b=0.8),
        'CPU',
    )
    with DeferredInitialization() as capture:
        result = torch.ops.deferred_python_initializer_test.initialize(torch.ones(32))
        total = result.sum().item()
    actual = capture.materialize(result)
    assert actual.sum().item() == total
    assert torch.equal(actual, capture.materialize(result))
    assert actual.min() >= -0.5 and actual.max() <= 0.8
    assert 'fallback' in caplog.text.lower()


def test_generic_invalid_inputs_are_not_fallback(caplog):
    with DeferredInitialization():
        with pytest.raises(RuntimeError):
            torch.ones(2, 3) @ torch.ones(4, 5)
    assert "fallback" not in caplog.text.lower()


@pytest.mark.parametrize("operation", [
    lambda x: torch.view_as_complex(x.reshape(8, 2)).conj(),
])
def test_generic_aliasing_outputs_are_rejected(operation):
    with DeferredInitialization() as capture:
        source = torch.ones(4, 4)
        with pytest.raises(RuntimeError, match="unsupported aliasing operation"):
            operation(source)
        source[0].fill_(3)
    expected = torch.ones(4, 4)
    expected[0].fill_(3)
    assert torch.equal(capture.materialize(source), expected)


@pytest.mark.skipif(torch.__version__ < (2, 2), reason='Library.define tags require PyTorch 2.2+')
@pytest.mark.parametrize('use_cuda_rng', [False, True])
def test_missing_meta_custom_operation_replays_shared_outputs_and_preserves_rng(caplog, use_cuda_rng):
    from torch.multiprocessing.reductions import StorageWeakRef

    if use_cuda_rng and not torch.cuda.is_available():
        pytest.skip('requires CUDA RNG')

    calls = []
    output_storages = []
    namespace = f"deferred_init_test_{int(use_cuda_rng)}"
    lib = torch.library.Library(namespace, "FRAGMENT")
    lib.define("data_op(Tensor x) -> (Tensor, Tensor)", tags=(torch.Tag.nondeterministic_seeded,))

    def implementation(x):
        calls.append(1)
        value = x + random.random() + np.random.rand() + torch.rand_like(x)
        if use_cuda_rng:
            value = value + torch.rand((), device='cuda').cpu()
        outputs = value, x.nonzero()
        output_storages.extend(StorageWeakRef(value.untyped_storage()) for value in outputs)
        return outputs

    lib.impl("data_op", implementation, "CPU")
    state = torch.get_rng_state()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_state = torch.cuda.get_rng_state() if use_cuda_rng else None
    with AllocationRecorder() as recorder, DeferredInitialization() as capture:
        source = torch.arange(4, dtype=torch.float32)
        first, second = getattr(torch.ops, namespace).data_op(source)
        downstream = first.sin() + second.sum()
        source.fill_(99)
    assert calls == [1]
    assert any(device == "cpu" for device, _ in recorder.outputs)
    # the output storages should be expired after use in replay.
    assert all(storage.expired() for storage in output_storages)
    with preserve_rng_state():
        random.seed(999)
        np.random.seed(999)
        torch.random.default_generator.manual_seed(999)
        if use_cuda_rng:
            torch.cuda.manual_seed(999)
            replay_cuda_state = torch.cuda.get_rng_state()
        actual = capture.materialize(downstream)
        assert random.getstate() == random.Random(999).getstate()
        assert np.array_equal(np.random.get_state()[1], np.random.RandomState(999).get_state()[1])
        if use_cuda_rng:
            assert torch.equal(torch.cuda.get_rng_state(), replay_cuda_state)
    assert calls == [1, 1]
    assert all(storage.expired() for storage in output_storages)
    first_value = capture.materialize(first)
    torch.testing.assert_close(actual, first_value.sin() + 6, rtol=0, atol=0)
    first_value.zero_()
    torch.testing.assert_close(capture.materialize(downstream), actual, rtol=0, atol=0)
    assert torch.equal(capture.materialize(second), torch.tensor([[1], [2], [3]]))
    assert calls == [1] * 5
    del first_value
    assert all(storage.expired() for storage in output_storages)
    assert torch.equal(torch.get_rng_state(), state)
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    assert np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    if use_cuda_rng:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_state)
    assert "concrete fallback" in caplog.text


def test_factory_missing_meta_fallback_replays_with_captured_dtype(caplog):
    class MissingMetaFactory(TorchDispatchMode):
        calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func == torch.ops.aten.ones.default:
                if kwargs.get("device") == torch.device("meta"):
                    raise NotImplementedError("implementation unavailable")
                self.calls += 1
            return func(*args, **kwargs)

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with MissingMetaFactory() as recorder, DeferredInitialization() as capture:
            tensor = torch.ones(4)
            downstream = tensor.sin()
        assert tensor.is_meta
        # 1 capture fallback
        assert recorder.calls == 1
        torch.set_default_dtype(torch.float32)
        with recorder:
            first = capture.materialize(tensor)
            first.zero_()
            actual = capture.materialize(downstream)
        # 1 capture fallback + 2 materialize
        assert recorder.calls == 3
        assert actual.dtype == torch.float64
        torch.testing.assert_close(actual, torch.ones(4, dtype=torch.float64).sin(), rtol=0, atol=0)
        assert "concrete fallback" in caplog.text
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.skipif(torch.__version__ < (2, 2), reason='Library.define tags require PyTorch 2.2+')
@pytest.mark.parametrize("use_generator", [False, True])
def test_generic_positional_generator_and_following_arguments(use_generator):
    namespace = f"deferred_init_generator_{int(use_generator)}"
    lib = torch.library.Library(namespace, "FRAGMENT")
    lib.define(
        "sample(Tensor x, Generator? generator=None, float offset=0.) -> Tensor",
        tags=(torch.Tag.nondeterministic_seeded,),
    )
    lib.impl("sample", lambda x, generator=None, offset=0.: x.clone(), "Meta")

    def cpu(x, generator=None, offset=0.):
        return torch.rand(x.shape, dtype=x.dtype, generator=generator if use_generator else None) + offset

    lib.impl("sample", cpu, "CPU")
    generator = torch.Generator().manual_seed(42)
    generator_state = generator.get_state()
    state = torch.get_rng_state()
    with DeferredInitialization() as capture:
        result = getattr(torch.ops, namespace).sample(torch.ones(8), generator, 3.)
    seed = capture._storage(result).node.seed
    assert seed is not None
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        expected = cpu(torch.ones(8), torch.Generator().manual_seed(seed), 3.)
    torch.testing.assert_close(capture.materialize(result), expected, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), generator_state)
    generator.manual_seed(100)
    assert torch.equal(capture.materialize(result), expected)
    assert torch.equal(torch.get_rng_state(), state)
    assert not torch.equal(generator.get_state(), generator_state)


@pytest.mark.parametrize("operation,arguments", [
    ("erfinv_", ()),
    ("clamp_", (-1, 1)),
    ("exponential_", (0.5,)),
    ("geometric_", (0.5,)),
    ("log_normal_", (0.5,)),
    ("cauchy_", (0.5,)),
])
def test_writes_reject_external_and_overlapping(operation, arguments):
    external = torch.ones(3)
    with DeferredInitialization():
        with pytest.raises(RuntimeError, match="mutation of external data"):
            getattr(external, operation)(*arguments)
        with pytest.raises(RuntimeError, match="overlapping mutation"):
            getattr(torch.ones(1, 4).expand(3, 4), operation)(*arguments)
    assert torch.equal(external, torch.ones(3))


@pytest.mark.skipif(torch.__version__ < (2, 4), reason='torch.library.register_fake requires PyTorch 2.4+')
@pytest.mark.parametrize("tag", [
    torch.Tag.dynamic_output_shape, torch.Tag.data_dependent_output,
])
def test_custom_dynamic_shape_fallback(tag):
    namespace = f"deferred_init_{tag.name}"
    lib = torch.library.Library(namespace, "FRAGMENT")
    lib.define("positions(Tensor x) -> Tensor", tags=(tag,))
    lib.impl("positions", lambda x: x.nonzero(), "CPU")
    meta_calls = []

    @torch.library.register_fake(f"{namespace}::positions", lib=lib)
    def fake(x):
        meta_calls.append(1)
        return x.new_empty((torch.library.get_ctx().new_dynamic_size(), 1), dtype=torch.int64)

    with DeferredInitialization() as capture:
        result = getattr(torch.ops, namespace).positions(torch.arange(4))
        downstream = result + 2
    assert torch.equal(capture.materialize(downstream), torch.tensor([[3], [4], [5]]))
    assert meta_calls == []


def test_custom_meta_error_is_not_hidden(caplog):
    lib = torch.library.Library("deferred_init_invalid_meta", "FRAGMENT")
    lib.define("invalid(Tensor x) -> Tensor")
    calls = []
    lib.impl("invalid", lambda x: calls.append(1) or x.clone(), "CPU")

    def meta(x):
        raise RuntimeError("invalid input in custom meta kernel")

    lib.impl("invalid", meta, "Meta")
    with DeferredInitialization():
        with pytest.raises(RuntimeError, match="invalid input in custom meta kernel"):
            torch.ops.deferred_init_invalid_meta.invalid(torch.ones(3))
    assert calls == []
    assert "concrete fallback" not in caplog.text


def test_generic_replay_failure_restores_rng_and_dtype():
    lib = torch.library.Library("deferred_init_invalid_replay", "FRAGMENT")
    lib.define("invalid(Tensor x) -> Tensor")
    lib.impl("invalid", lambda x: x.clone(), "Meta")

    def cpu(x):
        torch.rand(1)
        random.random()
        np.random.rand()
        raise RuntimeError("custom replay failure")

    lib.impl("invalid", cpu, "CPU")
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with DeferredInitialization() as capture:
            result = torch.ops.deferred_init_invalid_replay.invalid(torch.ones(3))
        torch.set_default_dtype(torch.float32)
        state = torch.get_rng_state()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        with pytest.raises(RuntimeError, match="custom replay failure"):
            capture.materialize(result)
        assert torch.get_default_dtype() == torch.float32
        assert torch.equal(torch.get_rng_state(), state)
        assert random.getstate() == python_state
        assert np.array_equal(np.random.get_state()[1], numpy_state[1])
        assert np.random.get_state()[2:] == numpy_state[2:]
    finally:
        torch.set_default_dtype(previous_dtype)
