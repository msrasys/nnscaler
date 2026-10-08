#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import gc
import random
import weakref

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from nnscaler.runtime.deferred_initialization import DeferredInitialization


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
    previous_storage = None
    for parameter in list(model.parameters()) + snapshots:
        assert previous_storage is None or previous_storage.expired()
        assert all(source.is_meta for source in model.parameters())
        value = capture.materialize(parameter)
        previous_storage = StorageWeakRef(value.untyped_storage())
        assert not previous_storage.expired()
        del value
        # Refcounting must release storage immediately, without a gc.collect().
        assert previous_storage.expired()


def test_versions_shared_parameters_and_partial_writes():
    with DeferredInitialization() as capture:
        module = torch.nn.Module()
        module.first = torch.nn.Parameter(torch.ones(3, 4))
        module.second = module.first
        with torch.no_grad():
            copied_before = module.first.clone()
            module.first[1].fill_(5)
            copied_after = torch.empty_like(module.first)
            copied_after.copy_(module.first)
            module.second.mul_(2)
            module.first.add_(3)
    assert module.first is module.second
    expected = torch.ones(3, 4)
    expected[1] = 5
    assert torch.equal(capture.materialize(copied_before), torch.ones(3, 4))
    assert torch.equal(capture.materialize(copied_after), expected)
    assert torch.equal(capture.materialize(module.first), expected * 2 + 3)
    assert torch.equal(capture.materialize(module.second), expected * 2 + 3)


@pytest.mark.parametrize("operation", ["add_", "sub_", "mul_", "div_"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_scalar_arithmetic_preserves_previous_values(operation, partial, dtype):
    def initialize():
        tensor = torch.arange(-4, 4, dtype=dtype)
        before = tensor.clone()
        target = tensor[1:6] if partial else tensor
        kwargs = {"alpha": 3} if operation in {"add_", "sub_"} else {}
        getattr(target, operation)(2, **kwargs)
        after = target.clone()
        return tensor, before, after

    expected = initialize()
    with DeferredInitialization() as capture:
        actual = initialize()
    for tensor, reference in zip(actual, expected):
        torch.testing.assert_close(capture.materialize(tensor), reference, rtol=0, atol=0)


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_scalar_division_rounding_mode(rounding_mode, dtype):
    expected = torch.arange(-5, 5, dtype=dtype).div_(2, rounding_mode=rounding_mode)
    with DeferredInitialization() as capture:
        tensor = torch.arange(-5, 5, dtype=dtype)
        tensor.div_(2, rounding_mode=rounding_mode)
    torch.testing.assert_close(capture.materialize(tensor), expected, rtol=0, atol=0)


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


@pytest.mark.parametrize("operation", ["add_", "sub_", "mul_", "div_"])
def test_arithmetic_rejects_tensor_operands(operation):
    with DeferredInitialization():
        tensor = torch.ones(3)
        other = torch.ones(3)
        with pytest.raises(RuntimeError, match="tensor-valued arithmetic"):
            getattr(tensor, operation)(other)


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
    materialized = capture.materialize(actual)
    assert materialized.dtype == expected.dtype
    assert torch.equal(materialized, expected)


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


def test_tensor_literal_view_mutation_preserves_clone():
    with DeferredInitialization() as capture:
        tensor = torch.tensor([[1., 2.], [3., 4.]])
        before = tensor.clone()
        tensor[0].fill_(7)
    assert torch.equal(capture.materialize(before), torch.tensor([[1., 2.], [3., 4.]]))
    assert torch.equal(capture.materialize(tensor), torch.tensor([[7., 7.], [3., 4.]]))


def test_rng_determinism_private_generators_and_capture_state():
    def factory():
        return [torch.rand(7), torch.randn(7), torch.empty(7).normal_(2, 3)]

    original = torch.get_rng_state()
    try:
        torch.manual_seed(923)
        state = torch.get_rng_state()
        with DeferredInitialization() as first_capture:
            first = factory()
        assert torch.equal(torch.get_rng_state(), state)
        with DeferredInitialization() as second_capture:
            second = factory()
        for a, b in zip(reversed(first), reversed(second)):
            expected = first_capture.materialize(a)
            assert torch.equal(torch.get_rng_state(), state)
            assert torch.equal(second_capture.materialize(b), expected)
        torch.manual_seed(924)
        with DeferredInitialization() as third_capture:
            third = factory()
        assert not torch.equal(first_capture.materialize(first[0]), third_capture.materialize(third[0]))
    finally:
        torch.set_rng_state(original)


def test_explicit_generator_seed_and_state():
    generator = torch.Generator().manual_seed(132)
    state = generator.get_state()
    with DeferredInitialization() as capture:
        first = torch.rand(8, generator=generator)
        second = torch.empty(8).uniform_(generator=generator)
    assert torch.equal(generator.get_state(), state)
    before = torch.get_rng_state()
    assert torch.equal(capture.materialize(first), capture.materialize(first))
    assert not torch.equal(capture.materialize(first), capture.materialize(second))
    assert torch.equal(generator.get_state(), state)
    assert torch.equal(torch.get_rng_state(), before)


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


@pytest.mark.parametrize("factory", [
    pytest.param(lambda x: torch.rand_like(x), id="rand-like"),
    pytest.param(lambda x: torch.randn_like(x), id="randn-like"),
    pytest.param(lambda x: torch.randint_like(x, 7), id="randint-like-high"),
    pytest.param(lambda x: torch.randint_like(x, -4, 7), id="randint-like-low-high"),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
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
@pytest.mark.parametrize("source", ["literal", "computed", "external"])
@pytest.mark.parametrize("explicit_generator", [False, True])
def test_randint_like_tensor_bound(source, explicit_generator):
    external = torch.tensor(7)
    generator = torch.Generator().manual_seed(132) if explicit_generator else None
    state = generator.get_state() if generator is not None else torch.get_rng_state()
    with DeferredInitialization() as capture:
        if source == "literal":
            high = torch.tensor(7)
        elif source == "computed":
            high = torch.full((), 3, dtype=torch.int64).add_(4)
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
    assert torch.equal(actual, expected)
    assert actual.stride() == expected.stride()
    assert torch.equal(capture.materialize(sample), expected)
    assert torch.equal(generator.get_state() if generator is not None else torch.get_rng_state(), state)
    if source == "external":
        external.fill_(2)
        with pytest.raises(RuntimeError, match="external tensor data changed"):
            capture.materialize(sample)


@pytest.mark.skipif(
    not hasattr(torch.ops.aten.randint_like, "Tensor"),
    reason="requires the tensor upper-bound overload",
)
def test_randint_like_can_use_partially_initialized_tensor_bound():
    with DeferredInitialization() as capture:
        bounds = torch.empty(2, dtype=torch.int64)
        bounds[0].fill_(7)
        tensor = torch.randint_like(torch.empty(8), bounds[0])
    actual = capture.materialize(tensor)
    assert actual.device.type == "cpu"
    assert ((actual >= 0) & (actual < 7)).all()
    assert torch.equal(actual, capture.materialize(tensor))


def test_bernoulli_tensor_probability_uses_layout_only_template():
    with DeferredInitialization() as capture:
        probabilities = torch.full((4, 8), 0.25)
        sample = torch.ops.aten.bernoulli.Tensor(torch.empty(4, 8), probabilities)
        probabilities.fill_(1.)
    expected = torch.empty(4, 8).bernoulli_(
        torch.full((4, 8), 0.25),
        generator=torch.Generator().manual_seed(capture._storage(sample).node.seed),
    )
    assert torch.equal(capture.materialize(sample), expected)


def test_sampling_dependencies_and_partial_random_writes():
    with DeferredInitialization() as capture:
        probabilities = torch.full((4, 8), 0.25)
        sample = torch.bernoulli(probabilities)
        mean = torch.full((4, 8), 3.)
        normal = torch.normal(mean, 0.5)
        destination = torch.ones(4, 8)
        before = destination.clone()
        destination[1].bernoulli_(probabilities[1])
        probabilities.zero_()
        mean.fill_(20.)
    assert torch.equal(capture.materialize(before), torch.ones(4, 8))
    sample_seed = capture._storage(sample).node.seed
    expected = torch.bernoulli(
        torch.full((4, 8), 0.25), generator=torch.Generator().manual_seed(sample_seed),
    )
    assert torch.equal(capture.materialize(sample), expected)
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


@pytest.mark.parametrize("operation", ["rand_like", "randn_like"])
def test_complex_random_like(operation):
    factory = getattr(torch, operation)
    with DeferredInitialization() as capture:
        tensor = factory(torch.empty(8, dtype=torch.complex64))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(capture._storage(tensor).node.seed)
        expected = factory(torch.empty(8, dtype=torch.complex64))
    assert torch.equal(capture.materialize(tensor), expected)


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


@pytest.mark.parametrize("operation", ["exponential_", "geometric_", "log_normal_", "cauchy_"])
def test_random_writes_reject_external_destination(operation):
    external = torch.ones(8)
    with DeferredInitialization():
        with pytest.raises(RuntimeError, match="mutation of external data"):
            getattr(external, operation)(0.5)
    assert torch.equal(external, torch.ones(8))


@pytest.mark.parametrize("factory", [
    lambda out: torch.randint(5, (8,), out=out),
    lambda out: torch.randperm(8, out=out),
    lambda out: torch.multinomial(torch.ones(8), 8, out=out),
])
def test_extended_random_rejects_out(factory):
    with DeferredInitialization():
        with pytest.raises(RuntimeError, match="out= overload"):
            factory(torch.empty(8, dtype=torch.int64))


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


@pytest.mark.parametrize("operation", [
    pytest.param(lambda x, out: torch.ones(3, out=out), id="factory-out"),
    pytest.param(lambda x, out: torch.add(x, 2, out=out), id="add-out"),
    pytest.param(
        lambda x, out: torch.ops.aten.add.Scalar_out(x, 2, out=out),
        id="scalar-out",
    ),
    pytest.param(
        lambda x, out: torch.aminmax(x, out=(out[0], out[1])),
        id="min-max-outputs",
    ),
    pytest.param(
        lambda x, out: torch.sort(x, out=(out, torch.empty(3, dtype=torch.int64))),
        id="values-indices-outputs",
    ),
])
def test_out_overloads_are_rejected(operation):
    with DeferredInitialization():
        source = torch.ones(3)
        destination = torch.empty(3)
        with pytest.raises(RuntimeError, match="out= overload"):
            operation(source, destination)


@pytest.mark.parametrize("factory, message", [
    (lambda: torch.ones(3).sum(), "unsupported operation"),
    (lambda: torch.ones(()).item(), "_local_scalar_dense"),
    (lambda: torch.ones(3, 4).t().fill_(1), "noncontiguous mutation"),
    (lambda: torch.ones(3).add_(torch.ones(3)), "tensor-valued arithmetic"),
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


def test_untracked_meta_rejected():
    with DeferredInitialization() as capture:
        pass
    with pytest.raises(RuntimeError, match="untracked meta"):
        capture.materialize(torch.empty(3, device="meta"))


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


def test_capture_lifecycle_and_constructor_failure():
    capture = DeferredInitialization()
    with pytest.raises(RuntimeError, match="param_init_strategy='file'"):
        with capture:
            torch.ones(()).item()
    assert not torch.ones(1).is_meta
    with pytest.raises(RuntimeError, match="cannot be reused"):
        with capture:
            pass
    with DeferredInitialization() as another:
        value = torch.ones(1)
        with pytest.raises(RuntimeError, match="after leaving capture"):
            another.materialize(value)


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
        with pytest.raises(RuntimeError, match="replay failed.*param_init_strategy='file'"), FailingCopy():
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
