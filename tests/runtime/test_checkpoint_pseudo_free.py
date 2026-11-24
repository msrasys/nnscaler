# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Checkpoint replay after pipeline output payloads are genuinely released."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import copy
import gc
from types import SimpleNamespace

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from nnscaler.runtime.checkpoint import checkpoint
from nnscaler.runtime.activation_autograd import CheckpointSubmissionPolicy, SavedActivationContext
from nnscaler.runtime.executor import Executor
from nnscaler.runtime.function import multiref
from nnscaler.parallel import ComputeConfig, parallelize
from nnscaler.policies import OpPlan, get_pas_ops
from tests.utils import mock_cube_env, replace_all_device_with


@pytest.fixture(autouse=True)
def clear_executor():
    Executor.clear()
    yield
    Executor.clear()


def _run(checkpoint_fn, *, pseudo_free, hook_mode=None, transpose=False):
    data = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4)
    if transpose:
        data = data.t()
    source = data.detach().requires_grad_()
    weight = torch.linspace(0.5, 1.5, data.shape[-1], dtype=torch.float64, requires_grad=True)
    shapes, packed, unpacked, input_storage = [], [], [], []

    def pack(value):
        packed.append((tuple(value.shape), value.untyped_storage().data_ptr()))
        return value.detach().clone() if hook_mode == 'copy' else value

    def unpack(value):
        unpacked.append(tuple(value.shape))
        return value

    def region(value):
        shapes.append(tuple(value.shape))
        left, right = multiref(value, times=2)
        return torch.nn.functional.rms_norm(left, (data.shape[-1],), weight, 1e-6), right

    def segment(value):
        intermediate = value * 2
        input_storage.append(intermediate.untyped_storage().data_ptr())
        return checkpoint_fn(region, intermediate, use_reentrant=False)

    hooks = torch.autograd.graph.saved_tensors_hooks(pack, unpack) if hook_mode else nullcontext()
    with hooks:
        outputs = Executor.fexecute('segment', segment, source)
    incoming = tuple(torch.ones_like(output) for output in outputs)
    assert outputs[1]._base is None  # Pseudo-free must not be skipped as a view.
    if pseudo_free:
        Executor.pseudo_free_tensor(outputs[1])
        assert outputs[1].shape == (1,)
    gradients = Executor.backward('segment', (source,), outputs, incoming)
    Executor.check_clear()
    return gradients, weight.grad, shapes, packed, unpacked, input_storage


@pytest.mark.parametrize('hook_mode, transpose', [
    (None, False), (None, True), ('retain', False), ('copy', True),
])
def test_checkpoint_preserves_replay_gradients_and_external_saved_hooks(hook_mode, transpose):
    expected = _run(torch_checkpoint, pseudo_free=False, hook_mode=hook_mode, transpose=transpose)
    actual = _run(checkpoint, pseudo_free=True, hook_mode=hook_mode, transpose=transpose)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    shape = (4, 3) if transpose else (3, 4)
    assert actual[2] == [shape, shape]
    if hook_mode:
        assert [item for item in actual[3] if item[0] != (0,)] == [(shape, actual[5][0])]
        assert [item for item in actual[4] if item != (0,)] == [shape]


class _CPUStore:
    """Exercise the real saved-activation adapter with CPU snapshot storage."""
    def __init__(self, commit):
        self.config = SimpleNamespace(forward_commit_unit=commit)
        self.checkpoint_submission = CheckpointSubmissionPolicy()
        self.captured, self.restored, self.flushes = [], [], []
        self.group = SimpleNamespace(closed=False, flush=lambda: self.flushes.append(True))
        self._active_batch = ContextVar('checkpoint_test_batch', default=self.group)

    def _capture(self, tensor, group):
        assert group is self.group
        self.captured.append(tuple(tensor.shape))
        saved = tensor.detach().clone()

        def unpack():
            self.restored.append(tuple(saved.shape))
            return saved

        return SimpleNamespace(unpack=unpack)


@pytest.mark.parametrize('commit', ['group', 'checkpoint'])
def test_saved_activation_context_checkpoint_uses_stable_inputs(commit):
    store = _CPUStore(commit)

    def offload_checkpoint(function, *args, **kwargs):
        with SavedActivationContext(store) as saved:
            return saved.checkpoint(function, *args, **kwargs)

    expected = _run(torch_checkpoint, pseudo_free=False)
    actual = _run(offload_checkpoint, pseudo_free=True)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert [shape for shape in store.captured if shape != (0,)] == [(3, 4)]
    assert [shape for shape in store.restored if shape != (0,)] == [(3, 4)]
    assert len(store.flushes) == (1 if commit == 'checkpoint' else 0)


def test_checkpoint_keeps_passthrough_identity_and_repeated_inputs():
    value = torch.randn(3, 4, requires_grad=True) * 2

    def region(left, right, *, metadata):
        assert left is right and metadata['input'] is left
        return {'norm': left.sin(), 'aliases': (left, right), 'label': metadata['label']}

    with torch.autograd.graph.saved_tensors_hooks(lambda v: v.detach().clone(), lambda v: v):
        output = checkpoint(region, value, value, metadata={'input': value, 'label': 'kept'})
    assert output['aliases'] == (value, value)
    assert output['aliases'][0] is value and output['aliases'][1] is value
    assert output['label'] == 'kept'
    output['norm'].sum().backward()  # Replay also preserves repeated-input identity.


def test_checkpoint_preserves_tensor_hook_on_passthrough_input():
    source = torch.ones(3, 4, requires_grad=True)
    value = source * 2

    def region(original):
        assert original is value and not original.is_leaf
        original.register_hook(lambda gradient: gradient * 2)
        return original

    output = checkpoint(region, value)
    assert output is value
    output.sum().backward()
    assert torch.equal(source.grad, torch.full_like(source, 4))


def test_checkpoint_requires_non_reentrant_execution():
    with pytest.raises(ValueError, match='use_reentrant=False'):
        checkpoint(torch.sin, torch.ones(3, requires_grad=True), use_reentrant=True)


def test_checkpoint_preserves_higher_order_and_repeated_backward():
    value = torch.tensor([2., 3.], requires_grad=True)
    output = checkpoint(lambda x: x ** 3, value).sum()
    first, = torch.autograd.grad(output, value, create_graph=True, retain_graph=True)
    second, = torch.autograd.grad(first.sum(), value)
    repeated, = torch.autograd.grad(output, value)
    assert torch.equal(first, 3 * value.square())
    assert torch.equal(second, 6 * value)
    assert torch.equal(repeated, first)


def test_checkpoint_forwards_context_and_rng_controls():
    events = []

    @contextmanager
    def context(name):
        events.append(name)
        yield

    value = torch.ones(3, 4, requires_grad=True)
    torch.manual_seed(125)
    output = checkpoint(lambda x: x * torch.rand_like(x), value,
                        context_fn=lambda: (context('forward'), context('replay')),
                        preserve_rng_state=True)
    rng = torch.get_rng_state().clone()
    output.sum().backward()
    assert events == ['forward', 'replay']
    assert torch.equal(value.grad, output)
    assert torch.equal(torch.get_rng_state(), rng)


def test_checkpoint_still_detects_real_input_mutation():
    source = torch.randn(3, 4, requires_grad=True)
    value = source * 2
    output = checkpoint(torch.sin, value, use_reentrant=False)
    value.add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        output.sum().backward()


@pytest.mark.parametrize('kind', ['raw', 'checkpoint', 'activation_context'])
def test_copy_pack_releases_original_storage_without_gc_or_backward(kind):
    source = torch.arange(1., 13.).reshape(3, 4).requires_grad_()
    storages = []
    store = _CPUStore('group')

    def segment(value):
        intermediate = value * 2
        storages.append(StorageWeakRef(intermediate.untyped_storage()))
        if kind == 'activation_context':
            with SavedActivationContext(store) as saved:
                result = saved.checkpoint(torch.sin, intermediate)
        else:
            cp = torch_checkpoint if kind == 'raw' else checkpoint
            with torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach().clone(), lambda t: t):
                result = cp(torch.sin, intermediate, use_reentrant=False)
        return result, intermediate

    enabled = gc.isenabled()
    gc.disable()
    try:
        outputs = Executor.fexecute('storage', segment, source)
        incoming = tuple(torch.ones_like(output) for output in outputs)
        Executor.pseudo_free_tensor(outputs[1])
        assert outputs[1].shape == (1,)
        # Check the C++ allocation before either GC or backward can release it.
        assert storages[0].expired()
        actual = Executor.backward('storage', (source,), outputs, incoming)
        assert torch.equal(actual, 2 * torch.cos(source * 2) + 2)
        Executor.check_clear()
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize('copy_pack', [False, True])
def test_compiled_norm_retraces_after_original_input_is_pseudo_freed(copy_pack):
    torch._dynamo.reset()
    compilations = []

    def backend(graph, inputs):
        compilations.append(True)
        return graph.forward  # CPU eager backend still runs Dynamo/FakeTensor conversion.

    source = torch.arange(1., 13., dtype=torch.float64).reshape(3, 4).requires_grad_()
    weight = torch.linspace(0.5, 1.5, 4, dtype=torch.float64, requires_grad=True)

    def norm(value):
        return torch.nn.functional.rms_norm(value, (4,), weight, 1e-6)

    compiled_norm = torch.compile(norm, backend=backend)
    reference = source * 2
    expected_outputs = norm(reference), reference
    expected = torch.autograd.grad(expected_outputs, (source, weight),
                                   tuple(torch.ones_like(output) for output in expected_outputs))

    def region(value):
        return compiled_norm(value), value

    def segment(value):
        return checkpoint(region, value * 2)

    hooks = (torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach().clone(), lambda t: t)
             if copy_pack else nullcontext())
    try:
        with hooks:
            outputs = Executor.fexecute('compiled_norm', segment, source)
        incoming = tuple(torch.ones_like(output) for output in outputs)
        Executor.pseudo_free_tensor(outputs[1])
        assert outputs[1].shape == (1,)
        # Force Dynamo to inspect replay inputs after the original payload changed.
        torch._dynamo.reset()
        actual = Executor.backward('compiled_norm', (source,), outputs, incoming)
        assert len(compilations) == 2
        torch.testing.assert_close(actual, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, expected[1], rtol=0, atol=0)
        Executor.check_clear()
    finally:
        torch._dynamo.reset()


class _NormPassthrough(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(4, dtype=torch.float64)

    def forward(self, x):
        value = x * 2
        return self.norm(value), value


@replace_all_device_with('cpu')
@pytest.mark.parametrize('offload', [False, True])
def test_generated_checkpoint_survives_pseudo_free_of_passthrough_output(tmp_path, offload):
    from nnscaler.runtime.cpu_offloading import CPUOffloadConfig

    model = _NormPassthrough()
    value = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4).requires_grad_()
    expected_outputs = model(value)
    incoming = tuple(torch.ones_like(output) for output in expected_outputs)
    expected = torch.autograd.grad(expected_outputs, (value, *model.parameters()), incoming)

    def policy(graph, config):
        for op in get_pas_ops(graph):
            selected = torch.nn.LayerNorm in op.module_class_chain
            yield OpPlan(op, recompute_id=0 if selected else -1, stage_id=0,
                         offload_id=0 if selected and offload else -1)

    with mock_cube_env(0, 1):
        torch.distributed.init_process_group('gloo', rank=0, world_size=1,
                                            init_method=(tmp_path / 'rendezvous').as_uri())
        try:
            compiled = parallelize(copy.deepcopy(model), {'x': value.detach().clone().requires_grad_()},
                                   policy, ComputeConfig(1, 1, cpu_offload=CPUOffloadConfig() if offload else None),
                                   gen_savedir=tmp_path, instance_name='stable_checkpoint', load_module=True)
            source = next(tmp_path.rglob('gencode0.py')).read_text()
            assert 'import nnscaler.runtime.checkpoint as ckpt' in source
            assert 'multiref(' in source and 'def recompute(' in source
            if offload:
                assert '_offload.checkpoint(' in source
                store = _CPUStore('group')
                compiled.cpu_offloading_hooks = lambda: SavedActivationContext(store)
            outputs = Executor.fexecute('compiled', compiled, value)
            assert outputs[1]._base is None
            Executor.pseudo_free_tensor(outputs[1])
            assert outputs[1].shape == (1,)
            actual = Executor.backward('compiled', (value,), outputs, incoming)
            torch.testing.assert_close(actual, expected[0], rtol=1e-14, atol=1e-14)
            for parameter, gradient in zip(compiled.parameters(), expected[1:]):
                torch.testing.assert_close(parameter.grad, gradient, rtol=1e-14, atol=1e-14)
            Executor.check_clear()
        finally:
            torch.distributed.destroy_process_group()
