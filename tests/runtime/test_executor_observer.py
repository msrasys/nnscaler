# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
from contextvars import Context
import weakref

import pytest
import torch

from nnscaler.runtime.executor import Executor, observe_forward_outputs


@pytest.fixture(autouse=True)
def clear_executor():
    Executor.clear()
    yield
    Executor.clear()


@pytest.mark.parametrize('training', [False, True])
def test_observer_preserves_output_structure_identity_and_gradients(training):
    seen = []
    def observe(name, outputs):
        seen.append((name, id(outputs), weakref.ref(outputs['value'])))
        return object()  # The observer's return value cannot replace outputs.
    x = torch.ones(4, requires_grad=training)
    with observe_forward_outputs(observe):
        outputs = Executor.fexecute('s', lambda v: {'value': v * 3}, x, requires_grad=training)
    assert seen[0][:2] == ('s', id(outputs)) and seen[0][2]() is outputs['value']
    if training:
        actual = Executor.backward('s', (x,), (outputs['value'],), (torch.ones(4),))
        torch.testing.assert_close(actual, torch.full_like(x, 3), rtol=0, atol=0)
    else:
        assert not outputs['value'].requires_grad
    del outputs
    assert seen[0][2]() is None
    Executor.check_clear()


def test_nested_observers_compose_and_do_not_escape_their_context():
    seen = []
    x = torch.ones(4)
    def run(name):
        return Executor.fexecute(name, lambda v: v * 2, x, requires_grad=False)
    with observe_forward_outputs(lambda name, _: seen.append(('outer', name))):
        run('a')
        with observe_forward_outputs(lambda name, _: seen.append(('inner', name))):
            run('b')
        Context().run(run, 'isolated')
        run('c')
    run('after')
    assert seen == [('outer', 'a'), ('outer', 'b'), ('inner', 'b'), ('outer', 'c')]


@pytest.mark.parametrize('release_inputs', [(), (0,)])
def test_failed_observer_rolls_back_only_its_forward_and_is_removed(release_inputs):
    x = torch.ones(4, requires_grad=True)
    first = Executor.fexecute('s', lambda v: v * 2, x)
    def fail(name, outputs):
        raise RuntimeError('observer failed')
    with pytest.raises(RuntimeError, match='observer failed'):
        with observe_forward_outputs(fail):
            Executor.fexecute('s', lambda v: v * 3, x, release_inputs=release_inputs)
    assert len(Executor._detach['s']) == 1
    actual = Executor.backward('s', (x,), (first,), (torch.ones(4),))
    torch.testing.assert_close(actual, torch.full_like(x, 2), rtol=0, atol=0)
    output = Executor.fexecute('after', lambda v: v * 4, x)
    Executor.backward('after', (x,), (output,), (torch.ones(4),))
    Executor.check_clear()
