# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import gc
import weakref

import pytest
import torch

from nnscaler.runtime.executor import AsyncCommHandler, Executor


class Work:
    def __init__(self, name, events, completed=True):
        self.name, self.events, self.completed = name, events, completed

    def is_completed(self):
        self.events.append((self.name, 'query'))
        return self.completed

    def wait(self):
        self.events.append((self.name, 'wait'))


@pytest.fixture(autouse=True)
def clear_state():
    Executor.clear()
    AsyncCommHandler().clear()
    yield
    AsyncCommHandler().clear()
    Executor.clear()


@pytest.mark.parametrize('pending_input', [False, True])
@pytest.mark.parametrize('requires_grad', [False, True])
def test_input_consumption_preserves_active_bundle_and_pseudo_free_lifetime(pending_input, requires_grad):
    handler = AsyncCommHandler()
    events = []
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3).requires_grad_()
    output = Executor.fexecute('producer', torch.sin, x)
    original_output = output.detach().clone()

    # Completed and incomplete unbundled sends must not be opportunistically
    # cleaned up while the scheduler is still constructing the current bundle.
    done = Work('old_done', events)
    pending = Work('old_pending', events, completed=False)
    handler.hold_send(torch.ones(2), done, lambda: events.append(('old_done', 'callback')))
    handler.hold_send(torch.ones(2), pending, lambda: events.append(('old_pending', 'callback')))

    handler.begin_send_bundle('current_boundary')
    Executor.defer_pseudo_free_tensor(output)
    output_ref = weakref.ref(output)

    def release_output():
        events.append(('current_send', 'callback'))
        Executor.complete_deferred_pseudo_free_tensor(output_ref())

    handler.hold_send(output, Work('current_send', events), release_output)
    output_id = id(output)
    del output
    gc.collect()
    assert output_ref() is not None

    received = torch.tensor([3., 5.], requires_grad=True)
    placeholder = torch.zeros(2) if pending_input else received
    if pending_input:
        handler.submit(placeholder, [Work('input', events)], lambda _: received)
    actual = Executor.aexecute(lambda a, b, scale: (a + b) * scale,
                               placeholder, placeholder, 3, requires_grad=requires_grad)
    torch.testing.assert_close(actual, received * 6)
    assert actual.requires_grad == requires_grad
    assert events == ([('input', 'wait')] if pending_input else [])
    assert handler._active_send_bundle[0] == 'current_boundary'
    assert len(handler._active_send_bundle[1]) == 1
    assert id(handler._active_send_bundle[1][0][0]) == output_id
    assert Executor._pseudo_free_pending_sends[output_id] == 1
    torch.testing.assert_close(output_ref(), original_output)

    # Finishing the bundle transfers ownership to its queue; the ordinary
    # nonblocking cleanup outside a bundle still releases completed old sends,
    # never queued bundles or unfinished sends.
    handler.end_send_bundle()
    Executor.sync_tensors([])
    assert ('old_done', 'wait') in events
    assert ('old_done', 'callback') in events
    assert ('old_pending', 'wait') not in events
    assert ('current_send', 'wait') not in events
    assert ('current_send', 'callback') not in events
    assert len(handler._send_holds) == 1
    assert handler._send_holds[0][1] is pending
    assert len(handler._send_bundle_queues['current_boundary']) == 1
    output = output_ref()
    torch.testing.assert_close(output, original_output)

    # The scheduler's explicit capacity wait is the point that may release this
    # source. Its backward edge must remain usable after pseudo-free.
    handler.reserve_send_bundle('current_boundary', max_pending=1)
    assert output.shape == (1,)
    assert events.count(('current_send', 'wait')) == 1
    assert events.count(('current_send', 'callback')) == 1
    grad = Executor.backward('producer', [x], [output], [torch.ones_like(original_output)])
    torch.testing.assert_close(grad, torch.cos(x))
    handler.drain_sends()
    assert events.count(('old_pending', 'wait')) == 1
    assert events.count(('current_send', 'callback')) == 1
    Executor.check_clear()
    handler.check_clear()


def test_first_adapter_waits_for_input_then_adds_send_to_current_bundle():
    handler = AsyncCommHandler()
    events = []
    source = torch.zeros(2)
    completed = torch.tensor([7., 11.])
    handler.submit(source, [Work('input', events)], lambda _: completed)
    handler.begin_send_bundle('first_send')

    def adapter(tensor):
        torch.testing.assert_close(tensor, completed)
        handler.hold_send(tensor, Work('send', events), lambda: events.append(('send', 'callback')))

    Executor.aexecute(adapter, source, requires_grad=False)
    assert events == [('input', 'wait')]
    assert handler._active_send_bundle[0] == 'first_send'
    assert handler._active_send_bundle[1][0][0] is completed
    handler.end_send_bundle()
    handler.drain_sends()
    assert events == [('input', 'wait'), ('send', 'wait'), ('send', 'callback')]
    handler.check_clear()


@pytest.mark.parametrize('wait', [False, True])
def test_explicit_drain_still_rejects_unfinished_bundle(wait):
    handler = AsyncCommHandler()
    handler.begin_send_bundle('unfinished')
    with pytest.raises(RuntimeError, match='send bundle is active'):
        handler.drain_sends(wait=wait)
    handler.end_send_bundle()
    handler.check_clear()
