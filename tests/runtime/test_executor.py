import torch

from nnscaler.runtime.executor import Executor


def teardown_function():
    Executor.clear()


def test_fexecute_does_not_retain_non_grad_inputs():
    x = torch.randn(4, requires_grad=True)
    metadata = torch.randn(4)

    out = Executor.fexecute('segment', lambda x, metadata: x * 2, x, metadata)

    saved_pairs = Executor._detach['segment'][0]
    assert saved_pairs[0][0] == id(x)
    assert saved_pairs[0][1] is not None
    assert saved_pairs[1] == (id(metadata), None)

    grad_x = Executor.backward('segment', (x,), (out,), (torch.ones_like(out),))
    torch.testing.assert_close(grad_x, torch.full_like(x, 2))


def test_fexecute_no_grad_does_not_create_detach_entries():
    x = torch.randn(4, requires_grad=True)

    out = Executor.fexecute('segment', lambda x: x * 2, x, requires_grad=False)

    assert out.requires_grad is False
    assert 'segment' not in Executor._detach
