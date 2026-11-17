import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from nnscaler.runtime.kernel_graph import (
    kernel_graph_scope, run_module_kernel_graph, prepare_module_kernel_graphs,
    clear_module_kernel_graphs, module_kernel_graph_stats,
)
from nnscaler.runtime.stream import ModuleStreams


@pytest.fixture(autouse=True)
def clear_graphs():
    clear_module_kernel_graphs()
    yield
    clear_module_kernel_graphs()


def test_disabled_kernel_graph_keeps_ordinary_autograd():
    x = torch.randn(4, requires_grad=True)
    run_module_kernel_graph(torch.square, (x,)).sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    assert module_kernel_graph_stats()['pending'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_registration_and_replay_do_not_wait_for_unrelated_communication():
    x = torch.randn(64, 64, device='cuda')
    weight = torch.randn_like(x)
    side, comm = torch.cuda.Stream(), torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm = x @ weight
    torch.cuda.synchronize()
    del warm
    for replay in (False, True):
        finished, done = torch.cuda.Event(), torch.cuda.Event()
        with torch.cuda.stream(comm):
            torch.cuda._sleep(1_000_000_000)
            finished.record()
        with torch.cuda.stream(side), torch.no_grad(), kernel_graph_scope(True):
            output = run_module_kernel_graph(torch.mm, (x, weight))
            done.record()
        done.synchronize()
        assert not finished.query(), 'registration/replay synchronized unrelated communication'
        torch.testing.assert_close(output, x @ weight)
        torch.cuda.synchronize()
        if not replay:
            assert module_kernel_graph_stats()['graphs'] == 0
            cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            assert prepare_module_kernel_graphs() == 1
            torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng)
            torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng)
    assert module_kernel_graph_stats()['replays'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_replay_refreshes_weights_and_preserves_multiple_live_autograd_calls():
    class Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            ctx.save_for_backward(x, weight)
            return run_module_kernel_graph(torch.mm, (x, weight))

        @staticmethod
        def backward(ctx, grad):
            x, weight = ctx.saved_tensors
            return grad @ weight.t(), x.t() @ grad

    weight = torch.nn.Parameter(torch.randn(16, 16, device='cuda'))

    def forward(x):
        with ModuleStreams() as streams:
            with streams.fork('linear', 'graph_test', (x, weight), kernel_graphs=True,
                              preserve_accumulators=True):
                y = Linear.apply(x, weight)
            streams.publish((y,))
            streams.join('linear')
        return y

    sample = torch.randn(8, 16, device='cuda', requires_grad=True)
    reference = weakref.ref(sample)
    y = forward(sample)
    y.sum().backward()
    del sample, y
    assert reference() is None, 'pending warmup retained the original autograd graph'
    assert prepare_module_kernel_graphs() == 1
    weight.grad = None
    with torch.no_grad():
        weight.mul_(.7)
    ref_weight = weight.detach().clone().requires_grad_()
    inputs = [torch.randn(8, 16, device='cuda', requires_grad=True) for _ in range(2)]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    actual = [forward(x) for x in inputs]
    expected = [x @ ref_weight for x in refs]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    sum(y.square().sum() for y in actual).backward()
    sum(y.square().sum() for y in expected).backward()
    torch.testing.assert_close(weight.grad, ref_weight.grad)
    for x, ref in zip(inputs, refs):
        torch.testing.assert_close(x.grad, ref.grad)
    assert module_kernel_graph_stats()['replays'] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_capacity_is_bounded_and_outputs_do_not_alias_replay_storage():
    inputs = [torch.randn(n, device='cuda') for n in range(4, 9)]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad(), kernel_graph_scope(True):
        for x in inputs:
            torch.testing.assert_close(run_module_kernel_graph(torch.neg, (x,)), -x)
    assert prepare_module_kernel_graphs() == 4
    assert module_kernel_graph_stats()['graphs'] == 4
    assert module_kernel_graph_stats()['capacity_fallbacks'] == 1
    with torch.cuda.stream(side), torch.no_grad(), kernel_graph_scope(True):
        first = run_module_kernel_graph(torch.neg, (inputs[0],))
        first.fill_(100)
        second = run_module_kernel_graph(torch.neg, (inputs[0],))
        torch.testing.assert_close(second, -inputs[0])
        torch.testing.assert_close(first, torch.full_like(first, 100))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_host_callers_cannot_mix_staging_buffers_on_one_stream():
    side = torch.cuda.Stream()
    pairs = [(torch.full((32, 32), value, device='cuda'), torch.eye(32, device='cuda'))
             for value in (2., 7.)]
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad(), kernel_graph_scope(True):
        run_module_kernel_graph(torch.mm, pairs[0])
    assert prepare_module_kernel_graphs() == 1

    def run(pair):
        with torch.cuda.device(pair[0].device), torch.cuda.stream(side), torch.no_grad(), kernel_graph_scope(True):
            return [run_module_kernel_graph(torch.mm, pair) for _ in range(24)]

    with ThreadPoolExecutor(2) as workers:
        results = list(workers.map(run, pairs))
    torch.cuda.current_stream().wait_stream(side)
    for pair, outputs in zip(pairs, results):
        for output in outputs:
            torch.testing.assert_close(output, pair[0])
