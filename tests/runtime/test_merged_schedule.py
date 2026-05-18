import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch.distributed as dist
from torch import nn

from nnscaler.runtime.merged_schedule import (
    LayerCallables,
    MergedScheduler,
    ScheduleNode,
    get_comm_stream,
    get_comp_stream,
    manual_sync_grads,
    set_streams,
)
from tests.launch_torchrun import launch_torchrun


def _detach_for_layer_state(tensor):
    if tensor is None:
        return None
    detached = tensor.detach()
    detached.requires_grad = tensor.requires_grad
    return detached


class _CountingMergedScheduler(MergedScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merged_4phase_calls = 0

    def _merged_step_4phase(self, *args, **kwargs):
        self.merged_4phase_calls += 1
        return super()._merged_step_4phase(*args, **kwargs)


def _assert_current_stream(expected_stream):
    current = torch.cuda.current_stream()
    assert current.cuda_stream == expected_stream.cuda_stream


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='requires CUDA streams',
)
def test_comp_stream_defaults_to_current_stream(monkeypatch):
    monkeypatch.delenv('NNSCALER_MOE_OVERLAP_DEDICATED_COMP_STREAM', raising=False)
    set_streams()

    default_stream = torch.cuda.current_stream()
    assert get_comp_stream().cuda_stream == default_stream.cuda_stream
    assert get_comm_stream().cuda_stream != default_stream.cuda_stream

    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        assert get_comp_stream().cuda_stream == side_stream.cuda_stream


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='requires CUDA streams',
)
def test_dedicated_comp_stream_mode_is_opt_in(monkeypatch):
    monkeypatch.setenv('NNSCALER_MOE_OVERLAP_DEDICATED_COMP_STREAM', '1')
    set_streams()

    dedicated_stream = get_comp_stream()
    assert dedicated_stream.cuda_stream != torch.cuda.current_stream().cuda_stream

    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        assert get_comp_stream().cuda_stream == dedicated_stream.cuda_stream

    monkeypatch.delenv('NNSCALER_MOE_OVERLAP_DEDICATED_COMP_STREAM', raising=False)
    set_streams()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='requires CUDA streams',
)
def test_scheduler_checkpoint_policy_can_target_body_only(monkeypatch):
    monkeypatch.delenv('NNSCALER_MOE_OVERLAP_DEDICATED_COMP_STREAM', raising=False)
    set_streams()
    scheduler = MergedScheduler(
        parallel_module=object(),
        num_layers=1,
        checkpoint_attn=False,
        checkpoint_body=True,
    )
    event = torch.cuda.Event()

    dense_nodes = scheduler._create_nodes(
        LayerCallables(
            attn_fn=lambda h: h,
            body_fn=lambda h: h,
            is_moe=False,
        ),
        event,
    )
    assert dense_nodes[0].checkpoint is False
    assert dense_nodes[1].checkpoint is True

    moe_nodes = scheduler._create_nodes_4(
        LayerCallables(
            attn_fn=lambda h: (h, h, h),
            dispatch_fn=lambda h, p: (h, p),
            expert_fn=lambda h, p, h_ln: (h, h_ln),
            combine_fn=lambda h: h,
            is_moe=True,
        ),
        event,
    )
    assert moe_nodes[0].checkpoint is False
    assert moe_nodes[1].checkpoint is False
    assert moe_nodes[2].checkpoint is True
    assert moe_nodes[3].checkpoint is False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='requires CUDA streams',
)
def test_scheduler_use_checkpoint_remains_all_compute_nodes(monkeypatch):
    monkeypatch.delenv('NNSCALER_MOE_OVERLAP_DEDICATED_COMP_STREAM', raising=False)
    set_streams()
    scheduler = MergedScheduler(
        parallel_module=object(),
        num_layers=1,
        use_checkpoint=True,
    )
    event = torch.cuda.Event()

    dense_nodes = scheduler._create_nodes(
        LayerCallables(
            attn_fn=lambda h: h,
            body_fn=lambda h: h,
            is_moe=False,
        ),
        event,
    )
    assert dense_nodes[0].checkpoint is True
    assert dense_nodes[1].checkpoint is True


def test_overlap_dispatch_backward_with_expert_wgrad_async_order():
    trace = []
    dispatch_entered = threading.Event()
    release_dispatch = threading.Event()

    class DispatchNode:
        def backward(self, grads):
            assert grads == ('grad_tokens', 'grad_probs')
            trace.append('dispatch_bwd_enter')
            dispatch_entered.set()
            assert release_dispatch.wait(timeout=5)
            trace.append('dispatch_bwd_exit')
            return ('grad_h_ln', 'grad_router')

    class ExpertNode:
        def backward_dw(self):
            assert dispatch_entered.wait(timeout=5)
            trace.append('expert_wgrad')
            release_dispatch.set()

    scheduler = object.__new__(MergedScheduler)
    scheduler._async_pool = ThreadPoolExecutor(max_workers=1)
    try:
        dispatch_grads = scheduler._overlap_dispatch_backward_with_expert_wgrad(
            DispatchNode(), ExpertNode(),
            ('grad_tokens', 'grad_probs', 'grad_h_ln_from_expert'))
    finally:
        scheduler._async_pool.shutdown(wait=True)

    assert dispatch_grads == ('grad_h_ln', 'grad_router')
    assert trace == ['dispatch_bwd_enter', 'expert_wgrad', 'dispatch_bwd_exit']


def test_overlap_dispatch_backward_with_expert_wgrad_sequential_order():
    trace = []

    class DispatchNode:
        def backward(self, grads):
            assert grads == ('grad_tokens', 'grad_probs')
            trace.append('dispatch_bwd_enter')
            trace.append('dispatch_bwd_exit')
            return ('grad_h_ln', 'grad_router')

    class ExpertNode:
        def backward_dw(self):
            trace.append('expert_wgrad')

    scheduler = object.__new__(MergedScheduler)
    scheduler._async_pool = None
    dispatch_grads = scheduler._overlap_dispatch_backward_with_expert_wgrad(
        DispatchNode(), ExpertNode(),
        ('grad_tokens', 'grad_probs', 'grad_h_ln_from_expert'))

    assert dispatch_grads == ('grad_h_ln', 'grad_router')
    assert trace == ['dispatch_bwd_enter', 'dispatch_bwd_exit', 'expert_wgrad']


def test_manual_sync_grads_marks_parallel_module_clean():
    class Bucket:
        def __init__(self):
            self._async = True
            self.synced = 0
            self.reset_called = 0

        def sync_grads(self):
            assert self._async is False
            self.synced += 1

        def reset(self):
            self.reset_called += 1

    class Reducer:
        def __init__(self, bucket):
            self._buckets = [bucket]

    class ParallelModule:
        pass

    bucket = Bucket()
    module = ParallelModule()
    module._reducers = [Reducer(bucket)]
    module._sync_grad_required = True

    manual_sync_grads(module)

    assert bucket.synced == 1
    assert bucket.reset_called == 1
    assert bucket._async is True
    assert module._sync_grad_required is False


class _AuxOnlyBranch(torch.autograd.Function):
    backward_calls = 0

    @staticmethod
    def forward(ctx, x):
        return x.sin()

    @staticmethod
    def backward(ctx, grad_output):
        _AuxOnlyBranch.backward_calls += 1
        return torch.cos(grad_output)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='requires CUDA streams',
)
@pytest.mark.parametrize('checkpoint', (False, True))
def test_schedule_node_skips_outputs_with_none_grad(checkpoint):
    stream = torch.cuda.Stream()
    event = torch.cuda.Event()
    x = torch.randn(8, device='cuda', requires_grad=True)
    _AuxOnlyBranch.backward_calls = 0

    def forward_fn(t):
        return t * t, _AuxOnlyBranch.apply(t)

    node = ScheduleNode(
        forward_fn,
        stream,
        event,
        name='main_with_aux',
        checkpoint=checkpoint,
    )
    main, _aux = node.forward((x,))
    grad = node.backward((torch.ones_like(main), None))
    torch.cuda.synchronize()

    assert _AuxOnlyBranch.backward_calls == 0
    torch.testing.assert_close(grad, 2 * x.detach())


class _FakeMoELayer(nn.Module):
    def __init__(self, hidden_dim, use_aux_loss=False):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.router = nn.Linear(hidden_dim, hidden_dim)
        self.expert = nn.Linear(hidden_dim, hidden_dim)
        self.shared = nn.Linear(hidden_dim, hidden_dim)
        self.use_aux_loss = use_aux_loss
        self.step_data = None
        self.check_streams = False
        self.backward_dw_calls = 0
        self.delayed_wgrad_queue = []
        self.stream_counts = {'comp': 0, 'comm': 0, 'all_reduce': 0}

    def _check_comp_stream(self):
        if self.check_streams:
            _assert_current_stream(get_comp_stream())
            self.stream_counts['comp'] += 1

    def _check_comm_stream(self):
        if self.check_streams:
            _assert_current_stream(get_comm_stream())
            self.stream_counts['comm'] += 1

    def _all_reduce_avg_with_local_grad(self, tensor):
        # Keep the fake MoE collective in forward only. Autograd collectives make
        # the sequential and merged schedules launch backward collectives in
        # different orders, which is not what this test is trying to validate.
        world_size = dist.get_world_size()
        reduced = tensor.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        if self.check_streams:
            self.stream_counts['all_reduce'] += 1
        reduced = reduced / world_size
        return tensor + (reduced - tensor.detach())

    def attn_forward(self, h):
        self._check_comp_stream()
        h_ln = torch.tanh(self.attn(h))
        routing_probs = torch.sigmoid(self.router(h))
        if self.step_data is not None:
            self.step_data['_combine_residual'] = _detach_for_layer_state(h)
        if self.use_aux_loss:
            aux_loss = routing_probs.float().pow(2).mean()
            if self.step_data is not None:
                self.step_data['_loss_aux_tensors'] = (aux_loss,)
                self.step_data['gate_scores'] = aux_loss
            return h, h_ln, routing_probs, aux_loss
        return h, h_ln, routing_probs

    def dispatch_forward(self, h_ln, routing_probs):
        self._check_comm_stream()
        dispatched = self._all_reduce_avg_with_local_grad(h_ln * routing_probs)
        reduced_probs = self._all_reduce_avg_with_local_grad(routing_probs)
        return dispatched, reduced_probs

    def expert_forward(self, dispatched, reduced_probs, h_ln):
        self._check_comp_stream()
        expert_out = torch.relu(self.expert(dispatched)) * (reduced_probs + 0.5)
        shared_out = torch.tanh(self.shared(h_ln))
        if expert_out.requires_grad:
            expert_out.register_hook(self._queue_delayed_wgrad)
        if self.step_data is not None:
            self.step_data['_combine_shared_expert_out'] = _detach_for_layer_state(shared_out)
        return expert_out, shared_out

    def _queue_delayed_wgrad(self, grad):
        self.delayed_wgrad_queue.append(object())
        return grad

    def expert_backward_dw(self):
        self._check_comp_stream()
        assert self.delayed_wgrad_queue
        self.backward_dw_calls += 1
        self.delayed_wgrad_queue.clear()

    def combine_forward(self, expert_out, h_residual=None, shared_out=None):
        self._check_comm_stream()
        if h_residual is None:
            h_residual = self.step_data['_combine_residual']
            shared_out = self.step_data.get('_combine_shared_expert_out')
        combined = self._all_reduce_avg_with_local_grad(expert_out + shared_out)
        return h_residual + combined


class _FakeMoEModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=7, num_layers=2, use_aux_loss=False):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            _FakeMoELayer(hidden_dim, use_aux_loss=use_aux_loss)
            for _ in range(num_layers)
        )

    def set_stream_checks(self, enabled):
        for layer in self.layers:
            layer.check_streams = enabled
            layer.backward_dw_calls = 0
            layer.delayed_wgrad_queue.clear()
            layer.stream_counts = {'comp': 0, 'comm': 0, 'all_reduce': 0}

    def stream_counts(self):
        counts = {'comp': 0, 'comm': 0, 'all_reduce': 0}
        for layer in self.layers:
            counts['comp'] += layer.stream_counts['comp']
            counts['comm'] += layer.stream_counts['comm']
            counts['all_reduce'] += layer.stream_counts['all_reduce']
        return counts

    def backward_dw_calls(self):
        return sum(layer.backward_dw_calls for layer in self.layers)

    def pending_backward_dw_tasks(self):
        return sum(len(layer.delayed_wgrad_queue) for layer in self.layers)

    def forward(self, x, return_aux=False):
        h = self.embed(x)
        aux_losses = []
        for layer in self.layers:
            attn_out = layer.attn_forward(h)
            h_residual, h_ln, routing_probs = attn_out[:3]
            aux_losses.extend(
                t for t in attn_out[3:] if isinstance(t, torch.Tensor)
            )
            dispatched, reduced_probs = layer.dispatch_forward(h_ln, routing_probs)
            expert_out, shared_out = layer.expert_forward(
                dispatched, reduced_probs, h_ln)
            h = layer.combine_forward(expert_out, h_residual, shared_out)
        if return_aux:
            return h, aux_losses
        return h


def _distributed_grad_norm(model):
    params = list(model.parameters())
    total = torch.zeros((), device=params[0].device)
    for param in params:
        if param.grad is not None:
            total = total + param.grad.detach().float().pow(2).sum()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total.sqrt()


def _make_samples(device, rank):
    generator = torch.Generator(device=device)
    generator.manual_seed(100 + rank)
    samples = []
    for _ in range(2):
        samples.append({
            'x': torch.randn(4, 5, device=device, generator=generator),
            'target': torch.randn(4, 7, device=device, generator=generator),
        })
    return samples


def _run_sequential(model, samples, use_aux_loss=False):
    model.zero_grad(set_to_none=True)
    losses = []
    for sample in samples:
        if use_aux_loss:
            h, aux_losses = model(sample['x'], return_aux=True)
        else:
            h = model(sample['x'])
            aux_losses = ()
        loss = (h - sample['target']).pow(2).sum()
        if aux_losses:
            loss = loss + sum(aux_losses)
        losses.append(loss.detach())
        loss.backward()
    return torch.stack(losses), _distributed_grad_norm(model)


def _run_merged(model, samples, use_checkpoint=False, use_aux_loss=False,
                early_attn_memory_release=False, async_4phase=False):
    model.zero_grad(set_to_none=True)
    os.environ['ASYNC_4PHASE'] = '1' if async_4phase else '0'
    set_streams()
    model.set_stream_checks(True)
    scheduler = _CountingMergedScheduler(
        parallel_module=model,
        num_layers=len(model.layers),
        use_checkpoint=use_checkpoint,
        early_attn_memory_release=early_attn_memory_release,
    )

    def layer_callables_fn(step_idx, _sample):
        layer = model.layers[step_idx]
        step_data = {}

        def bind_step_data(fn):
            def wrapped(*args, _fn=fn, _layer=layer, _step_data=step_data):
                prev_step_data = _layer.step_data
                _layer.step_data = _step_data
                try:
                    return _fn(*args)
                finally:
                    _layer.step_data = prev_step_data
            return wrapped

        return LayerCallables(
            attn_fn=bind_step_data(layer.attn_forward),
            dispatch_fn=bind_step_data(layer.dispatch_forward),
            expert_fn=bind_step_data(layer.expert_forward),
            expert_backward_dw=bind_step_data(layer.expert_backward_dw),
            combine_fn=bind_step_data(layer.combine_forward),
            is_moe=True,
            step_data=step_data,
        )

    def embed_fn(sample):
        return model.embed(sample['x'])

    def loss_fn(h, sample, _routing_maps, expert_probs):
        output_info = {}
        aux_losses = tuple(
            t for t in expert_probs
            if use_aux_loss and isinstance(t, torch.Tensor)
        )

        def loss_forward(loss_input):
            _assert_current_stream(get_comp_stream())
            loss = (loss_input - sample['target']).pow(2).sum()
            if aux_losses:
                loss = loss + sum(aux_losses)
            output_info['output_tuple'] = (loss.detach(),)
            return loss

        loss_node = ScheduleNode(
            loss_forward,
            get_comp_stream(),
            torch.cuda.Event(),
            name='loss',
        )
        return loss_node, output_info

    results = scheduler.run(samples, layer_callables_fn, embed_fn, loss_fn)
    torch.cuda.synchronize()
    losses = torch.stack([result[0] for result in results])
    return (
        losses,
        _distributed_grad_norm(model),
        scheduler.merged_4phase_calls,
        model.stream_counts(),
        model.backward_dw_calls(),
        model.pending_backward_dw_tasks(),
    )


def _worker():
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    try:
        assert dist.get_world_size() >= 2
        rank = dist.get_rank()
        torch.manual_seed(10)
        case_results = []
        cases = [
            (False, False, False, False, 10),
            (True, False, False, False, 11),
            (False, True, False, False, 12),
            (True, True, False, False, 11),
            (False, True, True, False, 13),
            (False, True, False, True, 14),
        ]
        for (use_checkpoint, use_aux_loss, early_attn_memory_release,
             async_4phase, case_seed) in cases:
            torch.manual_seed(case_seed)
            sequential_model = _FakeMoEModel(use_aux_loss=use_aux_loss).to(device)
            merged_model = copy.deepcopy(sequential_model).to(device)
            samples = _make_samples(device, rank)

            sequential_losses, sequential_gnorm = _run_sequential(
                sequential_model, samples, use_aux_loss=use_aux_loss)
            (merged_losses, merged_gnorm, merged_4phase_calls, stream_counts,
             backward_dw_calls, pending_backward_dw_tasks) = _run_merged(
                merged_model, samples,
                use_checkpoint=use_checkpoint,
                use_aux_loss=use_aux_loss,
                early_attn_memory_release=early_attn_memory_release,
                async_4phase=async_4phase)

            case_label = (
                f'use_checkpoint={use_checkpoint}, use_aux_loss={use_aux_loss}, '
                f'early_attn_memory_release={early_attn_memory_release}, '
                f'async_4phase={async_4phase}'
            )
            torch.testing.assert_close(
                merged_losses, sequential_losses, rtol=1e-5, atol=1e-5,
                msg=case_label)

            for (seq_name, seq_param), (merged_name, merged_param) in zip(
                sequential_model.named_parameters(), merged_model.named_parameters()
            ):
                assert seq_name == merged_name
                torch.testing.assert_close(
                    merged_param.grad, seq_param.grad, rtol=1e-5, atol=1e-5,
                    msg=lambda msg: f'{case_label}, param={seq_name}\n{msg}')

            torch.testing.assert_close(
                merged_gnorm, sequential_gnorm, rtol=1e-5, atol=1e-5,
                msg=lambda msg: f'{case_label}\n{msg}')

            assert merged_4phase_calls == len(merged_model.layers)
            assert backward_dw_calls == len(samples) * len(merged_model.layers)
            assert pending_backward_dw_tasks == 0
            assert stream_counts['comp'] > 0
            assert stream_counts['comm'] > 0
            assert stream_counts['all_reduce'] > 0
            case_results.append({
                'use_checkpoint': use_checkpoint,
                'use_aux_loss': use_aux_loss,
                'early_attn_memory_release': early_attn_memory_release,
                'async_4phase': async_4phase,
                'merged_4phase_calls': merged_4phase_calls,
                'backward_dw_calls': backward_dw_calls,
                'pending_backward_dw_tasks': pending_backward_dw_tasks,
                'stream_counts': stream_counts,
            })
        return {
            'rank': rank,
            'cases': case_results,
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason='requires at least 2 CUDA devices for communication',
)
def test_merged_scheduler_fake_moe_matches_sequential_gnorm():
    results = launch_torchrun(2, _worker)
    assert len(results) == 2
    for result in results.values():
        assert len(result['cases']) == 6
        assert result['cases'][-3]['use_checkpoint']
        assert result['cases'][-2]['early_attn_memory_release']
        assert result['cases'][-1]['use_aux_loss']
        assert result['cases'][-1]['async_4phase']
        for case in result['cases']:
            assert case['merged_4phase_calls'] == 2
            assert case['backward_dw_calls'] == 4
            assert case['pending_backward_dw_tasks'] == 0
            assert case['stream_counts']['comp'] > 0
            assert case['stream_counts']['comm'] > 0
            assert case['stream_counts']['all_reduce'] > 0
