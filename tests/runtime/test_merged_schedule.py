import copy
import os

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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='requires CUDA streams',
)
@pytest.mark.parametrize('defer_step_data_release', (False, True))
def test_schedule_node_release_drops_runtime_references(defer_step_data_release):
    stream = torch.cuda.Stream()
    event = torch.cuda.Event()
    x = torch.randn(8, device='cuda', requires_grad=True)

    def forward_fn(t):
        return t * t

    node = ScheduleNode(forward_fn, stream, event, name='release_state')
    step_data = {'tensor': torch.ones(8, device='cuda')}
    node.step_data = step_data
    node.loss_aux_tensors = (step_data['tensor'],)
    node.loss_aux_step_data = {'aux': step_data['tensor']}
    if defer_step_data_release:
        node._defer_step_data_release = True

    out = node.forward((x,))
    grad = node.backward(torch.ones_like(out))
    torch.cuda.synchronize()

    torch.testing.assert_close(grad, 2 * x.detach())
    assert not hasattr(node, 'forward_func')
    assert not hasattr(node, 'backward_func')
    assert not hasattr(node, 'loss_aux_tensors')
    assert not hasattr(node, 'loss_aux_step_data')

    if defer_step_data_release:
        assert getattr(node, 'step_data') is step_data
        assert 'tensor' in step_data
        node._release_step_data()

    assert not hasattr(node, 'step_data')
    assert step_data == {}


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
        if self.step_data is not None:
            self.step_data['_combine_shared_expert_out'] = _detach_for_layer_state(shared_out)
        return expert_out, shared_out

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
            layer.stream_counts = {'comp': 0, 'comm': 0, 'all_reduce': 0}

    def stream_counts(self):
        counts = {'comp': 0, 'comm': 0, 'all_reduce': 0}
        for layer in self.layers:
            counts['comp'] += layer.stream_counts['comp']
            counts['comm'] += layer.stream_counts['comm']
            counts['all_reduce'] += layer.stream_counts['all_reduce']
        return counts

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
    return losses, _distributed_grad_norm(model), scheduler.merged_4phase_calls, model.stream_counts()


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
            merged_losses, merged_gnorm, merged_4phase_calls, stream_counts = _run_merged(
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
            assert stream_counts['comp'] > 0
            assert stream_counts['comm'] > 0
            assert stream_counts['all_reduce'] > 0
            case_results.append({
                'use_checkpoint': use_checkpoint,
                'use_aux_loss': use_aux_loss,
                'early_attn_memory_release': early_attn_memory_release,
                'async_4phase': async_4phase,
                'merged_4phase_calls': merged_4phase_calls,
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
            assert case['stream_counts']['comp'] > 0
            assert case['stream_counts']['comm'] > 0
            assert case['stream_counts']['all_reduce'] > 0
