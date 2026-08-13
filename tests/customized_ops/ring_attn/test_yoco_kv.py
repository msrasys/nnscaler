from functools import partial
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch import nn

import nnscaler.customized_ops.ring_attention.yoco_kv as yoco_kv_module
from nnscaler.codegen.emit import FuncEmission
from nnscaler.customized_ops.ring_attention.yoco_kv import (
    wrap_yoco_kv_allgather,
)
from nnscaler.graph.parser.converter import convert_model
from nnscaler.ir.operator import IRFwOperation
from nnscaler.runtime.device import DeviceGroup
from tests.launch_torchrun import torchrun


def test_yoco_kv_allgather_fuses_kv_into_one_collective(monkeypatch):
    key = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)
    value = key + 100
    calls = []

    def fake_gather(tensor, dim, ranks):
        calls.append((tensor.detach().clone(), dim, tuple(ranks)))
        return torch.cat((tensor, tensor), dim=dim)

    monkeypatch.setattr(
        yoco_kv_module, 'allgather_reducescatter', fake_gather
    )

    gathered_key, gathered_value = wrap_yoco_kv_allgather(
        key,
        value,
        process_group=(2, 3),
        cp_size=2,
    )

    assert len(calls) == 1
    fused_input, dim, ranks = calls[0]
    assert dim == 0
    assert ranks == (2, 3)
    assert torch.equal(fused_input, torch.cat((key, value), dim=1))
    assert torch.equal(gathered_key, torch.cat((key, key), dim=0))
    assert torch.equal(gathered_value, torch.cat((value, value), dim=0))


def _shared_kv_gradients(key, value, gather_calls, consumers):
    gathered_key = gathered_value = None
    losses = []
    for _ in range(gather_calls):
        gathered_key, gathered_value = wrap_yoco_kv_allgather(
            key,
            value,
            process_group=(0, 1),
            cp_size=2,
        )
        losses.append((gathered_key, gathered_value))

    loss = key.new_zeros(())
    for index in range(consumers):
        current_key, current_value = losses[index % gather_calls]
        loss = (
            loss
            + (index + 1) * current_key.square().sum()
            + (index + 2) * current_value.square().sum()
        )
    loss.backward()
    return key.grad.detach().clone(), value.grad.detach().clone()


def test_shared_gather_gradient_matches_per_layer_gather(monkeypatch):
    collective_calls = {'forward': 0, 'backward': 0}

    class CountingGather(torch.autograd.Function):

        @staticmethod
        def forward(ctx, tensor, dim, ranks):
            ctx.dim = dim
            ctx.group_size = len(ranks)
            collective_calls['forward'] += 1
            return torch.cat((tensor,) * ctx.group_size, dim=dim)

        @staticmethod
        def backward(ctx, grad):
            collective_calls['backward'] += 1
            chunks = torch.chunk(grad, ctx.group_size, dim=ctx.dim)
            return torch.stack(chunks).sum(dim=0), None, None

    def differentiable_fake_gather(tensor, dim, ranks):
        return CountingGather.apply(tensor, dim, tuple(ranks))

    monkeypatch.setattr(
        yoco_kv_module,
        'allgather_reducescatter',
        differentiable_fake_gather,
    )
    base_key = torch.randn(3, 2, 4)
    base_value = torch.randn(3, 2, 4)

    shared_grads = _shared_kv_gradients(
        base_key.clone().requires_grad_(True),
        base_value.clone().requires_grad_(True),
        gather_calls=1,
        consumers=4,
    )
    assert collective_calls == {'forward': 1, 'backward': 1}

    collective_calls.update(forward=0, backward=0)
    legacy_grads = _shared_kv_gradients(
        base_key.clone().requires_grad_(True),
        base_value.clone().requires_grad_(True),
        gather_calls=4,
        consumers=4,
    )
    assert collective_calls == {'forward': 4, 'backward': 4}
    torch.testing.assert_close(shared_grads[0], legacy_grads[0])
    torch.testing.assert_close(shared_grads[1], legacy_grads[1])


def test_yoco_kv_allgather_cp_one_is_identity(monkeypatch):
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)

    def unexpected_gather(*args, **kwargs):
        raise AssertionError('CP1 must not gather YOCO K/V')

    monkeypatch.setattr(
        yoco_kv_module, 'allgather_reducescatter', unexpected_gather
    )
    gathered_key, gathered_value = wrap_yoco_kv_allgather(
        key,
        value,
        process_group=(0,),
        cp_size=1,
    )

    assert gathered_key is key
    assert gathered_value is value


class _YocoKVModule(nn.Module):

    def forward(self, key, value):
        return wrap_yoco_kv_allgather(
            key,
            value,
            cp_size=2,
            require_full_plan_sequence_partition=True,
        )


def test_yoco_kv_cp_group_survives_parser_partition_and_codegen():
    key = torch.randn(16, 2, 8)
    value = torch.randn(16, 2, 8)
    with tempfile.TemporaryDirectory() as savedir:
        graph = convert_model(
            _YocoKVModule(),
            {'key': key, 'value': value},
            savedir,
            constant_folding=False,
        )

    node = next(
        node
        for node in graph.select(ntype=IRFwOperation)
        if node.fn is wrap_yoco_kv_allgather
    )
    assert node.kwargs['cp_size'] == 2
    assert node.kwargs['require_full_plan_sequence_partition'] is True

    sub_nodes = graph.partition(
        node,
        node.algorithm('dim'),
        idx=0,
        dim=0,
        num=8,
    )
    emitted = FuncEmission().emit_fnode(
        sub_nodes[5],
        runtime_devid=5,
        plan_ndevs=8,
        runtime_ndevs=8,
    )[-1]
    assert 'cp_size=2' in emitted
    assert 'require_full_plan_sequence_partition=True' in emitted
    assert 'process_group=[4, 5]' in emitted


def _cp2_ep4_yoco_kv_worker():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    device_group = DeviceGroup()
    device_group.get_group([0, 1])
    device_group.get_group([2, 3])
    group_start = rank // 2 * 2
    cp_ranks = (group_start, group_start + 1)

    key = torch.full(
        (2, 1, 4), float(rank + 1), device=device, requires_grad=True
    )
    value = torch.full(
        (2, 1, 4), float(rank + 11), device=device, requires_grad=True
    )
    gathered_key, gathered_value = wrap_yoco_kv_allgather(
        key,
        value,
        process_group=cp_ranks,
        cp_size=2,
    )

    expected_key = torch.cat([
        torch.full_like(key, float(peer + 1))
        for peer in cp_ranks
    ])
    expected_value = torch.cat([
        torch.full_like(value, float(peer + 11))
        for peer in cp_ranks
    ])
    torch.testing.assert_close(gathered_key, expected_key)
    torch.testing.assert_close(gathered_value, expected_value)

    weight = float(rank + 1)
    (weight * gathered_key.sum() + 10 * weight * gathered_value.sum()).backward()
    expected_grad = float(sum(peer + 1 for peer in cp_ranks))
    torch.testing.assert_close(key.grad, torch.full_like(key, expected_grad))
    torch.testing.assert_close(
        value.grad, torch.full_like(value, 10 * expected_grad)
    )

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason='needs four GPUs for CP2/EP4 lane-isolation validation',
)
def test_cp2_ep4_yoco_kv_forward_and_gradients():
    partial(torchrun, 4, _cp2_ep4_yoco_kv_worker)()
