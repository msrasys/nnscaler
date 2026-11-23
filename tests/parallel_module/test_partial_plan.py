#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""A module whose policy uses only the first plan ranks leaves the other plan ranks inactive."""

import copy
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nnscaler.parallel import (
    ComputeConfig, build_optimizer, deduped_state_dict, load_deduped_state_dict, parallelize,
)
from nnscaler.policies import OpPartition, OpPlan, fn, get_pas_ops
from nnscaler.runtime.hybrid_optimizer import HybridOptConfig, HybridOptimizer, HybridSubOptConfig

from .common import init_distributed
from ..launch_torchrun import launch_torchrun
from ..utils import replace_all_device_with


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(3, 4, bias=False)
        self.down = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.down(self.up(x))


def _replicate(graph, cfg):
    return (OpPlan(node) for node in get_pas_ops(graph))


def _megatron_tp(graph, cfg):
    # column-parallel first linear, row-parallel second linear
    nlinears = 0
    for node in get_pas_ops(graph):
        if node.signature == 'torch.nn.functional.linear':
            yield OpPlan(node, partition=OpPartition(input=1, dim=0 if nlinears == 0 else 1))
            nlinears += 1
        else:
            yield OpPlan(node)


def _first_rank_only(graph, cfg):
    return fn(graph, replace(cfg, plan_ngpus=1), _replicate)


def _tp_on_first_two_ranks(graph, cfg):
    return fn(graph, replace(cfg, plan_ngpus=2), _megatron_tp)


def _gencode(gen_savedir, rank):
    return next(Path(gen_savedir).rglob(f'gencode{rank}.py')).read_text()


@replace_all_device_with('cpu')
def test_codegen_inactive_plan_ranks(tmp_path):
    # TP on plan ranks 0 and 1 of a 3-rank plan, with 2 replicas.
    parallelize(Mlp(), {'x': torch.ones(2, 3)}, _tp_on_first_two_ranks,
                ComputeConfig(3, 6), gen_savedir=tmp_path, load_module=False)
    for rank in (2, 5):
        assert 'is_inactive_rank = True' in _gencode(tmp_path, rank)
    # Replica offsets follow the full plan size (3), not the number of active ranks (2).
    for rank, tp_ranks, dp_ranks in [(0, [0, 1], [0, 3]), (3, [3, 4], [0, 3]), (4, [3, 4], [1, 4])]:
        code = _gencode(tmp_path, rank)
        assert 'is_inactive_rank' not in code
        assert re.search(rf'allreduce_identity\(\w+, ranks={re.escape(str(tp_ranks))}\)', code)
        assert f'Reducer(ranks={dp_ranks}' in code


class VLModel(nn.Module):
    """``encoder`` runs on plan rank 0 only; ``head`` runs on the full plan."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        loss = self.head(x).square().sum()
        if not getattr(self.encoder, 'is_inactive_rank', False):
            loss = loss + self.encoder(x).square().sum()
        return loss


def _hybrid_param_clss(name):
    return (0, 0) if name.startswith('encoder.') else (1, 0)


def _build(mode, gen_savedir, encoder, head):
    config = ComputeConfig(2, 4, use_zero={'zero1': 1, 'zero3': 3}.get(mode, 0))
    # A hybrid optimizer builds the reducer buckets itself from param classes.
    kwargs = dict(gen_savedir=gen_savedir, build_module_buckets=mode != 'hybrid')
    model = VLModel(
        parallelize(encoder, {'x': torch.ones(2, 3)}, _first_rank_only, config, **kwargs),
        parallelize(head, {'x': torch.ones(2, 3)}, _megatron_tp, config, **kwargs),
    )
    if mode == 'hybrid':
        optimizer = build_optimizer(
            model, HybridOptimizer, param_clss_fn=_hybrid_param_clss,
            config=HybridOptConfig(optimizers=[
                HybridSubOptConfig(type=torch.optim.Adam, options={'lr': 0.01}),
                HybridSubOptConfig(type=torch.optim.SGD, options={'lr': 0.1}),
            ]),
        )
    else:
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    return model, optimizer


def _inactive_rank_training_worker(tmp_path, mode):
    init_distributed()
    rank = torch.distributed.get_rank()
    torch.manual_seed(0)
    encoder, head = Encoder(), Mlp()
    ref_encoder, ref_head = copy.deepcopy(encoder), copy.deepcopy(head)
    if mode == 'hybrid':
        ref_optimizers = [torch.optim.Adam(ref_encoder.parameters(), lr=0.01),
                          torch.optim.SGD(ref_head.parameters(), lr=0.1)]
    else:
        ref_optimizers = [torch.optim.Adam([*ref_encoder.parameters(), *ref_head.parameters()], lr=0.01)]

    model, optimizer = _build(mode, tmp_path, encoder, head)
    active = rank % 2 == 0
    assert model.encoder.is_inactive_rank == (not active)
    if not active:
        assert not list(model.encoder.parameters())
        with pytest.raises(RuntimeError, match='rank is inactive'):
            model.encoder(torch.ones(2, 3))

    for step in range(3):
        if step == 1 and mode == 'deduped':
            # Reload into freshly initialized modules to show the checkpoint restores everything.
            model_state, opt_state = deduped_state_dict(model, optimizer)
            torch.manual_seed(1)
            model, optimizer = _build(mode, tmp_path, Encoder(), Mlp())
            load_deduped_state_dict(model, model_state, optimizer, opt_state)

        samples = [torch.rand(2, 3, device='cpu', generator=torch.Generator().manual_seed(step * 2 + i)).cuda()
                   for i in range(2)]
        x = samples[rank // 2]
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        gnorm = optimizer.clip_gnorm()
        optimizer.step()

        for ref_optimizer in ref_optimizers:
            ref_optimizer.zero_grad()
        ref_head_losses = [ref_head(s).square().sum() for s in samples]
        ref_encoder_losses = [ref_encoder(s).square().sum() for s in samples]
        sum(ref_head_losses + ref_encoder_losses).backward()
        ref_params = [*ref_encoder.parameters(), *ref_head.parameters()]
        ref_gnorm = torch.cat([p.grad.flatten() for p in ref_params]).norm()
        for ref_optimizer in ref_optimizers:
            ref_optimizer.step()

        expected = ref_head_losses[rank // 2] + (ref_encoder_losses[rank // 2] if active else 0)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(gnorm, ref_gnorm)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
@pytest.mark.parametrize('mode', ['zero1', 'zero3', 'hybrid', 'deduped'])
def test_inactive_rank_training_matches_eager(tmp_path, mode):
    launch_torchrun(4, _inactive_rank_training_worker, tmp_path, mode)
