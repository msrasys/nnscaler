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
    ComputeConfig, build_optimizer, deduped_state_dict, gather_full_model_state_dict,
    load_deduped_state_dict, load_merged_state_dict, merge_state_dicts, parallelize,
)
from nnscaler.policies import OpPartition, OpPlan, fn, get_pas_ops
from nnscaler.runtime.hybrid_optimizer import HybridOptConfig, HybridOptimizer, HybridSubOptConfig

from .common import init_distributed
from ..launch_torchrun import clone_to_cpu_recursively, launch_torchrun
from ..utils import replace_all_device_with


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


class ScaledEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.full((3,), 2.0), persistent=False)

    def forward(self, x):
        return super().forward(x) * self.scale


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


def _plan_on_first_ranks(graph, cfg, nranks, plans):
    # Shrink the plan but keep the number of scale units,
    # so the reduced config stays valid and describes the same replicas.
    return fn(graph, replace(cfg, plan_ngpus=nranks, runtime_ngpus=cfg.num_scale_units * nranks), plans)


def _first_rank_only(graph, cfg):
    return _plan_on_first_ranks(graph, cfg, 1, _replicate)


def _tp_on_first_two_ranks(graph, cfg):
    return _plan_on_first_ranks(graph, cfg, 2, _megatron_tp)


def _first_three_ranks(graph, cfg):
    return _plan_on_first_ranks(graph, cfg, 3, _replicate)


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


@replace_all_device_with('cpu')
def test_codegen_active_ranks_not_dividing_runtime(tmp_path):
    # 3 active ranks in a plan of 4 with a single scale unit:
    # a reduced config with runtime_ngpus=4 and plan_ngpus=3 would be invalid.
    parallelize(Encoder(), {'x': torch.ones(2, 3)}, _first_three_ranks,
                ComputeConfig(4, 4), gen_savedir=tmp_path, load_module=False)
    for rank in range(3):
        assert 'is_inactive_rank' not in _gencode(tmp_path, rank)
    assert 'is_inactive_rank = True' in _gencode(tmp_path, 3)


@replace_all_device_with('cpu')
def test_codegen_declares_optimizer_groups_of_inactive_ranks(tmp_path):
    # ZeRO-1 with 2 groups broadcasts optimizer states with stride 8 // 2 when loading checkpoints;
    # the groups at plan rank 1, where the encoder is inactive, must be declared as well.
    parallelize(Encoder(), {'x': torch.ones(2, 3)}, _first_rank_only,
                ComputeConfig(2, 8, use_zero=1, zero_ngroups=2), gen_savedir=tmp_path, load_module=False)
    for rank in range(8):
        code = _gencode(tmp_path, rank)
        for stride_group in ([0, 4], [1, 5], [2, 6], [3, 7]):
            assert f'comm_groups.append({stride_group})' in code


class VLModel(nn.Module):
    """``encoder`` runs on plan rank 0 only; ``head`` runs on the full plan.

    Optional non-parallel layers: ``frozen`` is registered first,
    so its parameters precede every trainable parameter.
    """

    def __init__(self, encoder, head, frozen=None, post=None):
        super().__init__()
        if frozen is not None:
            self.frozen = frozen
        self.encoder = encoder
        self.head = head
        if post is not None:
            self.post = post

    def forward(self, x):
        loss = self.head(x).square().sum()
        if not getattr(self.encoder, 'is_inactive_rank', False):
            loss = loss + self.encoder(x).square().sum()
        if hasattr(self, 'post'):
            loss = loss + self.post(self.frozen(x)).square().sum()
        return loss


def _hybrid_param_clss(name):
    return (0, 0) if name.startswith('encoder.') else (1, 0)


def _plain_layers(mode):
    # Non-parallel layers after the encoder sit at different local optimizer indices
    # on active and inactive ranks, which checkpoint reloading must map back.
    if mode not in ('deduped', 'npp_zero1'):
        return {}
    return {'frozen': nn.Linear(3, 3).requires_grad_(False), 'post': nn.Linear(3, 3)}


def _build(mode, gen_savedir, encoder, head, plain):
    config = ComputeConfig(2, 4, use_zero={'zero1': 1, 'zero3': 3}.get(mode, 0))
    # A hybrid optimizer builds the reducer buckets itself from param classes.
    kwargs = dict(gen_savedir=gen_savedir, build_module_buckets=mode != 'hybrid')
    model = VLModel(
        parallelize(encoder, {'x': torch.ones(2, 3)}, _first_rank_only, config, **kwargs),
        parallelize(head, {'x': torch.ones(2, 3)}, _megatron_tp, config, **kwargs),
        **plain,
    )
    if mode == 'npp_zero1':
        # ZeRO-1 for the non-parallel layers only.
        optimizer = build_optimizer(model, torch.optim.Adam, replace(config, use_zero=1), lr=0.01)
    elif mode == 'hybrid':
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
    encoder, head, plain = Encoder(), Mlp(), _plain_layers(mode)
    ref_encoder, ref_head, ref_plain = copy.deepcopy((encoder, head, plain))
    ref_trainable = [p for m in (ref_encoder, ref_head, *ref_plain.values())
                     for p in m.parameters() if p.requires_grad]
    if mode == 'hybrid':
        ref_optimizers = [torch.optim.Adam(ref_encoder.parameters(), lr=0.01),
                          torch.optim.SGD(ref_head.parameters(), lr=0.1)]
    else:
        ref_optimizers = [torch.optim.Adam(ref_trainable, lr=0.01)]

    model, optimizer = _build(mode, tmp_path, encoder, head, plain)
    active = rank % 2 == 0
    assert model.encoder.is_inactive_rank == (not active)
    if not active:
        assert not list(model.encoder.parameters())
        with pytest.raises(RuntimeError, match='rank is inactive'):
            model.encoder(torch.ones(2, 3))

    for step in range(3):
        # Reload checkpoints into freshly initialized modules to show they restore everything.
        if step == 1 and mode == 'deduped':
            model_state, opt_state = deduped_state_dict(model, optimizer)
            torch.manual_seed(1)
            model, optimizer = _build(mode, tmp_path, Encoder(), Mlp(), _plain_layers(mode))
            load_deduped_state_dict(model, model_state, optimizer, opt_state)
        if step == 1 and mode == 'npp_zero1':
            states = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(
                states, clone_to_cpu_recursively((model.state_dict(), optimizer.state_dict())))
            model_state, opt_state = merge_state_dicts(*zip(*states))
            torch.manual_seed(1)
            model, optimizer = _build(mode, tmp_path, Encoder(), Mlp(), _plain_layers(mode))
            load_merged_state_dict(model, model_state, optimizer, opt_state)

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
        if ref_plain:
            ref_head_losses = [loss + ref_plain['post'](ref_plain['frozen'](s)).square().sum()
                               for loss, s in zip(ref_head_losses, samples)]
        ref_encoder_losses = [ref_encoder(s).square().sum() for s in samples]
        sum(ref_head_losses + ref_encoder_losses).backward()
        ref_gnorm = torch.cat([p.grad.flatten() for p in ref_trainable]).norm()
        for ref_optimizer in ref_optimizers:
            ref_optimizer.step()

        # Every rank computes the full-plan part; only active ranks add the encoder loss.
        expected = ref_head_losses[rank // 2] + (ref_encoder_losses[rank // 2] if active else 0)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(gnorm, ref_gnorm)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
@pytest.mark.parametrize('mode', ['zero1', 'zero3', 'hybrid', 'deduped', 'npp_zero1'])
def test_inactive_rank_training_matches_eager(tmp_path, mode):
    launch_torchrun(4, _inactive_rank_training_worker, tmp_path, mode)


def _gather_with_non_persistent_buffer_worker(tmp_path):
    init_distributed()
    torch.manual_seed(0)
    source = ScaledEncoder()
    expected = {f'0.{k}': v.detach().cpu() for k, v in source.state_dict().items()}
    model = nn.Sequential(parallelize(source, {'x': torch.ones(2, 3)}, _first_rank_only,
                                      ComputeConfig(2, 2), gen_savedir=tmp_path))
    # Every rank, including the inactive one, skips the non-persistent buffer alike.
    gathered = gather_full_model_state_dict(model)
    assert sorted(gathered) == sorted(expected)
    for key, value in expected.items():
        torch.testing.assert_close(gathered[key].cpu(), value)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires 2 GPUs')
def test_gather_full_model_state_dict_with_non_persistent_buffer(tmp_path):
    launch_torchrun(2, _gather_with_non_persistent_buffer_worker, tmp_path)
