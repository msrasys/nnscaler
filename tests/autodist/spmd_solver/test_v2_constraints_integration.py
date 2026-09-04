#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Integration test: V2 constraints applied through the SPMDSolver.

Mirrors the setup from test_partition_constraint.py but uses
PartitionConstraintV2 / ConstraintSet instead of legacy YAML.
"""

import pytest
import tempfile
import torch
import os
from pathlib import Path

from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.spmd_solver import SPMDSolver
from nnscaler.autodist.constraints import (
    PartitionConstraintV2,
    ConstraintSet,
)


class Attention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        score = torch.matmul(q, k.transpose(-2, -1))
        score = torch.nn.functional.softmax(score, dim=-1)
        out = torch.matmul(score, v)
        out = self.out_proj(out)
        return out


class FFN(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


class Decoder(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = Attention(hidden_dim)
        self.ffn = FFN(hidden_dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        x = x.sum()
        return x


def _build_solver(constraint_set, profile_dir):
    """Helper: trace model, build ModelGraph + SPMDSolver with V2 constraints."""
    bsz, seq_len, hidden_dim = 2, 128, 768
    dummy_input = {'x': torch.randn(bsz, seq_len, hidden_dim)}
    model = Decoder(hidden_dim)
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(
            fx_graph, dummy_input,
            attr_savedir=tempdir, constant_folding=True,
        )
        cfg = AutoDistConfig(
            mesh_col=2,
            profile_dir=str(profile_dir),
            constraints=constraint_set,
        )
        model_graph = ModelGraph(ir_graph, cfg)
        solver = SPMDSolver(
            graph=model_graph,
            mesh_desc=cfg.mesh_desc,
            autodist_config=cfg,
            stage_num=1,
            micro_batch_num=cfg.update_freq,
        )
        partition_counts = [
            solver.get_op_partition_count(i)
            for i in range(model_graph.op_num)
        ]
        return solver, model_graph, partition_counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_v2_force_replicate():
    """force_replicate should limit matching ops to exactly 1 partition (replicated)."""
    cs = ConstraintSet(partition_constraints=[
        PartitionConstraintV2(
            op_name='torch.nn.functional.linear',
            module_class='Attention',
            force_replicate=True,
        ),
    ])
    profile_dir = Path(os.path.dirname(os.path.realpath(__file__))) / 'test_v2_profile'
    solver, model_graph, partition_counts = _build_solver(cs, profile_dir)

    # Attention linear ops: q, k, v, out_proj -> all should have 1 partition (replicate only)
    # Operators: q_proj(0), k_proj(1), v_proj(2), transpose(3), matmul(4),
    #            softmax(5), matmul(6), out_proj(7), fc1(8), relu(9), fc2(10), sum(11)
    for i in [0, 1, 2, 7]:  # attention linear ops
        assert partition_counts[i] == 1, (
            f'Op {i} ({model_graph.operator_list[i].op_name}) should have 1 partition, '
            f'got {partition_counts[i]}'
        )
    # FFN linear ops should be unconstrained (more than 1)
    for i in [8, 10]:  # fc1, fc2
        assert partition_counts[i] > 1, (
            f'FFN op {i} should have > 1 partitions, got {partition_counts[i]}'
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_v2_allowed_dims():
    """allowed_dims should restrict partition choices for matching ops."""
    cs = ConstraintSet(partition_constraints=[
        PartitionConstraintV2(
            op_name='torch.nn.functional.linear',
            module_class='Attention',
            allowed_dims=[(0, 0)],  # only batch dim partition allowed
            forbid_replicate=True,
        ),
    ])
    profile_dir = Path(os.path.dirname(os.path.realpath(__file__))) / 'test_v2_profile'
    solver, model_graph, partition_counts = _build_solver(cs, profile_dir)

    # Attention linear ops should have exactly 1 partition (only (0,0) + no replicate)
    for i in [0, 1, 2, 7]:
        assert partition_counts[i] == 1, (
            f'Op {i} should have exactly 1 partition, got {partition_counts[i]}'
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_v2_forbid_replicate():
    """forbid_replicate should remove the replicate option."""
    cs = ConstraintSet(partition_constraints=[
        PartitionConstraintV2(
            op_name='torch.nn.functional.linear',
            module_class='FFN',
            forbid_replicate=True,
        ),
    ])
    profile_dir = Path(os.path.dirname(os.path.realpath(__file__))) / 'test_v2_profile'
    solver, model_graph, partition_counts = _build_solver(cs, profile_dir)

    # Without constraint, fc1 and fc2 have {replicate, dim0, dim1} = some count
    # With forbid_replicate, replicate is removed, count decreases by 1
    # Just verify the count is >= 1 (must partition) and all partitions are non-replicate
    for i in [8, 10]:
        count = partition_counts[i]
        assert count >= 1, f'Op {i} should have at least 1 partition'
        for j in range(count):
            p = solver._op_partitions[i][j]
            assert p.partition_dims[0] != -1, (
                f'Op {i} partition {j} should not be replicate'
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_v2_reset_matched():
    """reset_matched should clear matched state for reuse."""
    cs = ConstraintSet(partition_constraints=[
        PartitionConstraintV2(op_name='torch.nn.functional.linear', force_replicate=True),
    ])
    profile_dir = Path(os.path.dirname(os.path.realpath(__file__))) / 'test_v2_profile'

    # First run
    _build_solver(cs, profile_dir)
    assert len(cs.get_unmatched_constraints()) == 0

    # Reset and run again — should have clean state
    cs.reset_matched()
    assert len(cs.get_unmatched_constraints()) == 1

    _build_solver(cs, profile_dir)
    assert len(cs.get_unmatched_constraints()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_v2_matches_legacy_yaml_behavior():
    """V2 constraints equivalent to the legacy YAML should produce the same partition counts."""
    # Replicate the legacy test_pc.yaml constraints using V2 API
    cs = ConstraintSet(partition_constraints=[
        PartitionConstraintV2(
            op_name='torch.nn.functional.linear',
            module_class='Attention',
            allowed_dims=[(0, 0)],
            forbid_replicate=True,
        ),
        PartitionConstraintV2(
            op_name='torch.matmul',
            module_class='Attention',
            allowed_dims=[(0, 0)],
            forbid_replicate=True,
        ),
        PartitionConstraintV2(
            op_name='torch.nn.functional.linear',
            module_class='FFN',
            allowed_dims=[(1, 0), (1, 1)],
            forbid_replicate=True,
        ),
    ])
    profile_dir = Path(os.path.dirname(os.path.realpath(__file__))) / 'test_v2_profile'
    solver, model_graph, partition_counts = _build_solver(cs, profile_dir)

    # Expected from legacy test: [1, 1, 1, 4, 1, 3, 1, 1, 2, 4, 2, 4]
    assert partition_counts == [1, 1, 1, 4, 1, 3, 1, 1, 2, 4, 2, 4], (
        f'V2 partition counts {partition_counts} should match legacy YAML counts'
    )
