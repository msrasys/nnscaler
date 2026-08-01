#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import json

import pytest
import torch

import nnscaler
from nnscaler.autodist.apis import collect_tensor_split_info, parallelize_graph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.cost_database import CostDatabase
from nnscaler.autodist.descs import (
    MeshDesc,
    NodePartitionDesc,
    PipelineParallelDesc,
    PipelineSearchOutput,
    TensorParallelDesc,
)
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.op_partition import OpPartition
from nnscaler.autodist.spmd_solver import SPMDSolver
from nnscaler.autodist.util import partition_node
from nnscaler.graph.function import DimopSplit, TransformRule
from nnscaler.graph.gener.gen import IRAdapterGener
from nnscaler.graph.parser.converter import convert_model
from nnscaler.ir.adapter import IRWeightReducer
from nnscaler.ir.operator import IRFwOperation
from nnscaler.parallel import ComputeConfig


def fixed_partition_op(x: torch.Tensor, weight: torch.Tensor):
    return x + weight.sum(dim=0)


_token_only_rule = TransformRule(
    (DimopSplit.D(0), DimopSplit.R()),
    (DimopSplit.D(0),),
)
_token_and_expert_rule = TransformRule(
    (DimopSplit.D(0), DimopSplit.D(0)),
    (DimopSplit.D(0),),
)
nnscaler.register_op(
    'l h^, E h^ -> l h^',
    transform_rules=(_token_only_rule, _token_and_expert_rule),
)(fixed_partition_op)


@nnscaler.register_op('l h^ -> l h^')
def fixed_partition_identity(x: torch.Tensor):
    return x.clone()


class FixedPartitionBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 4))

    def forward(self, x):
        return fixed_partition_identity(fixed_partition_op(x, self.weight))


class FixedPartitionModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.block = FixedPartitionBlock()

    def forward(self, x):
        return self.block(x).sum()


def _build_graph(tmp_path):
    graph = convert_model(
        FixedPartitionModel(),
        {'x': torch.randn(16, 4)},
        attr_savedir=tmp_path,
        constant_folding=False,
    )
    fixed_node = next(
        node for node in graph.select(ntype=IRFwOperation)
        if node.fn == fixed_partition_op
    )
    identity_node = next(
        node for node in graph.select(ntype=IRFwOperation)
        if node.fn == fixed_partition_identity
    )
    return graph, fixed_node, identity_node


def _fixed_config(name):
    return [
        {
            'name': name,
            'parent_module': 'FixedPartitionBlock',
            'desc': [
                {'input': 0, 'dim': 0, 'num': 2},
                {'input': 1, 'dim': 0, 'num': 2},
            ],
        },
        {
            'name': name,
            'parent_module': '',
            'desc': [{'input': 0, 'dim': 0, 'num': 4}],
        },
    ]


def test_fixed_partition_config_validation(tmp_path):
    config = AutoDistConfig(
        mesh_col=4,
        profile_dir=tmp_path,
        fixed_partition_descs=_fixed_config('test.op'),
    )
    assert config.fixed_partition_descs[0].desc == (
        ((0, 0), 2),
        ((1, 0), 2),
    )

    with pytest.raises(ValueError, match='fixed partition degree'):
        AutoDistConfig(
            mesh_col=4,
            profile_dir=tmp_path,
            fixed_partition_descs=[{
                'name': 'test.op',
                'parent_module': '',
                'desc': [{'input': 0, 'dim': 0, 'num': 2}],
            }],
        )

    with pytest.raises(ValueError, match='single SPMD stage'):
        AutoDistConfig(
            mesh_col=4,
            profile_dir=tmp_path,
            pipeline_pivots='block',
            pipeline_nstages=2,
            fixed_partition_descs=_fixed_config('test.op'),
        )


def test_pas_config_forwards_fixed_partition_descs(monkeypatch, tmp_path):
    import nnscaler.policies as policies

    fixed_descs = _fixed_config('test.op')
    captured = {}

    def fake_parallelize_graph(graph, config):
        captured['config'] = config
        return graph

    monkeypatch.setattr(policies, 'parallelize_graph', fake_parallelize_graph)
    compute_config = ComputeConfig(
        plan_ngpus=4,
        runtime_ngpus=4,
        pas_config={
            'fixed_partition_descs': fixed_descs,
            'mem_constraint': 1,
            'profile_dir': str(tmp_path),
            'parallel_profile': False,
        },
    )
    marker = object()
    assert policies.pas_autodist(marker, compute_config) is marker
    assert captured['config'].fixed_partition_descs[0].desc == (
        ((0, 0), 2),
        ((1, 0), 2),
    )


def test_solver_forces_closest_multistep_candidate(monkeypatch, tmp_path):
    graph, fixed_node, _ = _build_graph(tmp_path)
    config = AutoDistConfig(
        mesh_col=4,
        profile_dir=tmp_path,
        parallel_profile=False,
        fixed_partition_descs=_fixed_config(fixed_node.signature),
    )
    model_graph = ModelGraph(graph, config)
    profiled_exact_nodes = []

    def fake_profile_comp(self, partition_degree, parallel_profile, re_profile,
                          exact_nodes=None):
        profiled_exact_nodes.extend(exact_nodes or [])

    monkeypatch.setattr(CostDatabase, 'profile_comp', fake_profile_comp)
    monkeypatch.setattr(SPMDSolver, 'build_following_relationships', lambda self: None)
    monkeypatch.setattr(SPMDSolver, 'calc_partition_info', lambda self: None)

    solver = SPMDSolver(model_graph, config, config.mesh_desc)
    fixed_idx = model_graph.get_op_idx(
        next(op for op in model_graph.operator_list
             if op.ir_cell.cid == fixed_node.cid)
    )
    partitions = solver._op_partitions[fixed_idx]
    assert len(partitions) == 1
    partition = partitions[0]
    assert partition.partition_positions == ((0, 0), (1, 0))
    assert partition.partition_nums == (2, 2)
    assert partition.ir_cell.input(0).shape == (4, 4)
    assert partition.ir_cell.input(1).shape == (4, 4)
    assert any(
        node.signature == fixed_node.signature and
        node.input(0).shape == (4, 4) and
        node.input(1).shape == (4, 4)
        for node in profiled_exact_nodes
    )

    desc = solver.partition_path2desc([(fixed_idx, 0)])[fixed_node.cid]
    assert desc.desc == (((0, 0), 2), ((1, 0), 2))


def test_fixed_parent_matching_uses_class_boundaries(tmp_path):
    graph, fixed_node, _ = _build_graph(tmp_path)
    config = AutoDistConfig(
        mesh_col=4,
        profile_dir=tmp_path,
        fixed_partition_descs=[
            {
                'name': fixed_node.signature,
                # This is only a substring of FixedPartitionBlock and must not
                # match it.
                'parent_module': 'PartitionBlock',
                'desc': [
                    {'input': 0, 'dim': 0, 'num': 2},
                    {'input': 1, 'dim': 0, 'num': 2},
                ],
            },
            {
                'name': fixed_node.signature,
                'parent_module': '',
                'desc': [{'input': 0, 'dim': 0, 'num': 4}],
            },
        ],
    )
    operator = next(
        op for op in ModelGraph(graph, config).operator_list
        if op.ir_cell.cid == fixed_node.cid
    )
    solver = SPMDSolver.__new__(SPMDSolver)
    solver.fixed_partition_descs = {
        fixed_node.signature: {
            desc.parent_module: desc
            for desc in config.fixed_partition_descs
        }
    }
    solver.non_used_fixed_partition_descs = set(config.fixed_partition_descs)
    selected = solver._select_fixed_partition_desc(operator)
    assert selected.parent_module == ''
    assert selected not in solver.non_used_fixed_partition_descs


@pytest.mark.parametrize(
    ('name_kind', 'parent_module'),
    [
        ('missing', 'FixedPartitionBlock'),
        ('actual', 'MissingParent'),
    ],
)
def test_unmatched_fixed_partition_desc_fails_closed(
    tmp_path, name_kind, parent_module
):
    graph, fixed_node, _ = _build_graph(tmp_path)
    name = fixed_node.signature if name_kind == 'actual' else 'missing.op'
    config = AutoDistConfig(
        mesh_col=4,
        profile_dir=tmp_path,
        parallel_profile=False,
        fixed_partition_descs=[{
            'name': name,
            'parent_module': parent_module,
            'desc': [{'input': 0, 'dim': 0, 'num': 4}],
        }],
    )

    with pytest.raises(
        ValueError,
        match='fixed partition descriptions did not match any operator',
    ):
        SPMDSolver(ModelGraph(graph, config), config, config.mesh_desc)


def test_multistep_plan_roundtrip_replay_and_layout(tmp_path):
    graph, fixed_node, _ = _build_graph(tmp_path)
    partition_descs = {
        node.cid: NodePartitionDesc([((-1, -1), 4)])
        for node in graph.select(ntype=IRFwOperation)
    }
    partition_descs[fixed_node.cid] = NodePartitionDesc([
        ((0, 0), 2),
        ((1, 0), 2),
    ])
    tp_desc = TensorParallelDesc(
        partition_descs=partition_descs,
        recompute_groups=[],
        mesh_desc=MeshDesc(1, 4),
        analysis={},
    )
    pp_desc = PipelineParallelDesc(
        spmd_descs=[tp_desc],
        recompute_groups=[],
        mesh_desc=MeshDesc(1, 4),
    )
    restored = PipelineParallelDesc.from_json(
        json.loads(json.dumps(pp_desc.to_json()))
    )

    split_info = collect_tensor_split_info(graph, restored)
    weight = fixed_node.input(1).parent
    [(kind, (indmap, _))] = list(split_info[weight][0])
    assert kind == 'PARTITIONED'
    assert indmap[0] == (0, 4)

    restored_desc = restored.spmd_descs[0].partition_descs[fixed_node.cid]
    partition_node(fixed_node, graph, list(range(4)), restored_desc)
    local_nodes = sorted(
        (node for node in graph.select(ntype=IRFwOperation)
         if node.fn == fixed_partition_op),
        key=lambda node: node.device[0],
    )
    assert [node.input(0).shape for node in local_nodes] == [(4, 4)] * 4
    assert [node.input(1).shape for node in local_nodes] == [(4, 4)] * 4
    assert [node.input(1).indmap[0] for node in local_nodes] == [
        (0, 4),
        (4, 8),
        (0, 4),
        (4, 8),
    ]


def test_load_plan_validates_fixed_partition_desc(tmp_path):
    graph, fixed_node, _ = _build_graph(tmp_path)
    partition_descs = {
        node.cid: NodePartitionDesc([((-1, -1), 4)])
        for node in graph.select(ntype=IRFwOperation)
    }
    # This is a valid four-way plan, but it violates the configured ordered
    # token-then-expert partition for the target operator.
    partition_descs[fixed_node.cid] = NodePartitionDesc([((0, 0), 4)])
    tp_desc = TensorParallelDesc(
        partition_descs=partition_descs,
        recompute_groups=[],
        mesh_desc=MeshDesc(1, 4),
        analysis={},
    )
    search_out = PipelineSearchOutput(
        desc=PipelineParallelDesc(
            spmd_descs=[tp_desc],
            recompute_groups=[],
            mesh_desc=MeshDesc(1, 4),
        ),
        e2e_time=0.0,
        stage_mems=[0.0],
        stage_all_times=[0.0],
        stage_comp_times=[0.0],
    )
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(search_out.to_json()))
    config = AutoDistConfig(
        mesh_col=4,
        profile_dir=tmp_path,
        load_plan_path=plan_path,
        fixed_partition_descs=[{
            'name': fixed_node.signature,
            'parent_module': 'FixedPartitionBlock',
            'desc': [
                {'input': 0, 'dim': 0, 'num': 2},
                {'input': 1, 'dim': 0, 'num': 2},
            ],
        }],
    )

    with pytest.raises(ValueError, match='does not match fixed description'):
        parallelize_graph(graph, config)


def test_multistep_expert_replica_reducer_groups(tmp_path):
    graph, fixed_node, _ = _build_graph(tmp_path)
    graph.backward(graph.output(0))
    partition_node(
        fixed_node,
        graph,
        list(range(4)),
        NodePartitionDesc([((0, 0), 2), ((1, 0), 2)]),
    )
    # Other operators are irrelevant to this weight reducer; place them so the
    # generator can inspect the complete graph.
    for node in graph.select(ntype=IRFwOperation):
        if not node.device:
            graph.assign(node, 0)

    IRAdapterGener.gen_weight(graph)
    weight_reducers = [
        reducer for reducer in graph.select(ntype=IRWeightReducer)
        if any(weight.parent.name == 'block.weight'
               for weight in reducer.inputs())
    ]
    assert {reducer.device for reducer in weight_reducers} == {
        (0, 2),
        (1, 3),
    }


def test_multistep_cost_uses_final_layout(tmp_path):
    graph, fixed_node, identity_node = _build_graph(tmp_path)
    config = AutoDistConfig(mesh_col=4, profile_dir=tmp_path)
    model_graph = ModelGraph(graph, config)
    fixed_operator = next(
        op for op in model_graph.operator_list if op.ir_cell.cid == fixed_node.cid
    )
    identity_operator = next(
        op for op in model_graph.operator_list if op.ir_cell.cid == identity_node.cid
    )
    token_dim = fixed_operator.pos2dim_id((0, 0))
    expert_dim = fixed_operator.pos2dim_id((1, 0))
    fixed_partition = OpPartition(
        (token_dim, expert_dim),
        (2, 2),
        fixed_operator,
        partition_positions=((0, 0), (1, 0)),
    )
    identity_partition = OpPartition(
        (identity_operator.pos2dim_id((0, 0)),),
        (4,),
        identity_operator,
        partition_positions=((0, 0),),
    )

    cost_db = CostDatabase.__new__(CostDatabase)
    assert cost_db.estimate_comm_cost(
        fixed_partition, identity_partition, is_forward=True
    ) == 0

    calls = []
    cost_db.query_single_mem = lambda obj, memory_type, round=False: \
        800 if memory_type == 'full_weight' else 400
    cost_db.primitive_to_cost = lambda dev_num, byte_size, primitive: \
        calls.append((dev_num, byte_size, primitive)) or 1.0
    assert cost_db.calc_weight_update_time(fixed_partition) == 1.0
    assert calls == [(2, 400, 'all reduce')]

    mixed_partition = OpPartition(
        (token_dim, -1),
        (2, 2),
        fixed_operator,
        partition_positions=((0, 0), (-1, -1)),
    )
    with pytest.raises(NotImplementedError, match='partition/replica'):
        cost_db.estimate_comm_cost(
            mixed_partition, identity_partition, is_forward=True
        )
