# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""CPU checks for checkpoint composition and its actual generated boundaries."""

import ast
import copy
from collections import OrderedDict
from dataclasses import replace

import pytest
import torch

from nnscaler import register_op
import nnscaler.graph.function as F
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.recompute import merge_recompute_regions as _merge
from nnscaler.ir.cten import IRObject
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.parallel import ComputeConfig, _load_parallel_module_class, parallelize
from nnscaler.policies import OpPartition, OpPlan, fn, get_pas_ops
from nnscaler.runtime.cpu_offloading import CPUOffloadConfig
from tests.utils import mock_cube_env, replace_all_device_with


def _output(op):
    for index, shape in op.infer_shape().items():
        op.set_output(index, _tensor(shape))
    return op


def _tensor(shape=(8, 8)):
    # Standalone IR fixtures need the gradient edges already present in PAS.
    full = IRFullTensor(shape, requires_grad=True)
    tensor = full.tosub()
    tensor.grad = full.grad.tosub()
    return tensor


def _chain(count=2):
    value = _tensor()
    plans = []
    for index in range(count):
        op = _output(F.Tanh(value, signature='torch.tanh'))
        plans.append(OpPlan(op, recompute_id=index, stage_id=0))
        value = op.output(0)
    return plans


def _policy(base, merge, reports, degree=1):
    def policy(graph, config):
        def transform(plans, cfg):
            merged, report = _merge(plans, partition_degree=degree)
            reports.append(report)
            return merged
        return fn(graph, config, base, plan_transform=transform if merge else None)
    return policy


def _compile_pair(tmp_path, model, base, config, *, inputs=None, degree=1, **kwargs):
    config = replace(config, recompute_single_ops=True)
    inputs = inputs if inputs is not None else {'x': torch.randn(4, 8, requires_grad=True)}
    codes, reports = {}, []
    for merge in (False, True):
        target = tmp_path / str(merge)
        parallelize(model, inputs, _policy(base, merge, reports, degree), config,
                    gen_savedir=target, load_module=False, **kwargs)
        codes[merge] = [ast.parse(next(target.rglob(f'gencode{rank}.py')).read_text())
                        for rank in range(config.runtime_ngpus)]
    return codes, reports[0]


def _checkpoint_bodies(tree):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == 'recompute']


def _adapters(tree):
    return [(ast.unparse(node.func), tuple((kw.arg, ast.unparse(kw.value)) for kw in node.keywords))
            for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func).startswith('nnscaler.runtime.adapter.')]


def _assert_communication_unchanged(before, after, groups_after):
    old, new = _checkpoint_bodies(before), _checkpoint_bodies(after)
    assert len(old) == 2 and len(new) == groups_after
    assert _adapters(before) == _adapters(after)
    assert [call for body in old for call in _adapters(body)] == [
        call for body in new for call in _adapters(body)]


def test_merging_copies_annotations_and_preserves_execution_structure():
    plans = _chain()
    plans[0].offload_id, plans[1].offload_id = 10, 20
    merged, report = _merge(plans)
    assert [p.recompute_id for p in plans] == [0, 1]
    assert [p.recompute_id for p in merged] == [0, 0]
    assert [p.offload_id for p in merged] == [10, 20]
    assert all(a is not b and a.op is b.op for a, b in zip(merged, plans))
    assert [replace(p, recompute_id=-1) for p in merged] == [replace(p, recompute_id=-1) for p in plans]
    assert report['groups_before'] == 2
    assert report['groups_after'] == 1
    assert report['merged_boundaries'] == [(0, 1)]
    assert all(p.op.recompute is None and p.op.offload is None for p in merged)


@pytest.mark.parametrize('barrier', ['unselected', 'stage', 'auto', 'pre_hook', 'post_hook',
                                    'inplace', 'out', 'offload', 'no_dependency'])
def test_barriers_never_extend_checkpoint_replay(barrier):
    plans = _chain(3)
    if barrier == 'unselected':
        plans[1].recompute_id = -1
    elif barrier == 'stage':
        plans[1].stage_id = 1
    elif barrier == 'auto':
        plans[1].partition = 'auto'
    elif barrier in ('pre_hook', 'post_hook'):
        setattr(plans[1], barrier, lambda *args: None)
    elif barrier in ('inplace', 'out'):
        plans[1].op.kwargs[barrier] = True
    elif barrier == 'offload':
        plans[1].offload_id = 9
    elif barrier == 'no_dependency':
        plans[1].op.set_input(0, _tensor())
        plans[2].op.set_input(0, _tensor())
    merged, report = _merge(plans)
    assert [p.recompute_id for p in merged] == [p.recompute_id for p in plans]
    assert report['merged_boundaries'] == []


@pytest.mark.parametrize('name', ['setitem', 'imul'])
def test_python_mutation_is_a_hard_barrier(name):
    plans = _chain()
    mutation = IRFwOperation(name, f'_operator.{name}', [plans[0].op.output(0), 2], 0)
    plans.insert(1, OpPlan(mutation, 0, 0))
    merged, report = _merge(plans)
    assert [p.recompute_id for p in merged] == [0, 0, 1]
    assert report['barriers']['mutation'] == 1


def test_integer_shape_arithmetic_inside_selected_region_remains_eligible():
    plans = _chain(3)
    plans[1].recompute_id = 0
    shape = IRObject('dim', value=4)
    arithmetic = IRFwOperation('imul', '_operator.imul', [shape, 2], 1)
    arithmetic.set_output(0, IRObject('new_dim', value=8))
    plans.insert(1, OpPlan(arithmetic, 0, 0))
    merged, report = _merge(plans)
    assert [p.recompute_id for p in merged] == [0, 0, 0, 0]
    assert report['groups_after'] == 1


def test_explicit_safe_hook_declaration_and_node_offload_selection():
    plans = _chain()
    hook = lambda *args: None
    plans[0].post_hook = hook
    for index, plan in enumerate(plans):
        plan.op.offload = index
    merged, report = _merge(plans, hooks_allow_recompute=lambda pre, post: pre is None and post is hook)
    assert report['groups_after'] == 1
    assert [p.op.offload for p in merged] == [0, 1]


def test_mixed_offload_inside_a_region_is_a_barrier():
    plans = _chain(3)
    plans[1].recompute_id = 0
    plans[1].offload_id = 0
    merged, report = _merge(plans)
    assert [p.recompute_id for p in merged] == [0, 0, 2]
    assert report['barriers']['mixed_offload'] == 1


@pytest.mark.parametrize('mutation', ['noncontiguous', 'cross_stage', 'already_annotated'])
def test_invalid_original_annotation_is_not_hidden(mutation):
    plans = _chain(3)
    if mutation == 'noncontiguous':
        plans[2].recompute_id = 0
    elif mutation == 'cross_stage':
        plans[1].recompute_id = 0
        plans[1].stage_id = 1
    else:
        plans[0].op.recompute = 0
    with pytest.raises(ValueError):
        _merge(plans)


def test_unlisted_custom_operator_inherits_selected_recompute_contract():
    plans = _chain(1)
    custom = IRFwOperation('custom_kernel', 'unlisted_package.custom_kernel',
                           [plans[0].op.output(0)], 1)
    custom.set_output(0, _tensor())
    plans.append(OpPlan(custom, 1, 0))
    merged, report = _merge(plans)
    assert [p.recompute_id for p in merged] == [0, 0]
    assert report['groups_after'] == 1


def test_known_effect_metadata_blocks_composition():
    plans = _chain(3)
    plans[1].op.recompute_barriers = ('custom_state',)
    merged, report = _merge(plans)
    assert [p.recompute_id for p in merged] == [0, 1, 2]
    assert report['merged_boundaries'] == []


@pytest.mark.parametrize('position', ['head', 'both'])
def test_trimmed_state_operations_are_never_absorbed_into_replay(position):
    plans = _chain()
    states = []
    for where in (('head', 'tail') if position == 'both' else (position,)):
        state = IRFwOperation('touch_state', 'custom.touch_state', [IRObject('counter', value=0)], 0)
        state.recompute_barriers = ('state',)
        states.append(state)
        plans.insert(0 if where == 'head' else len(plans) - 1, OpPlan(state, 0, 0))
    merged, report = _merge(plans)
    # Preserve graph.recompute's original head/tail exclusions.
    assert all(p.recompute_id == -1 for p in merged if p.op in states)
    tensor_plans = [p for p in merged if p.op not in states]
    assert [p.recompute_id for p in tensor_plans] == ([0, 0] if position == 'head' else [0, 1])
    assert all(p.recompute_id == 0 for p in plans if p.op in states)


@pytest.mark.parametrize('consumer_dim,expected_groups', [(0, 2), (1, 1)])
def test_actual_tensor_layout_matters_more_than_partition_argument_equality(consumer_dim, expected_groups):
    x = _tensor()
    transpose = _output(F.Transpose(x, 0, 1, signature='torch.transpose'))
    tanh = _output(F.Tanh(transpose.output(0), signature='torch.tanh'))
    plans = [OpPlan(transpose, 0, 0, partition=OpPartition(0, 0)),
             OpPlan(tanh, 1, 0, partition=OpPartition(0, consumer_dim))]
    merged, report = _merge(plans, partition_degree=2)
    assert len({p.recompute_id for p in merged}) == expected_groups
    assert report['groups_after'] == expected_groups


def test_repeated_input_slots_cannot_hide_conflicting_gradient_layouts():
    def mixed_grad(x, y, weight, signature='test.mixed_grad'):
        return IRDimops(mixed_grad, 'mixed_grad', signature,
                        ['a b, a b : /, b c -> a c'], [x, y, weight])

    plans = _chain(1)
    h = plans[0].op.output(0)
    weight = _tensor()
    op = _output(mixed_grad(h, h, weight))
    shard = op.algorithm('dim').instantiate(idx=2, dim=1, num=2)[0]
    assert shard.input(0) == shard.input(1)
    assert not shard.ignore_grad_reduce(input_idx=0)
    assert shard.ignore_grad_reduce(input_idx=1)
    plans.append(OpPlan(op, 1, 0, partition=OpPartition(2, 1)))
    merged, report = _merge(plans, partition_degree=2)
    assert [p.recompute_id for p in merged] == [0, 1]
    assert report['barriers']['layout'] == 1


def _side_input_plans(producer_partition=OpPartition(0, 1)):
    side = _output(F.Tanh(_tensor(), signature='torch.tanh'))
    first = _output(F.Tanh(_tensor(), signature='torch.tanh'))
    add = _output(F.Add(first.output(0), side.output(0), signature='torch.add'))
    last = _output(F.Tanh(add.output(0), signature='torch.tanh'))
    return [OpPlan(side, -1, 0, partition=producer_partition),
            OpPlan(first, 0, 0, partition=OpPartition(0, 0)),
            OpPlan(add, 1, 0, partition=OpPartition(0, 0)),
            OpPlan(last, 1, 0, partition=OpPartition(0, 0))]


def test_unresolved_side_input_layout_prevents_composition():
    plans = _side_input_plans('auto')
    merged, report = _merge(plans, partition_degree=2)
    assert report['groups_after'] == 2
    assert [plan.recompute_id for plan in plans] == [-1, 0, 1, 1]
    assert merged[0].recompute_id == -1


def test_all_side_input_producers_are_checked_without_last_write_wins():
    plans = _side_input_plans()
    same_layout = _output(F.Tanh(_tensor(), signature='torch.tanh'))
    same_layout.set_output(0, plans[0].op.output(0))
    plans.insert(1, OpPlan(same_layout, -1, 0, partition=OpPartition(0, 0)))
    _, report = _merge(plans, partition_degree=2)
    assert report['groups_after'] == 2


def test_side_input_value_partition_is_a_barrier():
    plans = _side_input_plans()
    partial = _output(F.Matmul(_tensor(), _tensor(), signature='torch.matmul'))
    plans[2].op.set_input(1, partial.output(0))
    plans[0] = OpPlan(partial, -1, 0, partition=OpPartition(0, 1))
    assert partial.algorithm('dim').instantiate(idx=0, dim=1, num=2)[0].output(0).valmap != partial.output(0).valmap
    _, report = _merge(plans, partition_degree=2)
    assert report['groups_after'] == 2


def test_side_input_gradient_reduction_is_a_barrier_even_with_equal_forward_layout():
    def mixed_grad(y, side, weight, signature='test.side_grad'):
        return IRDimops(mixed_grad, 'side_grad', signature,
                        ['a b : /, a b, b c -> a c'], [y, side, weight])

    side, first = _chain(2)
    side.recompute_id = -1
    # Give the first selected region an independent input.
    first.op.set_input(0, _tensor())
    first.recompute_id = 0
    consumer = _output(mixed_grad(first.op.output(0), side.op.output(0), _tensor()))
    shard = consumer.algorithm('dim').instantiate(idx=2, dim=1, num=2)[0]
    assert shard.ignore_grad_reduce(input_idx=0)
    assert not shard.ignore_grad_reduce(input_idx=1)
    assert shard.input(1).indmap == side.op.output(0).indmap
    assert shard.input(1).valmap == side.op.output(0).valmap
    _, report = _merge([side, first, OpPlan(consumer, 1, 0, partition=OpPartition(2, 1))],
                       partition_degree=2)
    assert report['groups_after'] == 2


@register_op('s d^, d^ -> s d^')
def _rms_norm(x, weight):
    return torch.nn.functional.rms_norm(x, (8,), weight, eps=1e-6)


class _Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(8))

    def forward(self, x):
        return _rms_norm(x, self.weight)


class _Compute(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return torch.tanh(self.proj(x))


class _NormCompute(torch.nn.Module):
    def __init__(self, expose_intermediate=False):
        super().__init__()
        self.norm = _Norm()
        self.compute = _Compute()
        self.expose_intermediate = expose_intermediate

    def forward(self, x):
        h = self.norm(x)
        y = self.compute(h)
        return (y, h) if self.expose_intermediate else y


@replace_all_device_with('cpu')
def test_compute_config_merges_once_after_resolved_annotation_transform(tmp_path, monkeypatch):
    from nnscaler import policies

    events = []
    original_merge = policies.merge_recompute_regions

    def record_merge(plans, **kwargs):
        events.append('merge')
        result, report = original_merge(plans, **kwargs)
        assert report['groups_before'] == 3 and report['groups_after'] == 1
        return result, report

    monkeypatch.setattr(policies, 'merge_recompute_regions', record_merge)

    def base(graph, config):
        events.append('base')
        for index, op in enumerate(get_pas_ops(graph)):
            yield OpPlan(op, stage_id=0, partition=OpPartition(0, 0) if index == 0 else 'auto')

    def transform(plans, config):
        events.append('transform')
        assert all(plan.partition != 'auto' for plan in plans)
        for plan in plans:
            plan.recompute_id = int(plan.op.get_module_fqn(_Compute).removeprefix('stage'))
        return plans

    def policy(graph, config):
        # parallelize wraps this policy in another fn; its IRGraph result must
        # bypass activation annotation rather than invoking the pass twice.
        return fn(graph, config, base, plan_transform=transform)

    model = torch.nn.Sequential(OrderedDict((f'stage{i}', _Compute()) for i in range(3)))
    parallelize(model, {'input': torch.randn(4, 8, requires_grad=True)}, policy,
                ComputeConfig(2, 2, recompute_merge_adjacent=True),
                gen_savedir=tmp_path, load_module=False)
    assert events == ['base', 'transform', 'merge']
    for rank in range(2):
        code = ast.parse(next(tmp_path.rglob(f'gencode{rank}.py')).read_text())
        assert len(_checkpoint_bodies(code)) == 1


@replace_all_device_with('cpu')
def test_annotation_transform_cannot_erase_known_effects(tmp_path):
    from tests.utils import raises_with_cause

    def base(graph, config):
        for node in get_pas_ops(graph):
            node.recompute_barriers = ('state',)
            yield OpPlan(node, 0, 0)

    def transform(plans, config):
        plans[0].op.recompute_barriers = ()
        return plans

    def policy(graph, config):
        return fn(graph, config, base, plan_transform=transform)

    with raises_with_cause(ValueError, match='activation annotations only'):
        parallelize(_NormCompute(), {'x': torch.randn(4, 8, requires_grad=True)},
                    policy, ComputeConfig(1, 1), gen_savedir=tmp_path,
                    load_module=False)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('expose_intermediate', [False, True])
@pytest.mark.parametrize('offload', [False, True])
def test_codegen_removes_only_internal_checkpoint_boundary(tmp_path, expose_intermediate, offload):
    def base(graph, config):
        for op in get_pas_ops(graph):
            rid = 0 if _Norm in op.module_class_chain else 1
            yield OpPlan(op, rid, 0, offload_id=rid if offload else -1)

    config = ComputeConfig(1, 1, cpu_offload=CPUOffloadConfig() if offload else None)
    codes, report = _compile_pair(tmp_path, _NormCompute(expose_intermediate), base, config)
    split, merged = _checkpoint_bodies(codes[False][0]), _checkpoint_bodies(codes[True][0])
    assert len(split) == 2 and len(merged) == 1
    assert report['groups_before'] == 2 and report['groups_after'] == 1
    # The second split checkpoint takes h; the merged closure takes only x.
    assert split[1].args.args[0].arg != split[0].args.args[0].arg
    assert len(merged[0].args.args) == 1
    returned = next(n.value for n in merged[0].body if isinstance(n, ast.Return))
    assert isinstance(returned, ast.Tuple) == expose_intermediate
    if expose_intermediate:
        assert len(returned.elts) == 2  # externally visible h is preserved


@replace_all_device_with('cpu')
@pytest.mark.parametrize('layout', ['sequence', 'column_tp'])
def test_two_rank_codegen_adds_no_communication_to_merged_checkpoint(tmp_path, layout):
    def base(graph, config):
        ops = get_pas_ops(graph)
        for op in ops:
            rid = 0 if _Norm in op.module_class_chain else 1
            if layout == 'sequence':
                part = OpPartition(0, 0) if op is ops[0] else 'auto'
            else:
                part = OpPartition(1, 0) if op.fn == torch.nn.functional.linear else 'auto'
            yield OpPlan(op, rid, 0, partition=part)

    codes, report = _compile_pair(tmp_path, _NormCompute(), base, ComputeConfig(2, 2), degree=2)
    expected = 1 if layout == 'sequence' else 2
    for before, after in zip(codes[False], codes[True]):
        _assert_communication_unchanged(before, after, expected)
    assert report['groups_after'] == expected


class _SideInputModel(torch.nn.Module):
    def forward(self, x, side):
        side = torch.sin(side)
        return torch.tanh(torch.tanh(x) + side)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('side_dim', [0, 1])
def test_codegen_keeps_side_input_reshard_at_original_checkpoint_boundary(tmp_path, side_dim):
    def base(graph, config):
        first = True
        for op in get_pas_ops(graph):
            source = op.fn == torch.sin
            rid = -1 if source else 0 if first else 1
            if not source:
                first = False
            yield OpPlan(op, rid, 0, partition=OpPartition(0, side_dim if source else 0))

    codes, report = _compile_pair(tmp_path, _SideInputModel(), base, ComputeConfig(2, 2),
                                  inputs={'x': torch.randn(8, 8), 'side': torch.randn(8, 8)}, degree=2)
    expected = 1 if side_dim == 0 else 2
    assert report['groups_after'] == expected
    for before, after in zip(codes[False], codes[True]):
        _assert_communication_unchanged(before, after, expected)
        if side_dim == 1:
            old, new = _checkpoint_bodies(before), _checkpoint_bodies(after)
            assert _adapters(old[0])
            assert [_adapters(body) for body in old] == [_adapters(body) for body in new]


class _DropoutRegion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        return self.dropout(torch.tanh(self.proj(x)))


class _DropoutPair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = _DropoutRegion()
        self.second = _DropoutRegion()

    def forward(self, x):
        return self.second(self.first(x))


@replace_all_device_with('cpu')
def test_compiled_dropout_regions_preserve_outputs_gradients_and_rng(tmp_path):
    torch.manual_seed(827)
    original = _DropoutPair().train()
    inputs = torch.randn(4, 8)
    results, reports = [], []

    def base(graph, config):
        for op in get_pas_ops(graph):
            fqn = op.get_module_fqn(_DropoutRegion)
            yield OpPlan(op, 0 if fqn == 'first' else 1, 0)

    with mock_cube_env(0, 1):
        torch.distributed.init_process_group('gloo', rank=0, world_size=1,
                                            init_method=(tmp_path / 'rendezvous').as_uri())
        try:
            for merge in (False, True):
                compiled = parallelize(copy.deepcopy(original), {'x': inputs.clone().requires_grad_()},
                                       _policy(base, merge, reports), ComputeConfig(1, 1),
                                       gen_savedir=tmp_path / str(merge),
                                       instance_name=f'dropout_{merge}', load_module=True)
                x = inputs.clone().requires_grad_()
                torch.manual_seed(1917)
                y = compiled(x)
                forward_rng = torch.get_rng_state().clone()
                y.square().sum().backward()
                backward_rng = torch.get_rng_state().clone()
                grads = {compiled.fullmap[name].orig_name: param.grad.clone()
                         for name, param in compiled.named_parameters()}
                assert all(torch.isfinite(grad).all() for grad in grads.values())
                assert all(torch.count_nonzero(grad) for grad in grads.values())
                assert torch.equal(forward_rng, backward_rng)
                results.append((y.detach(), x.grad.clone(), grads, backward_rng))
        finally:
            torch.distributed.destroy_process_group()

    split, merged = results
    assert reports[0]['groups_before'] == 2 and reports[0]['groups_after'] == 1
    assert torch.equal(split[0], merged[0])
    assert torch.equal(split[1], merged[1])
    assert split[2].keys() == merged[2].keys()
    for name in split[2]:
        assert torch.equal(split[2][name], merged[2][name]), name
    assert torch.equal(split[3], merged[3])


class _PipelineNormCompute(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = torch.nn.ModuleList([_NormCompute() for _ in range(3)])

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x.sum()


@replace_all_device_with('cpu')
@pytest.mark.parametrize('offload', [False, True])
def test_six_rank_pipeline_codegen_preserves_stages_and_communication(tmp_path, offload):
    """Compile PP3 with two shards per stage; no distributed GPU execution."""

    def base(graph, config):
        first_in_stage = set()
        for op in get_pas_ops(graph):
            if _NormCompute not in op.module_class_chain:
                yield OpPlan(op, -1, 2)
                continue
            stage_id = int(op.get_module_fqn(_NormCompute).rsplit('.', 1)[1])
            region_id = stage_id * 2 + (0 if _Norm in op.module_class_chain else 1)
            partition = OpPartition(0, 0) if stage_id not in first_in_stage else 'auto'
            first_in_stage.add(stage_id)
            yield OpPlan(op, region_id, stage_id, partition=partition,
                         offload_id=region_id if offload else -1)

    config = ComputeConfig(6, 6, use_end2end=True,
                           cpu_offload=CPUOffloadConfig() if offload else None,
                           pas_config={'pipeline_nmicros': 4, 'pp_size': 3,
                                       'pipeline_scheduler': '1f1b'})
    codes, report = _compile_pair(tmp_path, _PipelineNormCompute(), base, config,
                                  degree=2, autoset_requires_grad=False)
    for merge in (False, True):
        for rank in range(6):
            cls = _load_parallel_module_class(_PipelineNormCompute, gen_savedir=tmp_path / str(merge), rank=rank)
            names = {meta.orig_name for meta in cls.attr_meta_maps[rank].values()}
            assert names == {f'stages.{rank // 2}.norm.weight',
                             f'stages.{rank // 2}.compute.proj.weight'}
            assert cls.nmicros_per_scheduler_step == 4

    for before, after in zip(codes[False], codes[True]):
        _assert_communication_unchanged(before, after, 1)
    assert report['groups_before'] == 6 and report['groups_after'] == 3
