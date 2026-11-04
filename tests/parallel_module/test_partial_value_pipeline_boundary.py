#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Pipeline boundary tensors that are value partitions.

When the producer of a segment output is split along a reduction dim, every
device holds the full-shape output with a different ``valmap``.  The segment
expander narrows such outputs to these partial sums like index partitions, and
the cross-stage adapter reduces them while routing to the consumers.  Each user
of the per-device IO must identify a partition by indmap and valmap together:
``ExeReuseCell.dispatch`` must not compare shapes, and an internal consumer of
the full value is not satisfied by a partial sum of the same region.

The in-stage consumer case uses ``graph.partition`` directly: ``fn`` inserts an
identity for any output consumed inside its own stage, so through ``fn`` a
partial sum with an in-stage consumer never reaches the boundary.
"""

import ast
import builtins
import symtable
import tempfile
from pathlib import Path

import pytest
import torch

from nnscaler import parallelize, ComputeConfig
from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRDataOperation
from nnscaler.parallel import merge_state_dicts
from nnscaler.policies import _tp, _replica
from tests.launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from tests.utils import clear_dir_on_rank0, init_random, replace_all_device_with, PYTEST_RUN_ID
from tests.parallel_module.common import init_distributed
from tests.parallel_module.test_gencode import _gencode_contains

DIM = 16
NMICROS = 4


class PartialValueBoundaryModule(torch.nn.Module):
    """``layers[1]`` is split along its reduction dim, so its output ``y`` crosses the
    stage boundary as a partial sum.  With ``internal_consumer`` a replicated ``mul``
    in stage 0 also needs the full value of ``y``."""

    def __init__(self, internal_consumer: bool = False):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(DIM, DIM, bias=False) for _ in range(4)])
        self.internal_consumer = internal_consumer

    def forward(self, data: torch.Tensor):
        x = self.layers[0](data)   # stage 0, split on dim 0
        y = self.layers[1](x)      # stage 0, split on the reduction dim
        if self.internal_consumer:
            w = y * 2              # stage 0, replicated: needs the full value of y
        z = self.layers[2](y)      # stage 1, split on dim 0
        if self.internal_consumer:
            z = z + w
        z = self.layers[3](z)
        return torch.sum(z)


class InternalConsumerModule(PartialValueBoundaryModule):
    def __init__(self):
        super().__init__(internal_consumer=True)


def partial_value_boundary_pas(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_layer_index, get_pas_ops

    stage_id = 0
    for node in get_pas_ops(graph):
        if torch.nn.modules.linear.Linear in node.module_class_chain:
            layer_idx = get_layer_index(node.fqn)
            stage_id = layer_idx // 2
            dim = 1 if layer_idx == 1 else 0
            yield OpPlan(node, stage_id=stage_id, partition=OpPartition(input=0, dim=dim))
        elif node.fn is torch.sum:
            yield OpPlan(node, stage_id=stage_id, partition=OpPartition(input=0, dim=0))
        else:
            yield OpPlan(node, stage_id=stage_id)


def internal_consumer_pas(graph: IRGraph, cfg: ComputeConfig):
    """Old-style policy on purpose: this layout cannot be reached through ``fn``.

    ``fn`` wraps every stage output that is also consumed inside its stage in an
    identity (``_identity_segment_output``) and plans that identity as a replicated
    op, so the partial sums of ``layers[1]`` would be all-reduced before the identity
    and the segment output would be a full-value tensor.  Staging first and then
    partitioning by hand keeps the raw partial sums as the segment output while
    the replicated ``mul`` consumes them in stage 0.
    """
    linears = graph.select(name='linear')
    graph.staging([linears[0], linears[2]])
    stage0, stage1 = [seg for seg in graph.select(ntype=IRSegment, flatten=False) if seg.isfw()]
    half = cfg.plan_ngpus // 2
    devs0, devs1 = list(range(half)), list(range(half, cfg.plan_ngpus))
    for node in list(stage0.nodes()):
        if node is linears[1]:
            _tp(graph, node, idx=0, dim=1, devs=devs0)   # reduction dim -> partial sums
        elif node.name == 'linear':
            _tp(graph, node, idx=0, dim=0, devs=devs0)
        else:
            _replica(graph, node, devs=devs0)           # `mul` consumes the full value of y
    for node in list(stage1.nodes()):
        if node.name in ('linear', 'add', 'sum'):
            _tp(graph, node, idx=0, dim=0, devs=devs1)
        else:
            _replica(graph, node, devs=devs1)
    for dataloader in graph.select(ntype=IRDataOperation):
        _replica(graph, dataloader, devs=devs0 + devs1)
    cfg.apply_pipeline_scheduler(graph, 2, NMICROS, '1f1b')
    return graph


def _pipeline_config(ngpus: int, use_async_comm: bool = False) -> ComputeConfig:
    return ComputeConfig(
        ngpus, ngpus,
        use_end2end=True,
        use_async_comm=use_async_comm,
        pas_config={'pipeline_nmicros': NMICROS, 'pipeline_scheduler': '1f1b'},
    )


def _worker(use_async_comm: bool):
    torch.set_float32_matmul_precision("highest")
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'partial_value_boundary_{int(use_async_comm)}_{PYTEST_RUN_ID}') as tempdir:
        init_random()
        model = parallelize(
            PartialValueBoundaryModule(),
            {'data': torch.randn(8, DIM)},
            pas_policy=partial_value_boundary_pas,
            compute_config=_pipeline_config(4, use_async_comm),
            gen_savedir=tempdir,
            reuse='override',
        )
        model.cuda()
        init_random()
        model.train_step([torch.randn(8, DIM) for _ in range(NMICROS)])
        grads = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
        model._add_extra_state(grads, '')
        return clone_to_cpu_recursively(grads)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('use_async_comm', [
    False,
    # The reduce-scatter output is sent by the following `move` before the
    # collective completes: nothing waits on async dependencies inside an
    # adapter. Fixed by msrasys/nnscaler#146; drop the mark once it lands.
    pytest.param(True, marks=pytest.mark.xfail(strict=True, reason='async adapter dependencies, see #146')),
])
def test_partial_value_pipeline_boundary(use_async_comm):
    torch.set_float32_matmul_precision("highest")
    results = launch_torchrun(4, _worker, use_async_comm)
    merged_grads, _ = merge_state_dicts([results[rank] for rank in range(4)])

    torch.cuda.set_device(0)
    with torch.device('cuda:0'):
        init_random()
        reference = PartialValueBoundaryModule()
        init_random()
        losses = [reference(torch.randn(8, DIM)) for _ in range(NMICROS)]
        torch.stack(losses).sum().backward()

    reference_grads = {
        name: param.grad.cpu()
        for name, param in reference.named_parameters()
        if param.grad is not None
    }
    assert merged_grads.keys() == reference_grads.keys()
    for name, grad in reference_grads.items():
        torch.testing.assert_close(merged_grads[name], grad, atol=1e-5, rtol=1e-5)


def _unbound_names(gencode: Path):
    """Names a generated function references without any binding in scope.

    A per-device IO that is not narrowed consistently shows up here: the segment
    returns one tensor id while the adapter consumes another.
    """
    # `del x` counts as binding x in the function, which would turn a never-assigned
    # name into an "assigned local" and hide it; drop the del statements first so
    # such a name is resolved as a global and reported below.
    tree = ast.parse(gencode.read_text())
    for node in ast.walk(tree):
        for field, stmts in ast.iter_fields(node):
            if isinstance(stmts, list):
                setattr(node, field, [ast.Pass() if isinstance(stmt, ast.Delete) else stmt for stmt in stmts])
    top = symtable.symtable(ast.unparse(tree), str(gencode), 'exec')
    known = set(top.get_identifiers()) | set(dir(builtins))
    unbound = []
    pending = list(top.get_children())
    while pending:
        scope = pending.pop()
        pending.extend(scope.get_children())
        if scope.get_type() != 'function':
            continue
        unbound += [
            (scope.get_name(), symbol.get_name())
            for symbol in scope.get_symbols()
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in known
        ]
    return unbound


def _gencode(module_class, pas, ngpus: int, tempdir: Path):
    from nnscaler.parallel import _PARALLEL_MODULE_NAMESPACE, _get_full_qualified_name, _DEFAULT_INSTANCE_NAME

    parallelize(
        module_class(),
        {'data': torch.randn(8, DIM)},
        pas_policy=pas,
        compute_config=_pipeline_config(ngpus),
        gen_savedir=tempdir,
        load_module=False,
        reuse='override',
    )
    namespace = f'{_PARALLEL_MODULE_NAMESPACE}.{_get_full_qualified_name(module_class)}.{_DEFAULT_INSTANCE_NAME}'
    outdir = tempdir / namespace.replace('.', '/').strip('/')
    for rank in range(ngpus):
        gencode = outdir / f'gencode{rank}.py'
        assert _unbound_names(gencode) == [], f'unbound names in {gencode}'


@replace_all_device_with('cpu')
def test_gencode_partial_value_boundary(tmp_path):
    _gencode(PartialValueBoundaryModule, partial_value_boundary_pas, 8, tmp_path)
    # the partial sums are reduced while being routed to stage 1, not inside stage 0
    assert _gencode_contains(tmp_path, PartialValueBoundaryModule, 0, r'adapter\.reduce_scatter\(linear_1_\d+, dim=0')
    assert not _gencode_contains(tmp_path, PartialValueBoundaryModule, 0, r'allreduce_identity\(')


@replace_all_device_with('cpu')
def test_gencode_partial_value_internal_consumer(tmp_path):
    _gencode(InternalConsumerModule, internal_consumer_pas, 8, tmp_path)
    # the in-stage consumer needs the full value, so the output is not narrowed
    # and the partial sums are reduced inside stage 0
    assert _gencode_contains(tmp_path, InternalConsumerModule, 0, r'allreduce_identity\(linear_1_\d+')
    assert not _gencode_contains(tmp_path, InternalConsumerModule, 0, r'adapter\.reduce_scatter\(linear_1_')
