#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import contextlib
import copy
import tempfile
from pathlib import Path

import pytest
import torch
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

import nnscaler
import nnscaler.policies as policies
from nnscaler import ComputeConfig, build_optimizer, parallelize
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.ir import IRSubTensor
from nnscaler.ir.operator import IRBpOperation, IRFwOperation
from nnscaler.policies import OpPartition, OpPlan, get_layer_index, get_pas_ops
from nnscaler.runtime.function import identity
from tests.launch_torchrun import launch_torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0, raises_with_cause, replace_all_device_with


class PipelineWithAuxOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aux = torch.nn.Linear(4, 4, bias=False)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(4, 4, bias=False) for _ in range(2)
        ])

    def forward(self, x):
        aux = self.aux(x)
        hidden = self.layers[0](x)
        loss = self.layers[1](hidden).sum()
        return loss, aux, hidden, self.layers[0].weight


def two_stage_policy(graph, cfg):
    stage_id = 0
    for node in get_pas_ops(graph):
        if torch.nn.Linear in node.module_class_chain and 'layers.' in node.fqn:
            stage_id = get_layer_index(node.fqn)
        partition = None
        if torch.nn.Linear in node.module_class_chain:
            partition = OpPartition(input=1 if node.fqn == 'layers.0' else 0, dim=0)
            if node.fqn == 'aux' and cfg.pas_config.get('test_value_partition'):
                partition = OpPartition(input=1, dim=1)
        yield OpPlan(node, stage_id=stage_id, partition=partition)


@nnscaler.register_op('* -> *')
def _aux_with_grad_factory(x):
    # An opaque op can produce a grad-enabled output from detached inputs.
    return x.sin() + torch.ones_like(x, requires_grad=True)


class PipelineWithUnusedStageInputs(torch.nn.Module):
    # The loss follows layers 0 -> 1 -> 2 -> 3; aux uses an early value in stage 3.
    def __init__(self, detach_method='data'):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(4, 4, bias=False) for _ in range(4)
        ])
        self.aux = torch.nn.Linear(4, 4, bias=False)
        self.detach_method = detach_method

    def forward(self, x):
        first = self.layers[0](x)            # stage 0
        hidden = self.layers[1](first)       # stage 1
        hidden = self.layers[2](hidden)      # stage 2
        hidden = self.layers[3](hidden)      # stage 3
        loss = hidden.square().sum()
        aux = self._auxiliary(first)         # stage 3, not part of loss
        return self._outputs(loss, aux)

    def _auxiliary(self, x):
        return self.aux(x)

    def _outputs(self, loss, aux):
        if self.detach_method == 'data':
            aux = aux.data
        elif self.detach_method == 'detach':
            aux = aux.detach()
        return loss, aux


class PipelineWithIndependentAux(PipelineWithUnusedStageInputs):
    def __init__(self, detach_method='data'):
        super().__init__(detach_method)
        self.side = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        first = self.layers[0](x)
        carried = self.side(x)               # stage 0, never used by loss
        hidden = self.layers[1](first)
        hidden = self.layers[2](hidden)
        hidden = self.layers[3](hidden)
        loss = hidden.square().sum()
        aux = self.aux(carried)              # stage 3
        return self._outputs(loss, aux)


class PipelineWithLiveAuxInput(PipelineWithUnusedStageInputs):
    def forward(self, x):
        first = self.layers[0](x)
        hidden = self.layers[1](first)
        hidden = self.layers[2](hidden)
        hidden = self.layers[3](hidden)
        loss = hidden.square().sum()
        aux = self.aux(first)
        loss = loss + first.sum()           # stage 3 still needs first's gradient
        return self._outputs(loss, aux)


class PipelineWithSharedWeight(PipelineWithUnusedStageInputs):
    def __init__(self, detach_method='data'):
        super().__init__(detach_method)
        # In stage 3 the same weight has both a live loss use and an unused aux use.
        self.aux.weight = self.layers[0].weight
        self.layers[3].weight = self.layers[0].weight


class PipelineWithSinAux(PipelineWithUnusedStageInputs):
    def _auxiliary(self, x):
        return x.sin()


class PipelineWithGradFactoryAux(PipelineWithUnusedStageInputs):
    def _auxiliary(self, x):
        return _aux_with_grad_factory(x)


class PipelineWithNoGradAux(PipelineWithUnusedStageInputs):
    def _auxiliary(self, x):
        with torch.no_grad():
            return x.sin()


class PipelineWithAuxOnlyStage(PipelineWithUnusedStageInputs):
    def forward(self, x):
        first = self.layers[0](x)
        hidden = self.layers[1](first)
        hidden = self.layers[2](hidden)
        loss = hidden.square().sum()         # loss ends in stage 2
        hidden = self.layers[3](hidden)
        aux = self.aux(first) + hidden       # stage 3 has only auxiliary outputs
        return self._outputs(loss, aux)


def unused_stage_input_policy(graph, cfg):
    stage_id = 0
    memory_mode = cfg.pas_config.get('test_memory_mode')
    for node in get_pas_ops(graph):
        if torch.nn.Linear in node.module_class_chain and 'layers.' in node.fqn:
            stage_id = get_layer_index(node.fqn)
        partition = None
        if cfg.plan_ngpus == 4 and torch.nn.Linear in node.module_class_chain:
            partition = OpPartition(input=0, dim=0)
        yield OpPlan(
            node, stage_id=stage_id, partition=partition,
            recompute_id=stage_id if memory_mode == 'recompute' else -1,
            offload_id=stage_id if memory_mode == 'offload' else -1,
        )


def worker_aux_output(model_cls, model_args, policy, use_fbw, memory_mode=None, value_partition=False):
    nnscaler.init()
    torch.manual_seed(7)
    source = model_cls(*model_args)
    reference = copy.deepcopy(source).cuda()
    reference_parameters = dict(reference.named_parameters())
    optimizer_args = dict(lr=0.001, momentum=0.9, weight_decay=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), **optimizer_args)
    nstages = len(source.layers)

    directory = Path(tempfile.gettempdir()) / f'aux_graph_output_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        model = parallelize(
            source, {'x': torch.ones(4, 4)}, policy,
            ComputeConfig(
                4, 4, use_end2end=True, use_fbw=use_fbw,
                # Preserve None for unused parameters, with one parameter per bucket.
                reducer_none_grad=True, reducer_bucket_cap_mb=16 / 1024**2,
                pas_config={
                    'pipeline_size': 2, 'pipeline_nmicros': nstages,
                    'pipeline_scheduler': '1f1b_interleaved' if nstages == 4 else '1f1b',
                    'test_memory_mode': memory_mode, 'test_value_partition': value_partition,
                },
            ),
            gen_savedir=tempdir, reuse='override',
        ).cuda()
        optimizer = build_optimizer(model, torch.optim.SGD, **optimizer_args)
        for step in range(2):
            samples = [
                torch.arange(16, device='cuda').float().reshape(4, 4) / 16 + micro + step
                for micro in range(nstages)
            ]
            reference_optimizer.zero_grad()
            expected = [reference(sample) for sample in samples]
            torch.stack([output[0] for output in expected]).sum().backward()
            actual = model.train_step(samples)
            torch.testing.assert_close(actual, expected)
            for _, *outputs in actual:
                assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in outputs)
            for name, meta in model.fullmap.items():
                actual_grad = getattr(model, name).grad
                expected_grad = reference_parameters[meta.orig_name].grad
                if expected_grad is None:
                    assert actual_grad is None, meta.orig_name
                else:
                    torch.testing.assert_close(actual_grad, expected_grad[meta.slicers])
            # A fabricated zero gradient would update unused weights via weight decay.
            optimizer.step()
            reference_optimizer.step()
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(
                    getattr(model, name), reference_parameters[meta.orig_name][meta.slicers],
                )
            optimizer.zero_grad()
    return True


# Returned aux must be detached without cutting hidden/weight from the loss path.
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
@pytest.mark.parametrize('use_fbw,value_partition', [
    (False, False), # Dimension-partitioned outputs and a returned weight.
    (True, False),  # The same outputs with separate B/W execution.
    (False, True),  # Value-partitioned aux must be gathered before detach.
])
def test_fn_aux_outputs(use_fbw, value_partition):
    assert all(launch_torchrun(
        4, worker_aux_output, PipelineWithAuxOutput, (), two_stage_policy,
        use_fbw, None, value_partition,
    ).values())


# Only representative combinations are needed; the cases below target different cuts.
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
@pytest.mark.parametrize('use_fbw,model_cls,detach_method', [
    # Both output and boundary need detaching.
    (False, PipelineWithUnusedStageInputs, 'implicit'),
    # Output .data / .detach() alone does not cut the boundary.
    (False, PipelineWithUnusedStageInputs, 'data'),
    (False, PipelineWithUnusedStageInputs, 'detach'),
    # No loss consumer: cut at the producer stage.
    (False, PipelineWithIndependentAux, 'data'),
    # A late loss consumer must keep its gradient.
    (False, PipelineWithLiveAuxInput, 'data'),
    # Detach the aux weight use, not the live use in the same stage.
    (False, PipelineWithSharedWeight, 'data'),
    # Activation and shared-parameter boundary cuts with separate B/W execution.
    (True, PipelineWithUnusedStageInputs, 'implicit'),
    (True, PipelineWithSharedWeight, 'implicit'),
    # Preserve internally created gradients and the traced no_grad context.
    (False, PipelineWithGradFactoryAux, 'implicit'),
    (False, PipelineWithNoGradAux, 'implicit'),
])
def test_fn_aux_boundary(use_fbw, model_cls, detach_method):
    assert all(launch_torchrun(
        4, worker_aux_output, model_cls, (detach_method,), unused_stage_input_policy, use_fbw,
    ).values())


# Detaches must belong to the insertion stage's memory region, not the producer's.
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
@pytest.mark.parametrize('use_fbw,memory_mode,model_cls', [
    # Shared parameters must retain their loss gradient in both memory modes.
    (False, 'recompute', PipelineWithSharedWeight), (True, 'offload', PipelineWithSharedWeight),
    # Sin retains conservative True in the IR after its input is detached.
    (False, 'recompute', PipelineWithSinAux), (True, 'offload', PipelineWithSinAux),
])
def test_fn_aux_memory_regions(use_fbw, memory_mode, model_cls):
    assert all(launch_torchrun(
        4, worker_aux_output, model_cls, ('implicit',), unused_stage_input_policy, use_fbw, memory_mode,
    ).values())


# Removing an aux gradient also changes the live consumer's value map.
# Update both existing backward nodes, including an empty output list for sin.
@replace_all_device_with('cpu')
@pytest.mark.parametrize('model_cls', [
    PipelineWithUnusedStageInputs, PipelineWithSinAux, PipelineWithSharedWeight,
])
def test_fn_aux_updates_backward_locally(tmp_path, monkeypatch, model_cls):
    insert = policies._pp_detach_aux_outputs
    checked = []

    def checked_insert(graph, plans):
        backward_ids = [node.cid for node in graph.select(ntype=IRBpOperation)]
        layer, = [
            node for node in plans
            if node.fqn == 'layers.1' and torch.nn.Linear in node.module_class_chain
        ]
        source = layer.input(0)
        aux, = [node for node in graph.consumers(source.parent) if node != layer]
        assert source.grad.valmap == (0, 2)
        assert aux.input(0).grad.valmap == (1, 2)

        insert(graph, plans)

        assert [node.cid for node in graph.select(ntype=IRBpOperation)] == backward_ids
        assert layer.input(0) == source
        assert layer.input(0).grad.valmap == (0, 1)
        assert aux.input(0).parent != source.parent
        assert aux.input(0).grad is None
        for node in (layer, aux):
            assert node.mirror.outputs() == tuple(
                tensor.grad for tensor in node.iobjs()
                if isinstance(tensor, IRSubTensor) and tensor.grad is not None
            )
        detaches = [
            node for node in plans if node.comment == 'fn detached pipeline output'
        ]
        assert detaches and all(node.mirror is None for node in detaches)
        checked.append(True)

    monkeypatch.setattr(policies, '_pp_detach_aux_outputs', checked_insert)
    parallelize(
        model_cls('data'), {'x': torch.ones(4, 4)}, unused_stage_input_policy,
        ComputeConfig(4, 4, use_end2end=True, pas_config={
            'pipeline_size': 2, 'pipeline_nmicros': 4, 'pipeline_scheduler': '1f1b_interleaved',
        }),
        gen_savedir=tmp_path, load_module=False, reuse='override',
    )
    assert checked == [True]


# Input flags cannot determine output flags: sin may retain conservative True,
# a factory really creates a grad-enabled output, and no_grad must remain False.
@replace_all_device_with('cpu')
@pytest.mark.parametrize('model_cls,signature,requires_grad', [
    (PipelineWithSinAux, 'torch.sin', True),
    (PipelineWithGradFactoryAux, '._aux_with_grad_factory', True),
    (PipelineWithNoGradAux, 'torch.sin', False),
])
def test_fn_aux_gradient_metadata(tmp_path, monkeypatch, model_cls, signature, requires_grad):
    staging = IRGraph.staging
    checked = []

    def checked_staging(graph, nodes):
        aux, = [node for node in graph.select(ntype=IRFwOperation) if node.signature.endswith(signature)]
        assert not aux.input(0).requires_grad
        assert aux.output(0).requires_grad == requires_grad
        assert not graph.output(1).requires_grad and graph.output(1).grad is None
        if not requires_grad:
            assert aux.op_context['grad_mode']['no_grad_mode']
        checked.append(True)
        return staging(graph, nodes)

    monkeypatch.setattr(IRGraph, 'staging', checked_staging)
    parallelize(
        model_cls('implicit'),
        {'x': torch.ones(4, 4)}, unused_stage_input_policy,
        ComputeConfig(4, 4, use_end2end=True, pas_config={
            'pipeline_size': 2, 'pipeline_nmicros': 4, 'pipeline_scheduler': '1f1b_interleaved',
        }),
        gen_savedir=tmp_path, load_module=False, reuse='override',
    )
    assert checked == [True]


# Auto partition follows dimension shards, gathers value shards, and uses a
# replicated fallback for returned parameters (which have no producer).
@replace_all_device_with('cpu')
@pytest.mark.parametrize('value_partition', [False, True])
def test_fn_aux_output_partition(tmp_path, monkeypatch, value_partition):
    schedule = ComputeConfig.apply_pipeline_scheduler
    checked = []

    def checked_schedule(cfg, graph, *args):
        detaches = [
            node for stage in graph.select(ntype=IRSegment, flatten=False)
            for node in stage.select(ntype=IRFwOperation)
            if node.comment == 'fn detached pipeline output'
        ]
        for node in detaches:
            assert node.input(0).indmap == node.output(0).indmap
            assert node.input(0).valmap == node.output(0).valmap
            assert node.input(0).grad is None and node.output(0).grad is None
        aux = [node.output(0) for node in detaches if node.output(0).parent == graph.output(1).parent]
        parameter = [node.output(0) for node in detaches if node.output(0).parent == graph.output(3).parent]
        assert len(aux) == len(parameter) == 2
        assert all(tensor.valmap == (0, 1) for tensor in aux)
        assert {tensor.indmap for tensor in aux} == (
            {((0, 2), (0, 4))} if value_partition else {((0, 1), (0, 4)), ((1, 2), (0, 4))}
        )
        assert all(tensor.shape == (4, 4) and not tensor.splitdims() for tensor in parameter)
        checked.append(True)
        return schedule(cfg, graph, *args)

    monkeypatch.setattr(ComputeConfig, 'apply_pipeline_scheduler', checked_schedule)
    parallelize(
        PipelineWithAuxOutput(), {'x': torch.ones(2, 4)}, two_stage_policy,
        ComputeConfig(4, 4, use_end2end=True, pas_config={
            'pipeline_size': 2, 'pipeline_nmicros': 2, 'pipeline_scheduler': '1f1b',
            'test_value_partition': value_partition,
        }),
        gen_savedir=tmp_path, load_module=False, reuse='override',
    )
    assert checked == [True]


class PipelineWithCarriedOutput(torch.nn.Module):
    def __init__(self, detached):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(4, 4, bias=False) for _ in range(4)
        ])
        self.detached = detached

    def forward(self, x):
        x = self.layers[0](x)
        carried = x.detach() if self.detached else x
        hidden = x
        for layer in self.layers[1:]:
            hidden = layer(hidden)
        return (hidden + carried).sum(), {'carried': carried, 'repeated': [carried]}


# Every returned alias must follow the identities inserted at stage boundaries.
@replace_all_device_with('cpu')
@pytest.mark.parametrize('detached', [False, True])
def test_fn_nested_aux_outputs(tmp_path, monkeypatch, detached):
    staging = IRGraph.staging
    checked = []

    def checked_staging(graph, nodes):
        carried = graph.output(1)['carried']
        detach = None
        if not detached:
            detach, = graph.producers(carried.parent)
            carried = detach.input(0)
        staging(graph, nodes)
        stages = [stage for stage in graph.select(ntype=IRSegment, flatten=False) if stage.isfw()]
        assert len(stages) == 4
        for stage in stages[1:]:
            relay, = [
                node for node in stage.nodes()
                if node.isfw() and node.fn == identity and node.input(0) == carried
            ]
            carried = relay.output(0)
        if detach is not None:
            assert detach.input(0) == carried
        returned = detach.output(0) if detach is not None else carried
        assert graph.output(1) == {'carried': returned, 'repeated': [returned]}
        assert returned in stages[-1].outputs()
        checked.append(True)

    monkeypatch.setattr(IRGraph, 'staging', checked_staging)
    parallelize(
        PipelineWithCarriedOutput(detached), {'x': torch.ones(2, 4)}, unused_stage_input_policy,
        ComputeConfig(4, 4, use_end2end=True, pas_config={
            'pipeline_size': 4, 'pipeline_nmicros': 4, 'pipeline_scheduler': '1f1b',
        }),
        gen_savedir=tmp_path, load_module=False, reuse='override',
    )
    assert checked == [True]


# Inference has no loss gradient; auxiliary-output handling must leave compilation valid.
@replace_all_device_with('cpu')
def test_fn_aux_output_inference(tmp_path):
    parallelize(
        PipelineWithAuxOutput(), {'x': torch.ones(2, 4)}, two_stage_policy,
        ComputeConfig(4, 4, use_end2end=True, inference_only=True, pas_config={
            'pipeline_size': 2, 'pipeline_nmicros': 2, 'pipeline_scheduler': 'infer_pipe',
        }),
        gen_savedir=tmp_path, load_module=False, reuse='override',
    )


class PipelineWithUnusedWeight(torch.nn.Module):
    def __init__(self, return_weight):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(4, 4, bias=False) for _ in range(2)
        ])
        self.unused = torch.nn.Parameter(torch.ones(3, 7))
        self.return_weight = return_weight

    def forward(self, x):
        loss = self.layers[1](self.layers[0](x)).sum()
        return (loss, self.unused) if self.return_weight else loss


# An unreferenced weight is absent from the graph. Returning it introduces a
# parameter with no consumer, so there is no stage in which to place its detach.
@replace_all_device_with('cpu')
@pytest.mark.parametrize('return_weight', [False, True])
def test_fn_unused_weight_output(tmp_path, return_weight):
    expected_error = (
        raises_with_cause(RuntimeError, match='with no consumers')
        if return_weight else contextlib.nullcontext()
    )
    with expected_error:
        parallelize(
            PipelineWithUnusedWeight(return_weight), {'x': torch.ones(2, 4)}, two_stage_policy,
            ComputeConfig(4, 4, use_end2end=True, pas_config={
                'pipeline_size': 2, 'pipeline_nmicros': 2, 'pipeline_scheduler': '1f1b',
            }),
            gen_savedir=tmp_path, load_module=False, reuse='override',
        )


# Keep the known unsupported case separate from the supported boundary cuts.
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
@pytest.mark.parametrize('use_fbw', [False, True])
@pytest.mark.xfail(
    strict=True, raises=ChildFailedError,
    reason='A stage with only detached outputs lacks consistent backward interfaces and state.',
)
def test_fn_aux_only_stage(use_fbw):
    assert all(launch_torchrun(
        4, worker_aux_output, PipelineWithAuxOnlyStage, ('data',), unused_stage_input_policy, use_fbw,
    ).values())
