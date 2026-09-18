# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import copy
import tempfile
from pathlib import Path

import pytest
import torch

import nnscaler
from nnscaler import ComputeConfig, parallelize
from nnscaler.graph.schedule.predefined import PredefinedSched
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.policies import get_pas_ops, OpPlan
from tests.launch_torchrun import launch_torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0, replace_all_device_with


class MetadataPipeline(torch.nn.Module):
    def __init__(self, metadata_stage):
        super().__init__()
        self.metadata_stage = metadata_stage
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(4, 4, bias=False) for _ in range(4)
        ])

    def forward(self, sample):
        scale = sample['context'].get('scale', 7)
        x = self.layers[0](sample['x'])
        x = self.layers[1](x)
        x = self.layers[2](x)
        if self.metadata_stage == 2:
            x = x * scale
        x = self.layers[3](x)
        if self.metadata_stage == 3:
            x = x * scale
        return (x * x).sum()


def metadata_policy(graph, cfg):
    ops = get_pas_ops(graph)
    linears = [node for node in ops if node.fn == torch.nn.functional.linear]
    tensor_read = next(node for node in ops if node.signature == '_operator.getitem'
                       and isinstance(node.output(0), IRSubTensor))
    # A custom PAS groups tensor computation into virtual stages while leaving
    # sample metadata preprocessing at graph scope, outside scheduled segments.
    starts = [ops.index(tensor_read), *[ops.index(node) for node in linears[1:]], len(ops)]
    for start, end in zip(starts[:-1], starts[1:]):
        graph.group(ops[start:end])
    for node in list(graph.nodes()):
        if isinstance(node, (IRDataOperation, IRFwOperation)):
            for device, replica in enumerate(graph.replicate(node, 2)):
                graph.assign(replica, device)
    stages = [stage for stage in graph.select(ntype=IRSegment, flatten=False) if stage.isfw()]
    for index, stage in enumerate(stages):
        for node in stage.nodes():
            graph.assign(node, index % 2)
    PredefinedSched.sched_1f1b_interleaved(graph, cfg.pas_config['pipeline_nmicros'], 4)
    return graph


def fn_metadata_policy(graph, cfg):
    for op in get_pas_ops(graph):
        stage = -1
        if torch.nn.Linear in op.module_class_chain:
            stage = int(op.get_module_fqn(torch.nn.Linear).split('.')[-1])
        yield OpPlan(op, stage_id=stage, partition=None)


def metadata_config():
    return ComputeConfig(2, 2, use_end2end=True, constant_folding=False,
                         pas_config={'pipeline_nmicros': 2})


@replace_all_device_with('cpu')
@pytest.mark.parametrize('metadata_stage', [2, 3])
def test_vpp_sample_metadata_codegen(tmp_path, metadata_stage):
    parallelize(
        MetadataPipeline(metadata_stage),
        {'sample': {'x': torch.ones(2, 4), 'context': {'scale': -1}}},
        metadata_policy, metadata_config(),
        gen_savedir=tmp_path, load_module=False, reuse='override',
    )
    files = list(tmp_path.rglob(f'gencode{metadata_stage % 2}.py'))
    assert len(files) == 1
    tree = ast.parse(files[0].read_text())
    for function_name in ('_train_step', '_infer_step'):
        schedule = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == function_name)
        calls = [node for node in schedule.body
                 if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)]
        reads = [node for node in calls if ast.unparse(node.value.func) == 'next']
        lookups = [node for node in calls if ast.unparse(node.value.func) == 'dict.get']
        assert len(reads) == len(lookups) == 2
        roots = []
        for lookup in lookups:
            context, key, default = lookup.value.args
            assert ast.literal_eval(key) == 'scale' and ast.literal_eval(default) == 7
            assert ast.unparse(context.func) == '_operator.getitem'
            assert ast.literal_eval(context.args[1]) == 'context'
            roots.append(context.args[0].id)
        if metadata_stage == 2:
            # This rank already read each sample for its first virtual stage.
            assert roots == [node.targets[0].id for node in reads]
        else:
            # This rank has not read the sample yet; random access must leave
            # the two scheduled next() calls intact and use microbatches 0/1.
            random_reads = [node for node in calls
                            if isinstance(node.value.func, ast.Attribute)
                            and node.value.func.attr == 'get_micro_batch']
            assert [ast.literal_eval(node.value.args[0]) for node in random_reads] == [0, 1]
            assert roots == [node.targets[0].id for node in random_reads]


def metadata_worker(metadata_stage, use_fn=False):
    nnscaler.init()
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(7)
    source = MetadataPipeline(metadata_stage)
    reference = copy.deepcopy(source).cuda()
    directory = Path(tempfile.gettempdir()) / f'metadata_vpp_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        cfg = metadata_config()
        if use_fn:
            cfg.pas_config.update(pipeline_size=2, pipeline_scheduler='1f1b_interleaved')
        model = parallelize(
            source, {'sample': {'x': torch.ones(2, 4), 'context': {'scale': -1}}},
            fn_metadata_policy if use_fn else metadata_policy,
            cfg, gen_savedir=tempdir, reuse='override',
        ).cuda()
        samples = [
            {'x': torch.arange(8, device='cuda').reshape(2, 4).float() / 8,
             'context': {'scale': -1}},
            {'x': torch.ones(2, 4, device='cuda'), 'context': {}},
        ]
        expected = [reference(sample) for sample in samples]
        torch.stack(expected).sum().backward()
        actual = model.train_step(samples)
        for output, target in zip(actual, expected):
            torch.testing.assert_close(output, target, atol=1e-6, rtol=1e-5)
        params = dict(reference.named_parameters())
        for name, meta in model.fullmap.items():
            torch.testing.assert_close(
                getattr(model, name).grad, params[meta.orig_name].grad[meta.slicers],
                atol=1e-6, rtol=1e-5,
            )
        for output, target in zip(model.infer_step(samples), expected):
            torch.testing.assert_close(output, target, atol=1e-6, rtol=1e-5)
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires 2 GPUs')
@pytest.mark.parametrize('metadata_stage', [2, 3])
def test_vpp_sample_metadata_matches_eager(metadata_stage):
    assert all(launch_torchrun(2, metadata_worker, metadata_stage).values())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires 2 GPUs')
@pytest.mark.parametrize('metadata_stage', [2, 3])
def test_fn_sample_metadata_matches_eager(metadata_stage):
    # This is a compatibility check; fn already handles this case on main.
    assert all(launch_torchrun(2, metadata_worker, metadata_stage, True).values())
