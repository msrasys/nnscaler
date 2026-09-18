#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import tempfile
from pathlib import Path

import pytest
import torch

from nnscaler.autodist.apis import parallelize_graph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.flags import CompileFlag
from nnscaler.graph.gener.gen import IRAdapterGener
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.unique import IDGenerator
from nnscaler.parallel import ComputeConfig, parallelize
from nnscaler.policies import OpPlan, fn, pas_autodist
from nnscaler.program import Program, SemanticDataLoader, SemanticModel
from nnscaler.runtime.utils import microbatches
from tests.autodist.pas.test_multiref_activation import Decoder, ModelA
from tests.autodist.pas.test_multiref_param import Model
from tests.parallel_module.test_gencode import _gencode_contains


def test_legacy_is_default(tmp_path):
    assert AutoDistConfig(profile_dir=tmp_path).legacy is True


@pytest.mark.parametrize('update_freq', [[], {}, [1, 2], {0: 1, 10: 2}])
def test_multiple_update_freq_is_rejected(update_freq):
    config = ComputeConfig(1, 1, pas_config={'update_freq': update_freq})

    with pytest.raises(ValueError, match='only supports a single update_freq'):
        pas_autodist(object(), config)


@pytest.mark.parametrize('update_freq', [0, -1, '0'])
def test_non_positive_update_freq_is_rejected(update_freq):
    config = ComputeConfig(1, 1, pas_config={'update_freq': update_freq})

    with pytest.raises(ValueError, match='must be positive'):
        pas_autodist(object(), config)


@pytest.mark.parametrize('update_freq', [1.5, 'invalid', [2, 1.5]])
def test_non_integer_update_freq_is_rejected(update_freq):
    config = ComputeConfig(1, 1, pas_config={'update_freq': update_freq})

    with pytest.raises(ValueError, match='must be int'):
        pas_autodist(object(), config)


@pytest.mark.parametrize('update_freq', [2, '2', [2], (2, 2), {0: 2, 10: 2}])
def test_single_update_freq_is_supported(monkeypatch, tmp_path, update_freq):
    captured = {}

    def fake_parallelize_graph(graph, autodist_config):
        captured['update_freq'] = autodist_config.update_freq
        return graph

    monkeypatch.setattr('nnscaler.policies.parallelize_graph', fake_parallelize_graph)
    graph = object()
    config = ComputeConfig(
        1,
        1,
        pas_config={
            'update_freq': update_freq,
            'pipeline_nmicros': 4,
            'profile_dir': tmp_path,
            'mem_constraint': 1,
        },
    )

    assert pas_autodist(graph, config) is graph
    assert captured['update_freq'] == 2
    assert config.pas_config['pipeline_nmicros'] == 2


def test_inference_disables_auto_pipeline(monkeypatch, tmp_path):
    captured = {}

    def fake_parallelize_graph(graph, autodist_config):
        captured['pipeline_nstages'] = autodist_config.pipeline_nstages
        return graph

    monkeypatch.setattr('nnscaler.policies.parallelize_graph', fake_parallelize_graph)
    graph = object()
    config = ComputeConfig(
        2,
        2,
        inference_only=True,
        use_end2end=True,
        pas_config={
            'pipeline_nstages': 'auto',
            'pipeline_pivots': 'Layer',
            'profile_dir': tmp_path,
            'mem_constraint': 1,
        },
    )

    assert pas_autodist(graph, config) is graph
    assert captured['pipeline_nstages'] == 1


def test_inference_rejects_explicit_pipeline():
    config = ComputeConfig(
        2,
        2,
        inference_only=True,
        use_end2end=True,
        pas_config={
            'pipeline_nstages': 2,
            'pipeline_pivots': 'Layer',
        },
    )

    with pytest.raises(ValueError, match='not supported for inference'):
        pas_autodist(object(), config)


def test_autodist_rejects_zero3():
    config = ComputeConfig(1, 1, use_zero=3, pas_config={'mem_constraint': 1})

    with pytest.raises(ValueError, match='only supports zero_stage 1'):
        pas_autodist(object(), config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('cfg_fname', [
    'all_replicated_pp.json',
    'replicated_and_partition_pp.json',
    'replicated_and_partition_spmd.json',
])
def test_shared_param_pipeline_fn(cfg_fname, monkeypatch):
    batch_size, hidden_dim = 4, 1024

    monkeypatch.setattr(CompileFlag, 'dev_mode', True)

    with tempfile.TemporaryDirectory() as tempdir:
        model = Model(hidden_dim)
        model.train()

        program = Program()
        program.clear()
        IDGenerator().clear()

        dataloader = SemanticDataLoader(
            microbatches([{
                'x': torch.randn(batch_size, hidden_dim)
            }]))

        semantic_model = SemanticModel(model, attr_savedir=tempdir)
        semantic_model.dummy_input = {'x': torch.randn(batch_size, hidden_dim)}
        semantic_model.constant_folding = True
        program.set_input([dataloader.irobj])
        ir_dummy_input = next(dataloader)
        outputs = semantic_model(ir_dummy_input)
        outputs.backward()
        program.set_output([outputs])
        program.finalize()
        ir_graph = program.get_graph()

        plan_path = Path(os.path.dirname(__file__)) / cfg_fname
        autodist_config = AutoDistConfig(
            load_plan_path=plan_path,
            mesh_col=4,
            legacy=False,
        )
        compute_config = ComputeConfig(
            4,
            4,
            use_end2end='pp' in cfg_fname,
            pas_config={'pipeline_nmicros': autodist_config.update_freq},
        )

        op_plans = list(parallelize_graph(ir_graph, autodist_config))
        assert op_plans and all(isinstance(plan, OpPlan) for plan in op_plans)
        assert any(plan.stage_id == 0 for plan in op_plans)
        assert not ir_graph.select(ntype=IRSegment)

        graph = fn(ir_graph, compute_config, lambda *_: op_plans)
        graph = IRAdapterGener.gen(graph, cost_fn=None)
        if graph.sched is not None:
            graph.sched.apply()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_loss_output_identity_fn():
    model = ModelA()
    model.train()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    trace_data = torch.randn([2, 10], dtype=torch.float32, device=torch.cuda.current_device())

    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            model,
            {'x': trace_data},
            'autodist',
            ComputeConfig(
                1,
                1,
                use_end2end=True,
                pas_config={
                    'parallel_profile': False,
                    'legacy': False,
                },
            ),
            reuse='override',
            gen_savedir=tempdir,
            load_module=False,
        )

        assert len(_gencode_contains(
            tempdir,
            ModelA,
            0,
            'nnscaler.runtime.function.identity',
        )) == 1


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_activation_pipeline_fn():
    model = Decoder()
    model.train()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    trace_data = torch.randn([2, 10], dtype=torch.float32, device=torch.cuda.current_device())

    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            model,
            {'x': trace_data},
            'autodist',
            ComputeConfig(
                4,
                4,
                use_end2end=True,
                pas_config={
                    'load_plan_path': Path(__file__).parent / 'activation_pp.json',
                    'pipeline_nstages': 2,
                    'pipeline_pivots': 'Layer',
                    'legacy': False,
                },
            ),
            reuse='override',
            gen_savedir=tempdir,
            load_module=False,
        )

        assert True, "should not raise any exception"
