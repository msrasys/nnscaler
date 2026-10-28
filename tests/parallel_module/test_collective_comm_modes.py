# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
from dataclasses import replace

import pytest
import torch

from nnscaler import parallelize
from nnscaler.codegen.emit import FuncEmission
from nnscaler.flags import CompileFlag
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import AllGatherPrim
from nnscaler.ir.cten import IR
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.parallel import _compile_flags
from tests.parallel_module.test_collective_layout_model import (
    assert_generated_comm_mode, model_case, pipeline_config, requested_async_comm,
)
from tests.parallel_module.test_vpp_colocated_params import ReusedBlocks, colocated_config, shared_vpp_policy
from tests.utils import replace_all_device_with


@pytest.mark.parametrize('mode', [False, True])
def test_requested_mode_is_explicit_in_all_matrix_configurations(monkeypatch, mode):
    monkeypatch.setenv('ASYNC_COMM', str(int(mode)))
    requested = requested_async_comm()
    assert requested is mode
    assert pipeline_config(requested).use_async_comm is mode
    assert model_case(4, requested)[2].use_async_comm is mode
    assert replace(model_case(4, requested)[2], runtime_ngpus=8).use_async_comm is mode
    assert colocated_config(use_async_comm=requested).use_async_comm is mode


@pytest.mark.parametrize('mode', [False, True])
def test_configured_mode_reaches_an_ordinary_collective_call(monkeypatch, mode):
    source = IR.set_object_device(IRFullTensor((2,), name='source').tosub(), 0)
    gathered = IR.set_object_device(IRFullTensor((4,), name='gathered').tosub(), 0)
    adapter = IRAdapter([source], [gathered])
    adapter.prims = [AllGatherPrim([source], [gathered], dim=0, ranks=(0, 1))]
    config = model_case(4, use_async_comm=mode)[2]
    # The runtime environment flag must not silently replace the explicit
    # configuration, even when it initially requests the opposite mode.
    monkeypatch.setattr(CompileFlag, 'async_comm', not mode)
    with _compile_flags(config):
        code = '\n'.join(FuncEmission().emit_adapter(adapter, async_op=CompileFlag.async_comm))
    calls = [node for node in ast.walk(ast.parse(code))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == 'all_gather']
    assert len(calls) == 1
    keywords = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords if kw.arg == 'async_op'}
    assert keywords.get('async_op', False) is mode
    assert CompileFlag.async_comm is not mode


@pytest.mark.parametrize('mode', [False, True])
@pytest.mark.parametrize('case', ['mesh', 'dp', 'pipeline', 'vpp'])
@replace_all_device_with('cpu')
def test_matrix_codegen_checks_both_modes_and_retains_p2p_waits(tmp_path, mode, case):
    if case == 'vpp':
        source, policy, config = ReusedBlocks(), shared_vpp_policy, colocated_config(use_async_comm=mode)
        inputs = {'x': torch.ones(8, 8)}
    else:
        source, policy, config = model_case(8 if case == 'pipeline' else 4, use_async_comm=mode)
        if case == 'dp':
            config = replace(config, runtime_ngpus=8)
        inputs = {'sample': {'x': torch.ones(8, 8)}}
    parallelize(source, inputs, policy, config, gen_savedir=tmp_path,
                load_module=False, reuse='override')
    pipeline = case in ('pipeline', 'vpp')
    summary = assert_generated_comm_mode(tmp_path, mode, pipeline=pipeline)
    assert summary['output_sync_calls'] == (summary['files'] if mode else 0)
    with pytest.raises(AssertionError):
        assert_generated_comm_mode(tmp_path, not mode, pipeline=pipeline)
