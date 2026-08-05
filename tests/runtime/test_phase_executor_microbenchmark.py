#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Opt-in CPU microbenchmark for generic versus slot-based phase execution.

Run with:

    NN_SCALER_RUN_PHASE_MICROBENCH=1 pytest -s -q \
      tests/runtime/test_phase_executor_microbenchmark.py

It intentionally measures only Python/metadata/autograd dispatch overhead; it
is not a replacement for the synchronized distributed GPU benchmark.
"""
import os

import pytest
import torch
import torch.utils.benchmark

from nnscaler.runtime.executor import Executor, PhaseExecutor


RUN_BENCHMARK = os.environ.get('NN_SCALER_RUN_PHASE_MICROBENCH') == '1'


def _generic_step(module, input_data, grad):
    module.zero_grad(set_to_none=True)
    value = input_data.detach().clone().requires_grad_()
    output = Executor.fexecute('phase-bench', module, value)
    Executor.backward('phase-bench', [value], [output], [grad])
    Executor.clear()


def _phase_step(module, phase_executor, input_data, grad):
    module.zero_grad(set_to_none=True)
    value = input_data.detach().clone().requires_grad_()
    output = phase_executor.forward(0, module, value)
    phase_executor.backward(0, [output], [grad])
    phase_executor.check_clear()


@pytest.mark.skipif(not RUN_BENCHMARK, reason='set NN_SCALER_RUN_PHASE_MICROBENCH=1')
def test_phase_executor_microbenchmark():
    torch.manual_seed(2305)
    module = torch.nn.Linear(32, 32)
    input_data = torch.randn(8, 32)
    grad = torch.randn(8, 32)
    phase_executor = PhaseExecutor(1)

    benchmark_globals = {
        '_generic_step': _generic_step,
        '_phase_step': _phase_step,
        'module': module,
        'phase_executor': phase_executor,
        'input_data': input_data,
        'grad': grad,
    }
    generic = torch.utils.benchmark.Timer(
        stmt='_generic_step(module, input_data, grad)',
        globals=benchmark_globals,
    ).blocked_autorange(min_run_time=0.5)
    phase = torch.utils.benchmark.Timer(
        stmt='_phase_step(module, phase_executor, input_data, grad)',
        globals=benchmark_globals,
    ).blocked_autorange(min_run_time=0.5)
    print({
        'generic_median_us': generic.median * 1e6,
        'phase_median_us': phase.median * 1e6,
        'speedup': generic.median / phase.median,
    })
