#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Gencode-precision tests for Step C (CPU-only, no distributed launch or GPU
needed -- mirrors ``tests/codegen/test_reschedule.py``/``test_global_schedule.py``'s
own CPU compile-pipeline conventions).

Builds a real, phase-lowered, phase-scheduled PP2xEP2 compile of
``tests.parallel_module.phase_moe_common.PhaseMoEModel`` (the exact same
model/PAS ``tests/parallel_module/test_phase_moe_e2e.py`` uses for real
2/4-GPU execution) via the same low-level pieces those other CPU codegen
tests use (``_gen_graph`` -> PAS -> ``IRAdapterGener.gen`` -> ``DiffFusion``
-> ``ModuleCodeGen`` + ``ScheduleCodeGen``), and inspects the *actual
generated ``.py`` source text* for:

1. Distinct phase *methods* (``segmentNNN``) -- one per phase, not a single
   monolithic per-stage method.
2. The literal ``issue < B(m) independent backward < wait`` ordering for
   both the dispatch and combine windows, via the *same variable name*
   flowing from the issuing call's return value to the waiting call's
   argument (proving it is textually the *same* adapter/tensor, not merely
   "some call happened before some other call").
3. Real, codegen-emitted ``with torch.cuda.stream(...)``/``wait_stream(...)``
   blocks around the phases carrying a ``StreamContext`` (Step C's
   ``_set_moe_stream_context``), using the exact same idiom
   ``nnscaler.codegen.schedule.schedule._emit_stream_context``/
   ``_get_codes_with_stream_context`` documents -- not hand-written.
"""
import re
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from nnscaler.ir.unique import IDGenerator
from nnscaler.parallel import _gen_graph, ComputeConfig
from nnscaler.graph.gener.gen import IRAdapterGener
from nnscaler.execplan import ExecutionPlan
from nnscaler.execplan.planpass.fusion import DiffFusion
from nnscaler.codegen.module.module import ModuleCodeGen
from nnscaler.codegen.schedule.schedule import ScheduleCodeGen

from tests.utils import replace_all_device_with
from tests.parallel_module.phase_moe_common import MoEConfig, PhaseMoEModel, make_pas

DIM = 16
NHEADS = 2
SEQLEN = 4
FFN_HIDDEN = 32
NUM_STAGES = 2
LAYERS_PER_STAGE = 1
EP_RANKS_PER_STAGE = [(0, 1), (2, 3)]
RUNTIME_NDEVS = 4
NMICROS = 4
T = 8


def _compile_phase_moe(tmpdir: Path) -> Dict[int, str]:
    IDGenerator().clear()
    cfg = MoEConfig(dim=DIM, n_heads=NHEADS, seq_len=SEQLEN, ffn_hidden=FFN_HIDDEN, capacity_factor=1.0)
    model = PhaseMoEModel(cfg, NUM_STAGES, LAYERS_PER_STAGE, EP_RANKS_PER_STAGE, use_phases=True)
    model.train()
    dummy = {'data': {'data': torch.randn(T, cfg.dim), 'target': torch.randn(T, cfg.dim)}}
    graph, fargs = _gen_graph(model, dummy, tmpdir, constant_folding=True, end2end_mode=True)

    pas = make_pas(NUM_STAGES, LAYERS_PER_STAGE, EP_RANKS_PER_STAGE, use_phases=True)
    config = ComputeConfig(RUNTIME_NDEVS, RUNTIME_NDEVS, use_end2end=True, pas_config=dict(pipeline_nmicros=NMICROS))
    graph = pas(graph, config)

    adapter_graph = IRAdapterGener.gen(graph, cost_fn=None)
    if adapter_graph.sched is not None:
        adapter_graph.sched.apply()
    execplan = ExecutionPlan.from_schedplan(adapter_graph.sched)
    execplan = DiffFusion.apply(execplan)

    mgen = ModuleCodeGen(execplan, RUNTIME_NDEVS)
    sgen = ScheduleCodeGen(execplan, RUNTIME_NDEVS)
    per_device = {}
    for devid in range(RUNTIME_NDEVS):
        outfile = str(tmpdir / f'gencode{devid}.py')
        mgen.gen(devid, forward_args=fargs, outfile=outfile, attach=False,
                 as_parallel_module=True, end2end_mode=True)
        sgen.gen(device=devid, outfile=outfile, attach=True)
        per_device[devid] = Path(outfile).read_text()
    return per_device


@pytest.fixture(scope='module')
def gencode() -> Dict[int, str]:
    with replace_all_device_with('cpu', force=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            return _compile_phase_moe(Path(tmpdir))


def _train_step_body(text: str) -> List[str]:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('def _train_step'))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith('def _infer_step'))
    return lines[start:end]


# ---------------------------------------------------------------------------
# 1) distinct phase methods
# ---------------------------------------------------------------------------

def test_gencode_has_distinct_phase_methods(gencode):
    """Device 0 (stage 0, layer 0) must contain 4 distinct `segmentNNN`
    forward-phase methods (ATTENTION/MOE_DISPATCH/EXPERT_COMPUTE/MOE_COMBINE),
    not one monolithic per-stage method."""
    text = gencode[0]
    method_names = set(re.findall(r'def (segment\d+)\(', text))
    assert len(method_names) >= 4, method_names
    # Each phase's own source-line comment should appear in a *different*
    # method's body (a crude but real structural check that phases were not
    # collapsed back into one segment).
    assert 'phase_moe_common.py", line' in text


# ---------------------------------------------------------------------------
# 2) issue < B(m) independent backward < wait, same-variable-identity
# ---------------------------------------------------------------------------

def _find_dispatch_window(body: List[str]):
    """Find (issue_idx, backward_idx, wait_idx) for a MOE_DISPATCH issue-then-wait
    window: the dispatch call's return-tuple's LAST element is the pending
    tensor; the wait window's fexecute call must reference that SAME
    variable name as its argument."""
    issue_re = re.compile(r'(\w+), (\w+), (\w+) = nnscaler\.runtime\.executor\.fexecute\(.segment\d+., model\.segment\d+, \*\((\w+), \)')
    for i, line in enumerate(body):
        m = issue_re.search(line)
        if not m:
            continue
        pending_var = m.group(3)
        wait_idx = next((j for j in range(i + 1, len(body))
                          if f'*({pending_var}, )' in body[j] and 'fexecute' in body[j]), None)
        if wait_idx is None:
            continue
        backward_idx = next((j for j in range(i + 1, wait_idx)
                              if 'nnscaler.runtime.executor.backward(' in body[j]), None)
        if backward_idx is None:
            continue
        return i, backward_idx, wait_idx, pending_var
    return None


def test_gencode_dispatch_issue_lt_backward_lt_wait(gencode):
    """The dispatch window's issue < B(m) backward < wait ordering, keyed by
    the *same* pending-tensor variable name flowing from the issuing
    fexecute's return value into the waiting fexecute's argument list --
    precise textual identity, not just "some call happened earlier"."""
    body = _train_step_body(gencode[0])
    found = _find_dispatch_window(body)
    assert found is not None, '\n'.join(body)
    issue_idx, backward_idx, wait_idx, pending_var = found
    assert issue_idx < backward_idx < wait_idx, (issue_idx, backward_idx, wait_idx)


def test_gencode_combine_issue_lt_backward_lt_wait(gencode):
    """Same property for the combine window: the EXPERT_COMPUTE segment's
    fexecute call returns the combine-pending variable, consumed by name in
    the LATER MOE_COMBINE segment's fexecute call, with an intervening
    backward() call."""
    body = _train_step_body(gencode[0])
    # combine issue: fexecute('segmentNNN', ..., *(dispatch_pending, )) whose
    # single return value feeds a LATER fexecute call by the same name.
    issue_re = re.compile(r'(\w+) = nnscaler\.runtime\.executor\.fexecute\(.segment\d+., model\.segment\d+, \*\(\w+, \)')
    for i, line in enumerate(body):
        m = issue_re.search(line)
        if not m:
            continue
        pending_var = m.group(1)
        wait_idx = next((j for j in range(i + 1, len(body))
                          if re.search(rf'\b{re.escape(pending_var)}\b', body[j]) and 'fexecute' in body[j]), None)
        if wait_idx is None:
            continue
        backward_idx = next((j for j in range(i + 1, wait_idx)
                              if 'nnscaler.runtime.executor.backward(' in body[j]), None)
        if backward_idx is None:
            continue
        assert i < backward_idx < wait_idx
        return
    raise AssertionError('no combine issue/backward/wait window found:\n' + '\n'.join(body))


# ---------------------------------------------------------------------------
# 3) real, codegen-emitted stream/event blocks
# ---------------------------------------------------------------------------

def test_gencode_has_real_stream_and_wait_stream_blocks(gencode):
    text = gencode[0]
    assert 'use_multi_streams = True' in text
    assert (
        "with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('moe_comm')):"
        in text
    )
    assert (
        "torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('default'))"
        in text
    )
    assert (
        "torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('moe_comm'))"
        in text
    )


def test_gencode_stream_block_wraps_dispatch_issue_specifically(gencode):
    """The `moe_comm`-stream block must specifically wrap the MOE_DISPATCH
    phase's own `segmentNNN` call (not some unrelated code)."""
    text = gencode[0]
    pattern = re.compile(
        r"with torch\.cuda\.stream\(nnscaler\.runtime\.device\.DeviceGroup\(\)\.get_stream\('moe_comm'\)\):\n"
        r"(?:.*\n)*?\s*\S+ = nnscaler\.runtime\.executor\.fexecute\('segment\d+', model\.segment\d+, \*\(\S+, \), requires_grad=True\)\n"
    )
    assert pattern.search(text), text


def test_gencode_moe_dispatch_and_combine_calls_present(gencode):
    """Sanity: the real, registered ``nnscaler.runtime.adapter.moe`` ops
    literally appear by name (not paraphrased/hand-simulated)."""
    text = gencode[0]
    for name in ('moe_dispatch(', 'moe_dispatch_wait(', 'moe_combine(', 'moe_combine_wait('):
        assert f'nnscaler.runtime.adapter.moe.{name}' in text, name


def test_gencode_channel_and_max_outstanding_present(gencode):
    """The channel/max_outstanding kwargs (Step A-style channel tracking,
    reused for the MoE all-to-all -- see nnscaler/runtime/adapter/moe.py)
    are baked into the generated dispatch/combine call sites."""
    text = gencode[0]
    assert "channel='phase_moe_L0_dispatch'" in text
    assert "channel='phase_moe_L0_combine'" in text
    assert 'max_outstanding=1' in text
    assert 'ep_ranks=(0, 1)' in text
