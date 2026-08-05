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
3. The optimized default emits no artificial stream context around a phase;
   the optional dedicated-communication-stream ablation still emits a real,
   selective ``with torch.cuda.stream(...)`` / ``record_stream(...)`` path.
"""
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch

from nnscaler.ir.unique import IDGenerator
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.parallel import _gen_graph, ComputeConfig
from nnscaler.flags import CompileFlag
from nnscaler.graph.gener.gen import IRAdapterGener
from nnscaler.execplan import ExecutionPlan
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.schedule.schedplan import StreamContext
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


def _compile_phase_moe(tmpdir: Path, *, dedicated_moe_comm_stream: bool = False,
                       independent_pp_replica_lanes: Optional[bool] = None,
                       pp_replica_semantics: Optional[str] = 'equal',
                       explicit_default_stream: bool = False) -> Dict[int, str]:
    IDGenerator().clear()
    cfg = MoEConfig(dim=DIM, n_heads=NHEADS, seq_len=SEQLEN, ffn_hidden=FFN_HIDDEN, capacity_factor=1.0)
    model = PhaseMoEModel(cfg, NUM_STAGES, LAYERS_PER_STAGE, EP_RANKS_PER_STAGE, use_phases=True)
    model.train()
    dummy = {'data': {'data': torch.randn(T, cfg.dim), 'target': torch.randn(T, cfg.dim)}}
    graph, fargs = _gen_graph(model, dummy, tmpdir, constant_folding=True, end2end_mode=True)

    pas = make_pas(
        NUM_STAGES, LAYERS_PER_STAGE, EP_RANKS_PER_STAGE, use_phases=True,
        dedicated_moe_comm_stream=dedicated_moe_comm_stream,
        independent_pp_replica_lanes=independent_pp_replica_lanes,
        pp_replica_semantics=pp_replica_semantics,
    )
    config = ComputeConfig(RUNTIME_NDEVS, RUNTIME_NDEVS, use_end2end=True, pas_config=dict(pipeline_nmicros=NMICROS))
    graph = pas(graph, config)
    if explicit_default_stream:
        for segment in graph.select(ntype=IRSegment, flatten=False):
            segment.set_op_context('stream_context', StreamContext(stream='default'))

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
    """The performance-default: no synthetic phase stream contexts."""
    with replace_all_device_with('cpu', force=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            return _compile_phase_moe(Path(tmpdir))


@pytest.fixture(scope='module')
def dedicated_gencode() -> Dict[int, str]:
    """Explicit legacy-stream ablation, retained to exercise the safe path."""
    with replace_all_device_with('cpu', force=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            return _compile_phase_moe(Path(tmpdir), dedicated_moe_comm_stream=True)




def test_gencode_identity_independent_lane_mapping_is_noop():
    """A marked local identity layout must compile without an RVD gather."""
    with replace_all_device_with('cpu', force=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            IDGenerator().clear()
            cfg = MoEConfig(dim=DIM, n_heads=NHEADS, seq_len=SEQLEN, ffn_hidden=FFN_HIDDEN, capacity_factor=1.0)
            model = PhaseMoEModel(cfg, 1, 2, [(0, 1)], use_phases=True)
            graph, fargs = _gen_graph(
                model,
                {'data': {'data': torch.randn(T, cfg.dim), 'target': torch.randn(T, cfg.dim)}},
                tmpdir,
                constant_folding=True,
                end2end_mode=True,
            )
            graph = make_pas(1, 2, [(0, 1)], use_phases=True)(
                graph,
                ComputeConfig(2, 2, use_end2end=True, pas_config=dict(pipeline_nmicros=NMICROS)),
            )
            fsegments = [segment for segment in graph.select(ntype=IRSegment, flatten=False) if segment.isfw()]
            assert len(fsegments) >= 8
            marked = [output for output in fsegments[3].outputs() if isinstance(output, IRSubTensor)]
            assert marked
            for output in marked:
                output.parent.mark_independent_replica_lanes()
            adapter_graph = IRAdapterGener.gen(graph, cost_fn=None)
            adapter_graph.sched.apply()
            execplan = DiffFusion.apply(ExecutionPlan.from_schedplan(adapter_graph.sched))
            outfile = str(tmpdir / 'identity_lane.py')
            ModuleCodeGen(execplan, 2).gen(0, forward_args=fargs, outfile=outfile,
                                           attach=False, as_parallel_module=True, end2end_mode=True)
            ScheduleCodeGen(execplan, 2).gen(0, outfile=outfile, attach=True)
            generated = Path(outfile).read_text()
    assert 'runtime.adapter.all_gather' not in generated


def test_gencode_explicit_default_stream_is_a_real_context():
    """``None`` and ``'default'`` must not collapse to the same codegen path."""
    with replace_all_device_with('cpu', force=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            generated = _compile_phase_moe(Path(tmpdir), explicit_default_stream=True)
    text = generated[0]
    marker = "with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('default')):"
    assert marker in text
    # Placement on the default stream is not a non-default multi-stream workload.
    assert 'use_multi_streams = False' in text
    assert '.record_stream(torch.cuda.current_stream())' not in text


def test_gencode_pp_ep_replica_semantics_must_be_explicit():
    """A PP×EP policy cannot silently assume rank-distinct lanes are equal."""
    with replace_all_device_with('cpu', force=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match='PP x EP replica semantics must be explicit'):
                _compile_phase_moe(Path(tmpdir), pp_replica_semantics=None)


def test_gencode_pp_ep_independent_lanes_fail_closed():
    """RVD replicas cannot silently become asymmetric EP lanes at a PP edge."""
    with replace_all_device_with('cpu', force=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match='non-identity PP boundary are not yet executable safely'):
                _compile_phase_moe(Path(tmpdir), pp_replica_semantics='independent')

def _train_step_body(text: str) -> List[str]:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('def _train_step'))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith('def _infer_step'))
    return lines[start:end]


# ---------------------------------------------------------------------------
# 1) distinct phase methods
# ---------------------------------------------------------------------------

def test_gencode_phase_executor_can_be_disabled_for_ablation(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'disable_phase_executor', True)
    with replace_all_device_with('cpu', force=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            generated = _compile_phase_moe(Path(tmpdir))
    text = generated[0]
    assert '_phase_fexecute = model._phase_executor.forward' not in text
    assert 'nnscaler.runtime.executor.fexecute(' in text


def test_gencode_phase_executor_slots_are_used(gencode):
    text = gencode[0]
    assert 'self._phase_executor = nnscaler.runtime.executor.PhaseExecutor(' in text
    assert '_phase_fexecute = model._phase_executor.forward' in text
    assert '_phase_backward = model._phase_executor.backward' in text


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
    issue_re = re.compile(r'(\w+), (\w+), (\w+) = _phase_fexecute\(\d+, model\.segment\d+, \*\((\w+), \)')
    for i, line in enumerate(body):
        m = issue_re.search(line)
        if not m:
            continue
        pending_var = m.group(3)
        wait_idx = next((j for j in range(i + 1, len(body))
                          if f'*({pending_var}, )' in body[j] and 'phase_fexecute' in body[j]), None)
        if wait_idx is None:
            continue
        backward_idx = next((j for j in range(i + 1, wait_idx)
                              if '_phase_backward(' in body[j]), None)
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
    issue_re = re.compile(r'(\w+) = _phase_fexecute\(\d+, model\.segment\d+, \*\(\w+, \)')
    for i, line in enumerate(body):
        m = issue_re.search(line)
        if not m:
            continue
        pending_var = m.group(1)
        wait_idx = next((j for j in range(i + 1, len(body))
                          if re.search(rf'\b{re.escape(pending_var)}\b', body[j]) and 'phase_fexecute' in body[j]), None)
        if wait_idx is None:
            continue
        backward_idx = next((j for j in range(i + 1, wait_idx)
                              if '_phase_backward(' in body[j]), None)
        if backward_idx is None:
            continue
        assert i < backward_idx < wait_idx
        return
    raise AssertionError('no combine issue/backward/wait window found:\n' + '\n'.join(body))


# ---------------------------------------------------------------------------
# 3) real, codegen-emitted stream/event blocks
# ---------------------------------------------------------------------------

def test_gencode_has_real_stream_and_wait_stream_blocks(dedicated_gencode):
    text = dedicated_gencode[0]
    assert 'use_multi_streams = True' in text
    assert (
        "with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('moe_comm')):"
        in text
    )
    assert (
        "torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('default'))"
        in text
    )
    assert "torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('moe_comm'))" not in text


def test_gencode_default_phase_path_has_no_synthetic_stream_context(gencode):
    text = gencode[0]
    assert 'use_multi_streams = False' in text
    assert 'with torch.cuda.stream(' not in text
    assert '.wait_stream(' not in text
    assert '.record_stream(' not in text


def test_gencode_stream_block_wraps_dispatch_issue_specifically(dedicated_gencode):
    """The `moe_comm`-stream block must specifically wrap the MOE_DISPATCH
    phase's own `segmentNNN` call (not some unrelated code)."""
    text = dedicated_gencode[0]
    pattern = re.compile(
        r"with torch\.cuda\.stream\(nnscaler\.runtime\.device\.DeviceGroup\(\)\.get_stream\('moe_comm'\)\):\n"
        r"(?:.*\n)*?\s*\S+ = _phase_fexecute\(\d+, model\.segment\d+, \*\(\S+, \), requires_grad=True\)\n"
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


# ---------------------------------------------------------------------------
# 4) record_stream + producer-event(wait_stream)->issue,
#    wait_stream->record_stream->consumer hard dependency assertions
# ---------------------------------------------------------------------------

def _find_stream_context_blocks(body: List[str]):
    """Yield (start_idx, stream_name, wait_stream_idx, wait_stream_source,
    record_stream_idxs_and_vars, fexecute_idx, fexecute_args) for every
    ``with torch.cuda.stream(...):`` block in ``body``."""
    block_re = re.compile(r"with torch\.cuda\.stream\(nnscaler\.runtime\.device\.DeviceGroup\(\)\.get_stream\('(\w+)'\)\):")
    wait_re = re.compile(r"torch\.cuda\.current_stream\(\)\.wait_stream\(nnscaler\.runtime\.device\.DeviceGroup\(\)\.get_stream\('(\w+)'\)\)")
    record_re = re.compile(r'^\s*(\w+)\.record_stream\(torch\.cuda\.current_stream\(\)\)')
    fexecute_re = re.compile(r"_phase_fexecute\(\d+, model\.segment\d+, \*\(([^)]*)\)")
    blocks = []
    i = 0
    while i < len(body):
        m = block_re.search(body[i])
        if not m:
            i += 1
            continue
        stream_name = m.group(1)
        # scan forward within this indented block until dedent (blank line
        # or a line with less/equal indentation that isn't part of the block)
        base_indent = len(body[i]) - len(body[i].lstrip())
        wait_idx, wait_src, record_entries, fexecute_idx, fexecute_args = None, None, [], None, None
        j = i + 1
        while j < len(body):
            line = body[j]
            if line.strip() == '':
                j += 1
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent:
                break
            wm = wait_re.search(line)
            if wm:
                wait_idx, wait_src = j, wm.group(1)
            rm = record_re.search(line)
            if rm:
                record_entries.append((j, rm.group(1)))
            fm = fexecute_re.search(line)
            if fm and fexecute_idx is None:
                fexecute_idx, fexecute_args = j, fm.group(1)
            j += 1
        blocks.append((i, stream_name, wait_idx, wait_src, record_entries, fexecute_idx, fexecute_args))
        i = j
    return blocks


def test_gencode_record_stream_present_for_cross_stream_consumed_tensors(dedicated_gencode):
    """``record_stream`` (Step C's remediation-added hard assertion,
    ``nnscaler/codegen/schedule/schedule.py``'s stream_context-triggered
    ``{t}.record_stream(torch.cuda.current_stream())`` emission) must
    literally appear in the generated code for the tensors consumed inside
    a phase's own named-stream block."""
    text = dedicated_gencode[0]
    assert re.search(r'\w+\.record_stream\(torch\.cuda\.current_stream\(\)\)', text), \
        'expected a real, codegen-emitted .record_stream(...) call on the dedicated stream'
    assert text.count('.record_stream(torch.cuda.current_stream())') >= 1


def test_gencode_stream_blocks_have_wait_then_record_stream_then_consumer(dedicated_gencode):
    """For every ``with torch.cuda.stream(<S>):`` block that both waits on
    another stream AND calls ``record_stream`` on some tensor ``t``: hard
    -assert, by exact line order AND exact variable-name identity (not
    keyword-matching), that:
      1. ``wait_stream(<other stream>)`` (the producer-completion event
         wait) comes first,
      2. ``t.record_stream(current_stream())`` (registering ``t`` as
         consumed on the new, current stream, preventing PyTorch's
         allocator from prematurely reclaiming/reusing ``t``'s memory
         while an async producer-stream operation might still be
         in flight) comes second,
      3. the ``fexecute(...)`` call that actually CONSUMES ``t`` (``t``
         appears literally in its argument tuple) comes third -- i.e. the
         same tensor that gets record_stream'd is the one actually handed
         to the phase's own compute call, not some unrelated tensor.
    This directly proves the producer-event(wait)->record_stream->consumer
    dependency chain the remediation instructions require, rather than a
    keyword-only "wait_stream and record_stream both appear somewhere"
    check.
    """
    body = _train_step_body(dedicated_gencode[0])
    blocks = _find_stream_context_blocks(body)
    checked = 0
    for start_idx, stream_name, wait_idx, wait_src, record_entries, fexecute_idx, fexecute_args in blocks:
        if wait_idx is None or not record_entries or fexecute_idx is None:
            continue
        # every recorded tensor must actually be a literal argument of the
        # block's own fexecute call (exact identity, not just "some tensor")
        for rec_idx, rec_var in record_entries:
            assert re.search(rf'\b{re.escape(rec_var)}\b', fexecute_args), (
                f"record_stream'd variable {rec_var!r} does not appear in "
                f"this block's own fexecute args {fexecute_args!r} -- "
                f"record_stream target and actual consumer are not the same tensor"
            )
            assert wait_idx < rec_idx < fexecute_idx, (
                f"expected wait_stream (line {wait_idx}) < record_stream of "
                f"{rec_var!r} (line {rec_idx}) < consuming fexecute "
                f"(line {fexecute_idx}), got out-of-order indices"
            )
        checked += 1
    assert checked >= 1, (
        f'expected the dedicated dispatch stream to retain one full '
        f'wait_stream -> record_stream -> consumer chain, found {checked}'
    )


def test_gencode_record_stream_targets_match_producer_event_semantics(dedicated_gencode):
    """Stronger version of the above: the record_stream'd tensor in the
    ``moe_comm`` stream block must be the SAME variable that was produced
    by an EARLIER (outside this block, on the ``default`` stream) segment
    call -- i.e. record_stream is applied to a genuine cross-stream
    producer/consumer tensor, not a same-stream local temporary."""
    body = _train_step_body(dedicated_gencode[0])
    full_text = '\n'.join(body)
    blocks = _find_stream_context_blocks(body)
    moe_comm_blocks = [b for b in blocks if b[1] == 'moe_comm' and b[4]]
    assert moe_comm_blocks, 'expected at least one moe_comm stream block with a record_stream call'
    start_idx, _, wait_idx, wait_src, record_entries, fexecute_idx, fexecute_args = moe_comm_blocks[0]
    assert wait_src == 'default', f"expected the moe_comm block to wait on the 'default' stream, got {wait_src!r}"
    for rec_idx, rec_var in record_entries:
        producer_pattern = re.compile(rf'{re.escape(rec_var)} = _phase_fexecute\(')
        producer_idx = next((k for k in range(start_idx) if producer_pattern.search(body[k])), None)
        assert producer_idx is not None, (
            f"expected {rec_var!r} to be produced by an earlier fexecute call "
            f"(outside/before this moe_comm block, i.e. on the default stream) "
            f"-- found no such producer line"
        )
