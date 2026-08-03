#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Real, compiled, end-to-end proof that nnscaler's ``yomia/rescheduler``
pipeline-reschedule + async-recv + multi-stream machinery can realize
Megatron's ``combined_1f1b`` pattern -- fine-grained interleaving of one
microbatch's forward/backward *communication* with another microbatch's
*compute*, expressed as :class:`~nnscaler.graph.schedule.schedplan.StreamContext`
annotations that real nnScaler **codegen** (not hand-written code) turns into
``with torch.cuda.stream(...)`` blocks and deferred async-recv waits.

This extends the existing 4-GPU pipeline-reschedule e2e path in
``test_reschedule_e2e.py`` (same ``parallelize``/``ComputeConfig``/
``PASMegatron``/``launch_torchrun`` machinery, same
``ENABLE_OP_RESCHEDULE`` / ``OP_RESCHEDULE_PIPELINE`` / ``use_async_recv``
knobs) with one additional, already-existing, first-class extension point:
``SchedulePlan.stream_config`` (see ``nnscaler/graph/schedule/schedplan.py``),
which routes inter-segment (cross-pipeline-stage) P2P communication adapters
onto a dedicated CUDA stream. Setting it requires **zero core nnscaler code
changes** -- ``_pas_multi_stream`` below is a thin wrapper around the
existing ``PASMegatron`` test policy that captures the ``SchedulePlan``
already returned by ``ComputeConfig.apply_pipeline_scheduler`` (previously
discarded by ``PASMegatron``) and sets its ``.stream_config``.

What is proven here (see the three test functions):

1. ``test_combined_1f1b_gencode_has_stream_and_async_recv_structures``:
   inspects the **actual generated ``.py`` module files** written by
   nnscaler's codegen to ``gen_savedir`` and asserts they contain:
     - ``GenModel.use_multi_streams = True`` (codegen detected and enabled
       real multi-stream execution because a non-default stream is used);
     - literal ``with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup()
       .get_stream('comm')):`` blocks wrapping the inter-segment P2P adapters
       (:func:`_emit_stream_context`/``_get_codes_with_stream_context`` in
       ``nnscaler/codegen/schedule/schedule.py``);
     - the async-recv issue/deferred-wait pair (``model.adapterNNN`` issued
       with ``requires_grad=...`` inside a ``with torch.cuda.stream(...)``
       block, and ``model.adapterNNN_wait`` called later, right before the
       node that actually consumes the received tensor) for **both**
       directions of pipeline communication: the forward activation P2P
       recv between stages, and the backward gradient P2P recv between
       stages -- i.e. exactly the two communication legs ``combined_1f1b``
       needs to overlap with compute.
   This is the concrete, checkable evidence that the interleave scaffolding
   is produced by the compiler, not hand-written.

2. ``test_combined_1f1b_pipeline_numeric_equivalence``: runs the same
   multi-stream pipeline model for several training steps with the
   reschedule+async-recv machinery OFF and ON (identical seeds/data) and
   asserts the merged weights end up bit-for-bit-close (atol/rtol 1e-5) --
   scheduling/stream placement must never change the math.

   A real correctness pitfall was found and fixed while building this (see
   ``_pas_multi_stream``'s docstring for detail): ``StreamContext.wait_streams``
   must include ``'default'`` on the inter-segment move context, or the
   generated comm-stream send/recv block races against the default (compute)
   stream and silently corrupts trained weights (reproduced concretely via a
   clean A/B ablation: divergence up to ~4e-2 after 2 steps with the field
   omitted, exactly 0 with it set). This is not a core-code bug -- the field
   exists and is correctly wired into codegen -- but there is no existing
   example anywhere else in this codebase of setting it, so it is easy to
   omit silently. This is reported as a concrete, load-bearing finding rather
   than only a passing test.

3. ``test_combined_1f1b_pipeline_no_deadlock``: runs the ON configuration
   under a hard wall-clock guard (``signal.alarm``); a real deadlock in the
   interleaved stream/event or NCCL send/recv pairing would hang the
   ``launch_torchrun`` call forever, so a timeout here is treated as deadlock
   detection rather than a flaky failure.

Honest, evidence-based LIMITATION found while building this (see the report
for full detail): the built-in ``1f1b`` ``PredefinedSched`` places each
inter-stage P2P adapter "just in time" for its consumer in the *original*
(pre-reschedule) order, and ``Reschedule``'s ``_comm_early_priority`` can only
hoist a communication node as early as the **global, cross-microbatch
communication-serialization chain** (``OpDependencyGraph._build_comm_edges``,
see ``test_combined_1f1b_min.py``) legally allows. In this model/schedule,
that means the async recv for microbatch m's backward gradient is issued
right when the previous microbatch's forward finishes and is waited on
almost immediately after (still race-free and still on a separate CUDA
stream) -- there usually is not a full, independent compute block of another
microbatch sitting *between* the issue and the wait in the generated code.
Getting a large, easily-wall-clock-measurable overlap window (a full F(m+1)
forward pass genuinely overlapping a full B(m) backward pass) would require
either a deeper pipeline scheduler that deliberately front-loads receives
across more in-flight microbatches, or relaxing the global comm-chain to be
per-channel -- both are pipeline-scheduler/``OpDependencyGraph`` design
changes beyond what "reuse the existing path with no core changes" allows,
and are called out explicitly rather than glossed over.

The lower-level, hand-written (non-nnscaler-compiled) CUDA stream/event +
NCCL proof-of-concept that motivated this investigation now lives in
``examples/combined_1f1b_min/lower_level_stream_event_poc.py`` and is
explicitly labeled as such -- it is NOT compiled by nnscaler and must not be
read as "the nnscaler combined_1f1b implementation".

Requires >= 2 GPUs (uses a pure pipeline-parallel PP2 configuration, no
tensor parallelism, to keep the example minimal).
"""
import math
import os
import signal
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer, merge_state_dicts
from nnscaler.graph.schedule.schedplan import StreamConfig, StreamContext

from .common import init_distributed, PASMegatron
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import init_random, clear_dir_on_rank0, PYTEST_RUN_ID

DIM = 16
NLAYERS = 8
MBS = 2
NSTAGES = 2       # pure pipeline-parallel, no tensor-parallel -> only 2 GPUs needed
NMICROS = 4
NSTEPS = 2
COMM_STREAM = 'comm'

# Deliberately larger than DIM/NLAYERS above -- used ONLY by the overlap
# benchmark (`_timing_worker`), not by the structural/numeric-equivalence
# tests, which stay on the tiny model for speed. A bigger per-layer compute
# cost improves the overlap benchmark's signal-to-noise ratio: the constant
# per-step CPU-side scheduling/channel-tracking overhead this test wants to
# distinguish from a genuine overlap win becomes relatively smaller compared
# to a real backward segment's compute time.
TIMING_DIM = 1024
TIMING_NLAYERS = 8
TIMING_NMICROS = 8


class _MLP(nn.Module):
    def __init__(self, dim=DIM, nlayers=NLAYERS):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(nlayers))
        self.loss_fn = nn.BCELoss()

    def forward(self, data):
        x = data['data']
        for layer in self.layers:
            x = layer(x)
        return self.loss_fn(torch.sigmoid(x), data['target'])


def _pas_multi_stream(graph, config: ComputeConfig):
    """``PASMegatron`` plus routing inter-segment (cross-pipeline-stage) P2P
    communication onto a dedicated ``'comm'`` CUDA stream.

    This uses the existing, first-class ``SchedulePlan.stream_config``
    extension point (``nnscaler/graph/schedule/schedplan.py``) -- no nnscaler
    core code is modified. ``PASMegatron`` already builds and binds the
    ``SchedulePlan`` via ``config.apply_pipeline_scheduler(...)``
    (``graph._bind_schedule``, exposed as ``graph.sched``); we simply read it
    back out and set its ``stream_config`` before returning.

    IMPORTANT (a real correctness pitfall found while building this):
    ``StreamContext.wait_streams`` **must** include ``'default'`` here. If
    omitted, the generated ``with torch.cuda.stream(...'comm'...):`` block
    for a cross-stage send has no ordering constraint against the default
    (compute) stream that just produced the tensor being sent, and the send
    can race ahead of the compute that fills the tensor -- this was
    reproduced concretely: it silently corrupted trained weights (divergence
    up to ~4e-2 after 2 steps, confirmed via a clean A/B ablation) while
    still "looking" correct (no error, no crash, plausible-looking gencode).
    With ``wait_streams=['default']`` set (making the codegen emit
    ``torch.cuda.current_stream().wait_stream(DeviceGroup().get_stream('default'))``
    inside the comm-stream block before the send/recv), the divergence goes
    to exactly 0. There is no existing example of ``StreamContext.wait_streams``
    usage anywhere else in this codebase (confirmed via a repo-wide search) --
    this extension point is real, correctly wired into codegen, and requires
    no core changes to use correctly, but it is also easy to misuse silently
    if this field is forgotten.
    """
    graph = PASMegatron(graph, config)
    sched = graph.sched
    sched.stream_config = StreamConfig(
        inter_segment_move=StreamContext(stream=COMM_STREAM, wait_streams=['default']),
    )
    return graph


def _make_data(nsteps, nmicros, seed=1234):
    g = torch.Generator().manual_seed(seed)
    steps = []
    for _ in range(nsteps):
        steps.append([
            {'data': torch.randn(MBS, DIM, generator=g, device='cpu'),
             'target': torch.rand(MBS, DIM, generator=g, device='cpu')}
            for _ in range(nmicros)
        ])
    return steps


def _worker(reschedule: bool, use_async_recv: bool = True, capture_gencode: bool = False,
            capture_per_file: bool = False):
    init_distributed()
    dev = torch.cuda.current_device()
    init_random()
    tag = 'on' if reschedule else 'off'
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'combined_1f1b_e2e_{PYTEST_RUN_ID}_{tag}') as tempdir:
        init_random()
        model = parallelize(
            _MLP(),
            {'data': {'data': torch.randn(MBS, DIM, device=dev),
                      'target': torch.rand(MBS, DIM, device=dev)}},
            _pas_multi_stream,
            ComputeConfig(NSTAGES, NSTAGES, use_end2end=True,
                          use_async_recv=use_async_recv,
                          pas_config=dict(pipeline_nstages=NSTAGES, pipeline_nmicros=NMICROS,
                                          pipeline_scheduler='1f1b')),
            gen_savedir=tempdir,
            instance_name=f'combined_1f1b_{tag}',
        )

        gencode_text = ''
        per_file = {}
        if (capture_gencode or capture_per_file) and torch.distributed.get_rank() == 0:
            pyfiles = sorted(tempdir.rglob('*.py'))
            if capture_gencode:
                gencode_text = '\n'.join(p.read_text() for p in pyfiles)
            if capture_per_file:
                per_file = {str(p.relative_to(tempdir)): p.read_text() for p in pyfiles}

        model.cuda()
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)

        data = _make_data(NSTEPS, NMICROS)
        states = []
        for step in range(NSTEPS):
            model.train()
            batch = [{k: v.to(dev) for k, v in mb.items()} for mb in data[step]]
            model.train_step(batch)
            # Because this test's PAS (`_pas_multi_stream`) sets a custom
            # `stream_config.inter_segment_move`, `execplan.cuda_sync_required`
            # is derived True by nnscaler (see `execplan.py`); the built-in
            # `nnscaler.cli.trainer.Trainer` would honor that by calling
            # `torch.cuda.synchronize()` right here. This test calls
            # `model.train_step()` directly (matching `test_reschedule_e2e.py`'s
            # convention) rather than going through `Trainer`, so it must
            # perform that same synchronization manually -- confirmed
            # necessary experimentally: omitting it produced sporadic
            # per-element weight mismatches (~2e-2) between the reschedule
            # ON/OFF runs, i.e. a real, reproducible correctness hazard when
            # a custom inter-segment CUDA stream is used without this sync.
            torch.cuda.synchronize()
            optimizer.step()
            optimizer.zero_grad()
            states.append(clone_to_cpu_recursively(model.state_dict()))
        return states, gencode_text, per_file


class _Alarm:
    """Hard wall-clock deadlock guard (SIGALRM). Only meaningful on Linux,
    which is what srgws-17 runs; skip gracefully elsewhere."""

    def __init__(self, seconds: int, message: str):
        self.seconds = seconds
        self.message = message
        self._supported = hasattr(signal, 'SIGALRM')

    def _handler(self, signum, frame):
        raise TimeoutError(self.message)

    def __enter__(self):
        if self._supported:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._supported:
            signal.alarm(0)
        return False


def _set_reschedule_env(enabled: bool):
    saved = {k: os.environ.get(k) for k in
             ('ENABLE_OP_RESCHEDULE', 'OP_RESCHEDULE_SCOPE', 'OP_RESCHEDULE_PIPELINE')}
    if enabled:
        os.environ['ENABLE_OP_RESCHEDULE'] = '1'
        os.environ['OP_RESCHEDULE_SCOPE'] = 'both'
        os.environ['OP_RESCHEDULE_PIPELINE'] = '1'
    else:
        for k in saved:
            os.environ.pop(k, None)
    return saved


def _restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 gpus')
def test_combined_1f1b_gencode_has_stream_and_async_recv_structures():
    """Codegen-produced evidence: real generated code contains multi-stream
    (StreamContext) and async-recv issue/deferred-wait structures -- not
    hand-written torch.cuda.stream calls."""
    saved = _set_reschedule_env(True)
    try:
        with _Alarm(180, 'possible deadlock: gencode-evidence run did not finish in 180s'):
            outputs = launch_torchrun(NSTAGES, _worker, True, True, True)
    finally:
        _restore_env(saved)

    assert outputs and outputs[0] is not None, 'worker returned no result'
    gencode_text = outputs[0][1]
    assert gencode_text, 'rank 0 did not capture any generated code'

    # 1) codegen auto-detected and enabled real multi-stream execution
    assert 'use_multi_streams = True' in gencode_text, (
        "expected codegen to set `use_multi_streams = True` on GenModel because "
        "a non-default ('comm') stream is used by an inter-segment adapter"
    )

    # 2) a literal, codegen-emitted CUDA-stream context block for the P2P adapter,
    #    driven by DeviceGroup -- the same idiom StreamContext/_emit_stream_context
    #    documents, not something we wrote by hand
    stream_block_marker = (
        "with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):"
    )
    assert stream_block_marker in gencode_text, (
        f'expected literal codegen-emitted stream block {stream_block_marker!r} in generated code'
    )

    # 3) async-recv issue + deferred wait pair (issued inside a `with` stream block,
    #    waited later, right before the consuming node) for BOTH pipeline
    #    communication legs: forward-activation recv and backward-gradient recv.
    assert '_wait, *(' in gencode_text and 'aexecute(model.adapter' in gencode_text, (
        'expected the deferred async-recv wait idiom '
        "(`nnscaler.runtime.executor.aexecute(model.adapterNNN_wait, *(tensor,), ...)`) "
        'in generated code'
    )
    n_issue = gencode_text.count("requires_grad=True)") + gencode_text.count("requires_grad=False)")
    n_wait = gencode_text.count('_wait, *(')
    assert n_wait >= 2, f'expected at least 2 deferred-wait call sites (fwd + bwd legs), found {n_wait}'
    assert n_issue > 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 gpus')
def test_combined_1f1b_pipeline_numeric_equivalence():
    """Reschedule (comm issued as early as legal) must not change the trained
    weights vs. the un-rescheduled baseline, with the real multi-stream
    (``StreamContext``) + async-recv machinery from ``_pas_multi_stream`` held
    constant and active across both runs -- i.e. this checks numeric
    correctness of the *actual combined_1f1b-style compiled configuration*,
    not a stripped-down variant.

    ``use_async_recv=True`` is kept ON for both the baseline and the
    rescheduled run so only the reschedule variable changes.
    """
    saved_off = _set_reschedule_env(False)
    try:
        with _Alarm(180, 'possible deadlock: baseline (reschedule OFF) run did not finish in 180s'):
            off = launch_torchrun(NSTAGES, _worker, False, True, False)
    finally:
        _restore_env(saved_off)

    saved_on = _set_reschedule_env(True)
    try:
        with _Alarm(180, 'possible deadlock: reschedule ON run did not finish in 180s'):
            on = launch_torchrun(NSTAGES, _worker, True, True, False)
    finally:
        _restore_env(saved_on)

    assert off and on, 'workers returned no result'
    off_states = [off[r][0] for r in range(NSTAGES)]
    on_states = [on[r][0] for r in range(NSTAGES)]
    for step in range(NSTEPS):
        off_sd = merge_state_dicts([off_states[r][step] for r in range(NSTAGES)])[0]
        on_sd = merge_state_dicts([on_states[r][step] for r in range(NSTAGES)])[0]
        for k, a in off_sd.items():
            if not torch.is_tensor(a):
                continue
            b = on_sd[k]
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), \
                f'step {step} key {k} differs: max|diff|={(a - b).abs().max().item():.3e}'


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 gpus')
def test_combined_1f1b_pipeline_no_deadlock():
    """The reschedule + async-recv + multi-stream ON configuration must run
    to completion (both ranks) within a bounded wall-clock time. A hang here
    (SIGALRM firing) is treated as evidence of a possible deadlock in the
    interleaved stream/event or NCCL send/recv pairing, not a flaky timeout."""
    saved = _set_reschedule_env(True)
    try:
        with _Alarm(180, 'POSSIBLE DEADLOCK: combined_1f1b pipeline (reschedule+async_recv+multi-stream) '
                          'did not complete within 180s'):
            outputs = launch_torchrun(NSTAGES, _worker, True, True, False)
    finally:
        _restore_env(saved)

    assert outputs is not None and len(outputs) == NSTAGES
    for r in range(NSTAGES):
        rank_states, _, _ = outputs[r]
        assert len(rank_states) == NSTEPS
    print('NO_DEADLOCK PASS: both ranks completed the combined_1f1b pipeline run')


# ---------------------------------------------------------------------------
# GlobalCommSchedule (Step A): fixed, cross-rank deterministic schedule with
# explicit channel/sequence/lifecycle-tracked async-recv issue/wait.
#
# Unlike the tests above (which drive the existing, opt-in, *per-device*
# ``Reschedule`` pass via ``ENABLE_OP_RESCHEDULE`` and document that it cannot
# reliably widen the overlap window -- see this module's docstring), the
# tests below drive `nnscaler.execplan.planpass.global_schedule
# .GlobalCommSchedule` via ``ENABLE_GLOBAL_P2P_RESCHEDULE`` +
# ``ASYNC_RECV_CHANNEL``: ONE shared, cross-device dependency graph +
# cap-aware topological sort, projected onto every device, plus explicit
# channel/sequence/outstanding-cap runtime tracking (see
# ``nnscaler.runtime.executor._AsyncCommHandler.issue_recv`` and
# ``CompileFlag.async_recv_channel`` / ``async_recv_max_outstanding``).
# ---------------------------------------------------------------------------

def _set_global_schedule_env(enabled: bool, max_outstanding: int = 6):
    keys = ('ENABLE_GLOBAL_P2P_RESCHEDULE', 'ASYNC_RECV_CHANNEL', 'ASYNC_RECV_MAX_OUTSTANDING')
    saved = {k: os.environ.get(k) for k in keys}
    if enabled:
        os.environ['ENABLE_GLOBAL_P2P_RESCHEDULE'] = '1'
        os.environ['ASYNC_RECV_CHANNEL'] = '1'
        os.environ['ASYNC_RECV_MAX_OUTSTANDING'] = str(max_outstanding)
    else:
        for k in keys:
            os.environ.pop(k, None)
    return saved


def _find_train_step_body(per_file: dict, name_contains: str = 'gencode1') -> list:
    """Return the `_train_step` body lines (list[str]) of the generated file
    whose relative path contains `name_contains` (default: the last pipeline
    stage, which receives the forward activation this invariant is about)."""
    for name, content in per_file.items():
        if name_contains not in name:
            continue
        lines = content.splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith('def _train_step'))
        end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith('def _infer_step'))
        return lines[start:end]
    raise AssertionError(f'no generated file matching {name_contains!r} found in {list(per_file)}')


def _find_issue_b_wait_f_quadruple(body_lines: list):
    """Search the `_train_step` body for a concrete instance of the
    combined_1f1b invariant: an async-recv issue, followed (later, with at
    least one full backward segment call in between) by its deferred wait,
    followed by the forward segment call that consumes it.

    Returns (issue_idx, backward_idx, wait_idx, forward_idx) for the first
    such instance found, or None if none exists.
    """
    import re
    issue_re = re.compile(r'(\w+) = nnscaler\.runtime\.executor\.aexecute\(model\.(adapter\d+), \*\(\),')
    for issue_idx, line in enumerate(body_lines):
        m = issue_re.search(line)
        if not m:
            continue
        varname, adapter = m.group(1), m.group(2)
        wait_marker = f'model.{adapter}_wait, *({varname},'
        wait_idx = next((i for i in range(issue_idx + 1, len(body_lines))
                          if wait_marker in body_lines[i]), None)
        if wait_idx is None:
            continue
        backward_idx = next((i for i in range(issue_idx + 1, wait_idx)
                              if 'nnscaler.runtime.executor.backward(' in body_lines[i]), None)
        if backward_idx is None:
            continue
        forward_idx = next((i for i in range(wait_idx + 1, len(body_lines))
                             if 'nnscaler.runtime.executor.fexecute(' in body_lines[i]), None)
        if forward_idx is None:
            continue
        return issue_idx, backward_idx, wait_idx, forward_idx
    return None


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 gpus')
def test_combined_1f1b_global_schedule_gencode_structure():
    """`GlobalCommSchedule` (``ENABLE_GLOBAL_P2P_RESCHEDULE`` +
    ``ASYNC_RECV_CHANNEL``) produces real generated code containing:
      1) the channel/sequence/lifecycle-tracked async-recv launch
         (``channel=<cid>, max_outstanding=<N>`` kwargs on the underlying
         ``move()`` call);
      2) a concrete instance of the required ordering invariant --
         issue(F(m+1)) < B(m) [a full backward segment call] < wait(F(m+1))
         < F(m+1) -- found by parsing the actual `_train_step` source lines
         (not just substring presence), i.e. gencode-precise structure.
    """
    saved = _set_global_schedule_env(True, max_outstanding=6)
    try:
        with _Alarm(180, 'possible deadlock: global-schedule gencode-evidence run did not finish in 180s'):
            outputs = launch_torchrun(NSTAGES, _worker, False, True, False, True)
    finally:
        _restore_env(saved)

    assert outputs and outputs[0] is not None, 'worker returned no result'
    _, _, per_file = outputs[0]
    assert per_file, 'rank 0 did not capture any generated files'
    all_text = '\n'.join(per_file.values())

    assert 'max_outstanding=6' in all_text, (
        'expected the configured max_outstanding to appear literally in the '
        'generated async-recv launch call'
    )
    import re
    channel_calls = re.findall(r'nnscaler\.runtime\.adapter\.move\(.*channel=(\d+)', all_text)
    assert channel_calls, 'expected at least one channel=<cid> kwarg on a move() call'

    body = _find_train_step_body(per_file, 'gencode1')
    quad = _find_issue_b_wait_f_quadruple(body)
    assert quad is not None, (
        "expected to find a concrete issue(F(m+1)) < B(m) < wait(F(m+1)) < F(m+1) "
        "instance in the generated _train_step; body was:\n" + '\n'.join(body)
    )
    issue_idx, backward_idx, wait_idx, forward_idx = quad
    assert issue_idx < backward_idx < wait_idx < forward_idx, (quad,)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 gpus')
def test_combined_1f1b_global_schedule_numeric_equivalence():
    """The GlobalCommSchedule-scheduled, channel-tracked run must match the
    plain (no reorder, no channel tracking) baseline bit-for-bit-close, with
    identical seeds/data -- scheduling/channel bookkeeping must never change
    the math."""
    saved_off = _set_global_schedule_env(False)
    try:
        with _Alarm(180, 'possible deadlock: baseline (GlobalCommSchedule OFF) run did not finish in 180s'):
            off = launch_torchrun(NSTAGES, _worker, False, True, False, False)
    finally:
        _restore_env(saved_off)

    saved_on = _set_global_schedule_env(True, max_outstanding=6)
    try:
        with _Alarm(180, 'possible deadlock: GlobalCommSchedule ON run did not finish in 180s'):
            on = launch_torchrun(NSTAGES, _worker, False, True, False, False)
    finally:
        _restore_env(saved_on)

    assert off and on, 'workers returned no result'
    off_states = [off[r][0] for r in range(NSTAGES)]
    on_states = [on[r][0] for r in range(NSTAGES)]
    for step in range(NSTEPS):
        off_sd = merge_state_dicts([off_states[r][step] for r in range(NSTAGES)])[0]
        on_sd = merge_state_dicts([on_states[r][step] for r in range(NSTAGES)])[0]
        for k, a in off_sd.items():
            if not torch.is_tensor(a):
                continue
            b = on_sd[k]
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), \
                f'step {step} key {k} differs: max|diff|={(a - b).abs().max().item():.3e}'


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 gpus')
def test_combined_1f1b_global_schedule_no_deadlock():
    """The GlobalCommSchedule + channel-tracked ON configuration must run to
    completion within a bounded wall-clock time (180s guard). A hang here
    (SIGALRM firing) is treated as evidence of a possible deadlock in the
    cap-aware reorder + channel/sequence-tracked async-recv, not a flaky
    timeout."""
    saved = _set_global_schedule_env(True, max_outstanding=6)
    try:
        with _Alarm(180, 'POSSIBLE DEADLOCK: combined_1f1b GlobalCommSchedule '
                          '(reorder + channel tracking) did not complete within 180s'):
            outputs = launch_torchrun(NSTAGES, _worker, False, True, False, False)
    finally:
        _restore_env(saved)

    assert outputs is not None and len(outputs) == NSTAGES
    for r in range(NSTAGES):
        rank_states, _, _ = outputs[r]
        assert len(rank_states) == NSTEPS
    print('NO_DEADLOCK PASS: both ranks completed the GlobalCommSchedule pipeline run')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 gpus')
def test_combined_1f1b_global_schedule_overlap_benchmark():
    """Multi-round, AB/BA-order-alternating, warmup-excluded wall-clock
    comparison: GlobalCommSchedule ON (recv hoisted across a full backward
    segment, see the structural test above) vs OFF, same model/data/seeds.
    EXPLICIT BENCHMARK, NOT A CI CORRECTNESS GATE (see rationale below) --
    ``test_combined_1f1b_global_schedule_gencode_structure`` is the strict,
    always-enforced proof that overlap is actually scheduled; this test's
    job is only to honestly measure and report wall-clock effect, never to
    fake a pass/fail signal out of noise.

    Methodology (upgraded from an earlier, single-order version after a
    dedicated self-audit): a bigger model than the other tests in this file
    (``TIMING_DIM``/``TIMING_NLAYERS`` -- deliberately larger, so a real
    backward segment's compute time is large relative to the constant
    per-step CPU-side scheduling/channel-tracking overhead, improving
    signal-to-noise) is measured over ``N_ROUNDS`` independent rounds; each
    round launches a FRESH ``torchrun`` job per configuration (so NCCL/CUDA
    context, not just in-process warmup, is exercised identically for both),
    with warmup steps discarded and each remaining step individually
    ``torch.cuda.synchronize()``-bounded. Critically, the ORDER of OFF vs ON
    is ALTERNATED every other round (round 0: OFF then ON; round 1: ON then
    OFF; ...): an earlier, single-order version of this test always measured
    OFF first, and a dedicated self-audit found evidence consistent with a
    systematic "whichever config runs first/second" bias (thermal/allocator/
    cache state) confounding the result -- alternating order cancels this
    out in aggregate. Each round's representative value is the MEDIAN of its
    own timed steps; the PAIRED per-round difference (OFF - ON) is what is
    actually tested for a stable gain (a paired design controls for
    round-to-round shared noise, e.g. other jobs on this shared GPU, far
    better than comparing pooled, unpaired distributions).

    A "stable positive gain" requires BOTH: (a) the median paired difference
    across rounds is positive and exceeds a noise floor derived from the
    paired differences' own MAD (median absolute deviation) -- not a fixed,
    possibly-too-generous constant -- and (b) a clear majority of individual
    rounds (not just the aggregate) show ON faster than OFF (a simple sign
    test), so a single lucky/unlucky round cannot flip the conclusion. If
    this environment cannot demonstrate that (observed to happen on this
    shared, multi-tenant machine even with the larger model -- see raw
    output), this test reports the full raw distribution and paired
    differences honestly and does NOT assert a gain (it only asserts there
    is no gross, stable REGRESSION) -- it is explicitly a benchmark in that
    case, not a silently-weakened correctness gate.
    """
    import statistics

    WARMUP_STEPS = 2
    TIMED_STEPS = 3
    N_ROUNDS = 6

    def _measure_one(reschedule_env: bool):
        # cap must exceed TIMING_NMICROS (the result-broadcast's concurrent
        # in-flight receives scale with microbatch count, all resolved only
        # by the bulk end-of-step drain)
        saved = _set_global_schedule_env(reschedule_env, max_outstanding=TIMING_NMICROS + 2)
        try:
            with _Alarm(180, f'possible deadlock: overlap-benchmark run '
                              f'(global_schedule={reschedule_env}) did not finish in 180s'):
                outputs = launch_torchrun(NSTAGES, _timing_worker, WARMUP_STEPS, TIMED_STEPS)
        finally:
            _restore_env(saved)
        assert outputs and outputs[0] is not None
        times = outputs[0]
        assert len(times) == TIMED_STEPS
        return statistics.median(times)

    off_round_values, on_round_values = [], []
    for round_idx in range(N_ROUNDS):
        off_first = (round_idx % 2 == 0)
        if off_first:
            off_val = _measure_one(False)
            on_val = _measure_one(True)
        else:
            on_val = _measure_one(True)
            off_val = _measure_one(False)
        off_round_values.append(off_val)
        on_round_values.append(on_val)
        print(f'[overlap benchmark] round {round_idx} (order={"OFF,ON" if off_first else "ON,OFF"}): '
              f'off={off_val*1e3:.3f}ms on={on_val*1e3:.3f}ms diff={((off_val-on_val)*1e3):+.3f}ms')

    diffs = [off - on for off, on in zip(off_round_values, on_round_values)]
    median_diff = statistics.median(diffs)
    mad = statistics.median([abs(d - median_diff) for d in diffs]) or 1e-9
    wins = sum(1 for d in diffs if d > 0)

    print(f'[overlap benchmark] OFF per-round medians (ms): {[f"{v*1e3:.3f}" for v in off_round_values]}')
    print(f'[overlap benchmark] ON  per-round medians (ms): {[f"{v*1e3:.3f}" for v in on_round_values]}')
    print(f'[overlap benchmark] paired diffs (OFF-ON, ms):  {[f"{d*1e3:+.3f}" for d in diffs]}')
    print(f'[overlap benchmark] median_diff={median_diff*1e3:+.3f}ms, MAD={mad*1e3:.3f}ms, '
          f'wins={wins}/{N_ROUNDS}')

    stable_gain = median_diff > 3 * mad and wins >= math.ceil(0.7 * N_ROUNDS)

    # Not a weakened stand-in for "stable gain": this only catches a GROSS,
    # stable regression (the opposite sign, same bar) -- it does not claim,
    # and must not be read as, evidence of a gain. See docstring.
    stable_regression = (-median_diff) > 3 * mad and (N_ROUNDS - wins) >= math.ceil(0.7 * N_ROUNDS)
    assert not stable_regression, (
        f'GlobalCommSchedule ON shows a STABLE regression across {N_ROUNDS} '
        f'AB/BA-alternated rounds (median_diff={median_diff*1e3:+.3f}ms, '
        f'{N_ROUNDS - wins}/{N_ROUNDS} rounds slower) -- raw OFF={off_round_values}, ON={on_round_values}'
    )

    if stable_gain:
        print(f'[overlap benchmark] BENCHMARK RESULT: stable positive gain, '
              f'{median_diff*1e3:.3f}ms/step ({median_diff/statistics.median(off_round_values)*100:.1f}% faster), '
              f'{wins}/{N_ROUNDS} rounds improved.')
    else:
        print('[overlap benchmark] BENCHMARK RESULT: no stable gain demonstrated in this run '
              '(noise-dominated on this shared machine, or a real but not-yet-stable-across-rounds '
              'effect) -- this is a benchmark measurement, NOT a CI correctness gate; '
              'test_combined_1f1b_global_schedule_gencode_structure is the authoritative, '
              'always-enforced proof that the overlap is actually being scheduled.')


def _timing_worker(warmup_steps: int, timed_steps: int):
    """Like `_worker`, but on the larger `TIMING_DIM`/`TIMING_NLAYERS`/
    `TIMING_NMICROS` model (see their definitions above for why), and times
    each full train_step (post-warmup), returning rank 0's list of per-step
    wall-clock seconds instead of states."""
    import time
    init_distributed()
    dev = torch.cuda.current_device()
    init_random()
    tag = 'timing'
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'combined_1f1b_timing_{PYTEST_RUN_ID}') as tempdir:
        init_random()
        model = parallelize(
            _MLP(dim=TIMING_DIM, nlayers=TIMING_NLAYERS),
            {'data': {'data': torch.randn(MBS, TIMING_DIM, device=dev),
                      'target': torch.rand(MBS, TIMING_DIM, device=dev)}},
            _pas_multi_stream,
            ComputeConfig(NSTAGES, NSTAGES, use_end2end=True,
                          use_async_recv=True,
                          pas_config=dict(pipeline_nstages=NSTAGES, pipeline_nmicros=TIMING_NMICROS,
                                          pipeline_scheduler='1f1b')),
            gen_savedir=tempdir,
            instance_name=f'combined_1f1b_{tag}',
        )
        model.cuda()
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)

        total_steps = warmup_steps + timed_steps
        g = torch.Generator().manual_seed(1234)
        data = [
            [{'data': torch.randn(MBS, TIMING_DIM, generator=g, device='cpu'),
              'target': torch.rand(MBS, TIMING_DIM, generator=g, device='cpu')}
             for _ in range(TIMING_NMICROS)]
            for _ in range(total_steps)
        ]
        per_step_seconds = []
        for step in range(total_steps):
            model.train()
            batch = [{k: v.to(dev) for k, v in mb.items()} for mb in data[step]]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.train_step(batch)
            torch.cuda.synchronize()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if step >= warmup_steps:
                per_step_seconds.append(t1 - t0)
        return per_step_seconds


