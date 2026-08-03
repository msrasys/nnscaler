#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Real, compiled, 2-GPU end-to-end proof of Step B (local segments inside
one physical pipeline stage,
``nnscaler.graph.schedule.local_segment``): local-segment splitting must
(1) never change the trained math versus an unsplit baseline, (2) never
deadlock, and (3) remain fully compatible with Step A's
``GlobalCommSchedule``/async-recv machinery -- in particular the
``issue(F(m+1)) < B(m) < wait(F(m+1)) < F(m+1)`` gencode invariant
(``test_combined_1f1b_pipeline_e2e.py``'s
``test_combined_1f1b_global_schedule_gencode_structure``) must still hold
with local segments enabled.

Same test harness conventions as ``test_combined_1f1b_pipeline_e2e.py``
(``launch_torchrun``, ``_Alarm`` 180s wall-clock deadlock guard,
``clone_to_cpu_recursively``/``merge_state_dicts`` state comparison).

Requires >= 2 GPUs.
"""
import os
import re
import signal
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer, merge_state_dicts
from nnscaler.ir.operator import IRFwOperation
import nnscaler.runtime.function as ncf
from nnscaler.graph.schedule.local_segment import AnchorBoundary, LocalSegmentSched

from .common import init_distributed, assert_close
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import init_random, clear_dir_on_rank0, PYTEST_RUN_ID

DIM = 16
NLAYERS = 8    # 4 layers per stage; each stage's 4 layers split 2+2 by an anchor
MBS = 2
NSTAGES = 2    # pure pipeline-parallel, no tensor-parallel -> only 2 GPUs needed
NMICROS = 4
NSTEPS = 2
LSEG_ANCHOR = 'lseg'


class _LSModel(nn.Module):
    """`NLAYERS` unbiased Linear layers; when `use_anchor` an ``lseg`` anchor
    is emitted before the (locally) second half of every stage's layers, so
    ``AnchorBoundary({'lseg'})`` splits each stage into exactly 2 local
    segments."""

    def __init__(self, dim=DIM, nlayers=NLAYERS, nstages=NSTAGES, use_anchor=False):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(nlayers))
        self.use_anchor = use_anchor
        per_stage = nlayers // nstages
        # global layer indices that start the second half of each stage
        self._split_layer_idx = {sid * per_stage + per_stage // 2 for sid in range(nstages)}

    def forward(self, data):
        x = data['data']
        for i, layer in enumerate(self.layers):
            if self.use_anchor and i in self._split_layer_idx:
                ncf.anchor(LSEG_ANCHOR)
            x = layer(x)
        return x.sum()


def _pas_local_segments(graph, config: ComputeConfig):
    """Stage the graph into ``NSTAGES`` physical stages (one Linear-layer
    block per stage), splitting each stage into local segments via
    ``AnchorBoundary({'lseg'})`` when ``use_local_segments`` is set (a plain
    ``graph.staging([...])`` call is not used -- see
    ``nnscaler.graph.schedule.local_segment`` module docstring: local
    segments must be created *before* any single-shot whole-stage grouping,
    by calling ``partition_stage_into_local_segments`` once per stage
    instead)."""
    num_stages = config.pas_config['pipeline_nstages']
    nmicros = config.pas_config['pipeline_nmicros']
    use_local_segments = config.pas_config['use_local_segments']

    from nnscaler.graph.schedule.local_segment import partition_stage_into_local_segments

    all_fwd = [n for n in graph.nodes() if isinstance(n, IRFwOperation)]
    linear_positions = [i for i, n in enumerate(all_fwd) if n.name == 'linear']
    assert len(linear_positions) == NLAYERS
    per_stage = NLAYERS // num_stages
    stage_start_positions = [linear_positions[sid * per_stage] for sid in range(num_stages)]
    # The first stage must also absorb any leading forward ops that precede
    # its first named-boundary node (e.g. the `getitem` op tracing
    # `data['data']` emits, ahead of the first `linear` call) -- otherwise
    # they are left dangling, belonging to no segment at all. This mirrors
    # `IRGraph.blocking()`'s own explicit "adjust the start of the first
    # stage to involve beginning operators" step.
    stage_start_positions[0] = 0

    all_segs = []
    for sid in range(num_stages):
        start = stage_start_positions[sid]
        end = stage_start_positions[sid + 1] if sid + 1 < num_stages else len(all_fwd)
        stage_nodes = all_fwd[start:end]
        boundary = AnchorBoundary({LSEG_ANCHOR}) if use_local_segments else None
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundary)
        all_segs.append(segs)

    dataloader = graph.nodes()[0]
    sub_nodes = graph.replicate(dataloader, num_stages)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)
    for sid in range(num_stages):
        for seg in all_segs[sid]:
            for node in seg.nodes():
                graph.assign(node, sid)

    config.apply_pipeline_scheduler(graph, num_stages, nmicros, LocalSegmentSched.sched_1f1b_local_segments)
    return graph


def _make_data(nsteps, nmicros, seed=1234):
    g = torch.Generator().manual_seed(seed)
    steps = []
    for _ in range(nsteps):
        steps.append([
            {'data': torch.randn(MBS, DIM, generator=g, device='cpu')}
            for _ in range(nmicros)
        ])
    return steps


def _worker(use_local_segments: bool, capture_per_file: bool = False):
    init_distributed()
    dev = torch.cuda.current_device()
    init_random()
    tag = 'split' if use_local_segments else 'unsplit'
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'local_segments_e2e_{PYTEST_RUN_ID}_{tag}') as tempdir:
        init_random()
        model = parallelize(
            _LSModel(use_anchor=use_local_segments),
            {'data': {'data': torch.randn(MBS, DIM, device=dev)}},
            _pas_local_segments,
            ComputeConfig(
                NSTAGES, NSTAGES, use_end2end=True,
                use_async_recv=True,
                pas_config=dict(
                    pipeline_nstages=NSTAGES, pipeline_nmicros=NMICROS,
                    use_local_segments=use_local_segments,
                ),
            ),
            gen_savedir=tempdir,
            instance_name=f'local_segments_{tag}',
        )

        per_file = {}
        if capture_per_file and torch.distributed.get_rank() == 0:
            pyfiles = sorted(tempdir.rglob('*.py'))
            per_file = {str(p.relative_to(tempdir)): p.read_text() for p in pyfiles}

        model.cuda()
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)

        data = _make_data(NSTEPS, NMICROS)
        states = []
        opt_states = []
        for step in range(NSTEPS):
            model.train()
            batch = [{k: v.to(dev) for k, v in mb.items()} for mb in data[step]]
            model.train_step(batch)
            torch.cuda.synchronize()
            optimizer.step()
            optimizer.zero_grad()
            states.append(clone_to_cpu_recursively(model.state_dict()))
            # includes the automatically-injected CUBE_EXTRA_STATE key
            # (added by build_optimizer's optimizer.state_dict patch),
            # required by merge_state_dicts to reassemble per-rank Adam
            # momentum/step across ranks -- see test below.
            opt_states.append(clone_to_cpu_recursively(optimizer.state_dict()))
        return states, opt_states, per_file


class _Alarm:
    """Hard wall-clock deadlock guard (SIGALRM), matching
    ``test_combined_1f1b_pipeline_e2e.py``'s ``_Alarm``."""

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


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 gpus')
def test_local_segments_numeric_equivalence_vs_unsplit_baseline():
    """Splitting each stage into 2 local segments (vs. the unsplit baseline,
    i.e. one whole-stage segment, both scheduled via the exact same
    ``LocalSegmentSched.sched_1f1b_local_segments``) must not change the
    trained weights *or* optimizer state (Adam momentum/step), across
    multiple steps."""
    with _Alarm(180, 'possible deadlock: unsplit baseline run did not finish in 180s'):
        off = launch_torchrun(NSTAGES, _worker, False, False)
    with _Alarm(180, 'possible deadlock: local-segment-split run did not finish in 180s'):
        on = launch_torchrun(NSTAGES, _worker, True, False)

    assert off and on, 'workers returned no result'
    off_states = [off[r][0] for r in range(NSTAGES)]
    on_states = [on[r][0] for r in range(NSTAGES)]
    off_opt_states = [off[r][1] for r in range(NSTAGES)]
    on_opt_states = [on[r][1] for r in range(NSTAGES)]
    for step in range(NSTEPS):
        off_sd, off_opt_sd = merge_state_dicts(
            [off_states[r][step] for r in range(NSTAGES)],
            [off_opt_states[r][step] for r in range(NSTAGES)],
        )
        on_sd, on_opt_sd = merge_state_dicts(
            [on_states[r][step] for r in range(NSTAGES)],
            [on_opt_states[r][step] for r in range(NSTAGES)],
        )
        for k, a in off_sd.items():
            if not torch.is_tensor(a):
                continue
            b = on_sd[k]
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), \
                f'step {step} key {k} differs: max|diff|={(a - b).abs().max().item():.3e}'

        # Also compare merged optimizer state (Adam exp_avg/exp_avg_sq
        # momentum + step counters), not just model weights -- uses the
        # same merge_state_dicts(model_sds, optimizer_sds) API, proven in
        # tests/parallel_module/test_checkpoint.py. (Post-commit
        # self-audit finding #4: the original commit only ever compared
        # model weights.)
        assert_close(off_opt_sd, on_opt_sd, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 gpus')
def test_local_segments_no_deadlock():
    """The local-segment-split pipeline (with its stage-0 B/F interleave)
    must run to completion within a bounded wall-clock time."""
    with _Alarm(180, 'POSSIBLE DEADLOCK: local-segment pipeline did not complete within 180s'):
        outputs = launch_torchrun(NSTAGES, _worker, True, False)
    assert outputs is not None and len(outputs) == NSTAGES
    for r in range(NSTAGES):
        rank_states, rank_opt_states, _ = outputs[r]
        assert len(rank_states) == NSTEPS
        assert len(rank_opt_states) == NSTEPS
    print('NO_DEADLOCK PASS: both ranks completed the local-segment pipeline run')


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


def _restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _find_train_step_body(per_file: dict, name_contains: str = 'gencode1') -> list:
    for name, content in per_file.items():
        if name_contains not in name:
            continue
        lines = content.splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith('def _train_step'))
        end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith('def _infer_step'))
        return lines[start:end]
    raise AssertionError(f'no generated file matching {name_contains!r} found in {list(per_file)}')


def _find_issue_b_wait_f_quadruple(body_lines: list):
    """Same search as
    ``test_combined_1f1b_pipeline_e2e._find_issue_b_wait_f_quadruple``: an
    async-recv issue, followed (later, with at least one backward *segment*
    call in between -- now possibly one of several local segments) by its
    deferred wait, followed by a forward segment call that consumes it."""
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
def test_local_segments_compatible_with_step_a_global_schedule():
    """Local segments combined with Step A's GlobalCommSchedule
    (``ENABLE_GLOBAL_P2P_RESCHEDULE`` + ``ASYNC_RECV_CHANNEL``) must still
    produce generated code with (1) multiple local-segment methods per
    split stage and (2) a concrete
    ``issue(F(m+1)) < B(m) < wait(F(m+1)) < F(m+1)`` instance -- i.e. Step B
    does not break Step A's structural invariant."""
    saved = _set_global_schedule_env(True, max_outstanding=6)
    try:
        with _Alarm(180, 'possible deadlock: local-segments + GlobalCommSchedule run did not finish in 180s'):
            outputs = launch_torchrun(NSTAGES, _worker, True, True)
    finally:
        _restore_env(saved)

    assert outputs and outputs[0] is not None, 'worker returned no result'
    _, _, per_file = outputs[0]
    assert per_file, 'rank 0 did not capture any generated files'
    all_text = '\n'.join(per_file.values())

    # Multiple local-segment methods actually present, and *exactly* 2 per
    # physical-stage gencodeN.py file (matching _LSModel's construction:
    # per_stage = NLAYERS // NSTAGES layers, split at the midpoint by
    # AnchorBoundary -> 2 local segments/stage) -- not merely the weaker
    # ">= 2 in total" this test originally checked, which would also be
    # (wrongly) satisfied by e.g. all local segments living in a single
    # stage's file and none in another. (Post-commit self-audit finding,
    # MEDIUM-LOW severity.)
    gencode_files = {name: content for name, content in per_file.items()
                      if re.search(r'gencode\d+\.py$', name)}
    assert len(gencode_files) == NSTAGES, (
        f'expected exactly {NSTAGES} per-stage gencodeN.py files, found {sorted(gencode_files)}'
    )
    expected_segs_per_stage = 2  # see _LSModel docstring
    all_seg_defs = []
    for sid in range(NSTAGES):
        name = next((n for n in gencode_files if n.endswith(f'gencode{sid}.py')), None)
        assert name is not None, f'no gencode{sid}.py found among {sorted(gencode_files)}'
        seg_defs = sorted(set(re.findall(r'^\s*def (segment\d+)\(', gencode_files[name], re.MULTILINE)))
        assert len(seg_defs) == expected_segs_per_stage, (
            f'stage {sid} ({name}): expected exactly {expected_segs_per_stage} local segment '
            f'methods, found {seg_defs}'
        )
        all_seg_defs.extend(seg_defs)
    assert len(set(all_seg_defs)) == NSTAGES * expected_segs_per_stage, (
        f'expected {NSTAGES * expected_segs_per_stage} distinct segment method names in total '
        f'across all stages, found {sorted(set(all_seg_defs))}'
    )

    assert 'max_outstanding=6' in all_text, (
        'expected the configured max_outstanding to appear literally in the generated async-recv launch call'
    )
    channel_calls = re.findall(r'nnscaler\.runtime\.adapter\.move\(.*channel=(\d+)', all_text)
    assert channel_calls, 'expected at least one channel=<cid> kwarg on a move() call'

    body = _find_train_step_body(per_file, 'gencode1')
    quad = _find_issue_b_wait_f_quadruple(body)
    assert quad is not None, (
        "expected to find a concrete issue(F(m+1)) < B(m) < wait(F(m+1)) < F(m+1) "
        "instance in the generated _train_step with local segments enabled; body was:\n" + '\n'.join(body)
    )
    issue_idx, backward_idx, wait_idx, forward_idx = quad
    assert issue_idx < backward_idx < wait_idx < forward_idx, (quad,)
