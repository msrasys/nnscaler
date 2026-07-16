#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from types import SimpleNamespace

import pytest

from nnscaler.profiler import chronotrigger_hook


def _trainer(rank=0, world=8, plan_size=8, pas_config=None, step=0):
    compute_config = SimpleNamespace(
        plan_ngpus=plan_size,
        pas_config=pas_config or {"pp_size": 4},
    )
    return SimpleNamespace(
        rank=rank,
        world_size=world,
        train_args=SimpleNamespace(compute_config=compute_config),
        train_status=SimpleNamespace(finished_train_steps=step),
    )


def test_rank_layout_uses_explicit_inner_parallelism(monkeypatch):
    monkeypatch.setenv("NNSCALER_TRACE_EP_SIZE", "2")

    layout = chronotrigger_hook._rank_layout(_trainer(rank=5))

    assert layout == {
        "rank": 5,
        "world": 8,
        "dp": 0,
        "pp": 2,
        "ep": 1,
        "tp": 0,
        "cp": 0,
    }


def test_rank_layout_does_not_guess_unidentified_inner_axis():
    layout = chronotrigger_hook._rank_layout(_trainer(rank=5))

    assert layout["pp"] == 2
    assert layout["ep"] == 0
    assert layout["tp"] == 0
    assert layout["cp"] == 0


def test_rank_layout_accepts_legacy_launcher_hints(monkeypatch):
    monkeypatch.setenv("PIPELINE_VPP_PP_SIZE", "4")
    monkeypatch.setenv("PIPELINE_VPP_EP_SIZE", "2")

    layout = chronotrigger_hook._rank_layout(
        _trainer(rank=5, pas_config={"pipeline_size": 1})
    )

    assert layout["pp"] == 2
    assert layout["ep"] == 1


def test_rank_layout_rejects_incomplete_inner_parallelism(monkeypatch):
    monkeypatch.setenv("NNSCALER_TRACE_EP_SIZE", "3")

    with pytest.raises(ValueError, match="must multiply to plan_ngpus"):
        chronotrigger_hook._rank_layout(_trainer())


def test_disabled_hook_does_not_resolve_rank_layout(monkeypatch):
    trainer = _trainer()
    monkeypatch.setenv("CT_TRACE", "off")
    monkeypatch.setenv("NNSCALER_TRACE_EP_SIZE", "invalid")
    init_calls = []
    monkeypatch.setattr(
        chronotrigger_hook.ct,
        "init",
        lambda **kwargs: init_calls.append(kwargs),
    )

    chronotrigger_hook.ChronoTriggerTrainHook().after_setup(trainer)

    assert init_calls == [
        {
            "profile": "nnscaler",
            "schema": "nnscaler.v1",
            "rank_layout": None,
            "enabled": False,
        }
    ]


def test_hook_accepts_legacy_trace_gate(monkeypatch):
    monkeypatch.setenv("NNSCALER_NVTX_TRACE", "1")
    monkeypatch.setenv("PIPELINE_VPP_EP_SIZE", "2")
    init_calls = []
    monkeypatch.setattr(
        chronotrigger_hook.ct,
        "init",
        lambda **kwargs: init_calls.append(kwargs),
    )

    chronotrigger_hook.ChronoTriggerTrainHook().after_setup(_trainer())

    assert init_calls[0]["enabled"] is True
    assert init_calls[0]["rank_layout"]["ep"] == 0


def test_hook_owns_step_and_delayed_capture(monkeypatch):
    trainer = _trainer(step=10)
    events = []
    monkeypatch.setenv("CT_TRACE", "1")
    monkeypatch.setenv("NNSCALER_TRACE_EP_SIZE", "2")
    monkeypatch.setenv("NNSCALER_NSYS_CAPTURE_START_STEP", "10")
    monkeypatch.setenv("NNSCALER_NSYS_CAPTURE_STEPS", "2")
    monkeypatch.setattr(
        chronotrigger_hook.ct,
        "init",
        lambda **kwargs: events.append(("init", kwargs)),
    )
    monkeypatch.setattr(
        chronotrigger_hook.ct,
        "set_step",
        lambda step: events.append(("step", step)),
    )
    monkeypatch.setattr(
        chronotrigger_hook.ct,
        "emit_metadata",
        lambda: events.append(("metadata",)),
    )
    monkeypatch.setattr(
        chronotrigger_hook,
        "_distributed_barrier",
        lambda: events.append(("barrier",)),
    )
    monkeypatch.setattr(
        chronotrigger_hook.torch.cuda,
        "synchronize",
        lambda: events.append(("synchronize",)),
    )
    monkeypatch.setattr(
        chronotrigger_hook,
        "_cuda_profiler_call",
        lambda name: events.append((name,)),
    )

    hook = chronotrigger_hook.ChronoTriggerTrainHook()
    hook.after_setup(trainer)
    hook.on_step_start(trainer, 0, 10)
    events.append(("zero_grad",))
    events.append(("train_step",))
    events.append(("optimizer_step",))
    events.append(("lr_scheduler",))
    events.append(("logging",))
    trainer.train_status.finished_train_steps = 11
    hook.on_step_end(trainer, 0, 10, {}, None)
    hook.on_step_start(trainer, 0, 11)
    events.append(("zero_grad",))
    events.append(("train_step",))
    events.append(("optimizer_step",))
    events.append(("lr_scheduler",))
    events.append(("logging",))
    trainer.train_status.finished_train_steps = 12
    hook.on_step_end(trainer, 0, 11, {}, None)

    assert events == [
        (
            "init",
            {
                "profile": "nnscaler",
                "schema": "nnscaler.v1",
                "rank_layout": {
                    "rank": 0,
                    "world": 8,
                    "dp": 0,
                    "pp": 0,
                    "ep": 0,
                    "tp": 0,
                    "cp": 0,
                },
                "enabled": True,
            },
        ),
        ("step", 10),
        ("barrier",),
        ("synchronize",),
        ("cudaProfilerStart",),
        ("metadata",),
        ("zero_grad",),
        ("train_step",),
        ("optimizer_step",),
        ("lr_scheduler",),
        ("logging",),
        ("step", 11),
        ("zero_grad",),
        ("train_step",),
        ("optimizer_step",),
        ("lr_scheduler",),
        ("logging",),
        ("synchronize",),
        ("barrier",),
        ("cudaProfilerStop",),
    ]