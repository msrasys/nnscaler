# AGENTS.md — nnscaler

Guidance for coding agents (Claude Code, Copilot, Cursor, etc.) working in this repo. Read this file before proposing changes. Task-specific procedures live as skills under [.agents/skills/](.agents/skills/) — load the matching one when relevant.

---

## 1. What nnscaler is

nnscaler is a compiler + runtime for distributed PyTorch training. It **traces** a user model, lowers it to an **IR**, runs **graph transforms + auto-distribution** to pick a TP/DP/PP parallelization strategy, builds an **execution plan / schedule**, **generates** per-rank Python modules, and runs them through its **runtime** (CUDA streams + NCCL collectives).

The single most important consequence: behavior you observe at runtime — performance, memory, numerical layout — is the product of decisions made at *every* stage of that pipeline. Most non-trivial changes touch more than one layer.

---

## 2. Compilation pipeline (mental model)

```
user model  ─►  trace  ─►  IR  ─►  graph partition / RVD  ─►  autodist policy
                                                                    │
                                                                    ▼
        runtime (streams, collectives)  ◄──  gencode{rank}.py  ◄──  codegen  ◄──  execplan / schedule
```

| Stage | Owns | Lives in |
|---|---|---|
| Trace / orchestration | entry points, compile cache, `ComputeConfig`, `ParallelModule` | [nnscaler/compiler.py](nnscaler/compiler.py), [nnscaler/parallel.py](nnscaler/parallel.py), [nnscaler/program.py](nnscaler/program.py) |
| IR | tensor / op representation, layouts, adapter & comm primitives | [nnscaler/ir/](nnscaler/ir/), esp. [nnscaler/ir/adapter/prim.py](nnscaler/ir/adapter/prim.py) |
| Graph | partition, fwd/bwd segmentation, pipeline schedules, RVD adapter selection | [nnscaler/graph/](nnscaler/graph/), schedules under [nnscaler/graph/schedule/](nnscaler/graph/schedule/) |
| Autodist | TP/DP/PP search, cost model, partition options, policies | [nnscaler/autodist/](nnscaler/autodist/), [nnscaler/policies.py](nnscaler/policies.py) |
| ExecPlan | per-device op sequence, adapter fusion, multi-stream flag | [nnscaler/execplan/](nnscaler/execplan/) |
| Codegen | emitted `_train_step`, stream contexts, tensor lifecycle | [nnscaler/codegen/](nnscaler/codegen/) |
| Runtime | streams, collectives, async work, gradient bucketing | [nnscaler/runtime/](nnscaler/runtime/) |
| Profiler | op/comm cost DB consumed by autodist solver | [nnscaler/profiler/](nnscaler/profiler/) |
| Custom ops / examples | user-facing ops and reference training entries | [nnscaler/customized_ops/](nnscaler/customized_ops/), [examples/](examples/) |

Every one of these layers is **part of nnscaler and may be modified**. A common agent mistake is treating generated code, schedules, or the IR as immutable; they are outputs of code in this repo, and the right fix usually lives in the layer that produced them, not downstream of it.

---

## 3. Where common kinds of work belong

This is a routing table, not an exhaustive list. Use it to pick the right layer before you start editing.

| Kind of task | Primary layer | Secondary / often touched |
|---|---|---|
| Add support for a new model architecture | [examples/](examples/) (training entry), possibly [nnscaler/customized_ops/](nnscaler/customized_ops/) for novel ops | Custom op registration ([docs/source/register_custom_op.md](docs/source/register_custom_op.md)); partition constraints if autodist picks a bad plan |
| Add a custom op (e.g. fused attention) | [nnscaler/customized_ops/](nnscaler/customized_ops/) | Dim annotations ([docs/source/dimops.md](docs/source/dimops.md)); profiler entry if it changes cost meaningfully |
| New collective / comm primitive | [nnscaler/ir/adapter/prim.py](nnscaler/ir/adapter/prim.py) + [nnscaler/runtime/adapter/collectives.py](nnscaler/runtime/adapter/collectives.py) | Fusion in [nnscaler/execplan/planpass/fusion.py](nnscaler/execplan/planpass/fusion.py); codegen if it needs special emission |
| New pipeline schedule | [nnscaler/graph/schedule/predefined.py](nnscaler/graph/schedule/predefined.py) (+ [schedplan.py](nnscaler/graph/schedule/schedplan.py)) | Schedule registry in [nnscaler/parallel.py](nnscaler/parallel.py); emission in [nnscaler/codegen/schedule/schedule.py](nnscaler/codegen/schedule/schedule.py) |
| Change parallelization policy / solver heuristic | [nnscaler/policies.py](nnscaler/policies.py), [nnscaler/autodist/](nnscaler/autodist/) | Profile DB invalidation if op costs change |
| Constrain the solver for a specific model | User-side `ComputeConfig.pas_config` + [docs/source/partition_constraints_guide.md](docs/source/partition_constraints_guide.md) | Usually no nnscaler-side change |
| Improve comm/compute overlap | Stream assignment in [nnscaler/codegen/schedule/schedule.py](nnscaler/codegen/schedule/schedule.py) (`_get_node_stream`); `StreamConfig` in [nnscaler/graph/schedule/schedplan.py](nnscaler/graph/schedule/schedplan.py); runtime in [nnscaler/runtime/](nnscaler/runtime/) | |
| Reduce memory / OOM | Recomputation flags; pipeline schedule choice; micro-batch count; tensor lifecycle in [nnscaler/codegen/lifecycle.py](nnscaler/codegen/lifecycle.py) | |
| Fix codegen bug (wrong emitted code) | [nnscaler/codegen/](nnscaler/codegen/) — *not* the generated file | Add a regression test under [tests/codegen/](tests/codegen/) |
| Runtime bug (hang, wrong gradient, NCCL error) | [nnscaler/runtime/](nnscaler/runtime/) | Often the upstream cause is in codegen or schedule |

If a task plausibly fits two rows, edit the **earlier** layer in the pipeline (§2) — fixes downstream of a buggy generator are erased on the next compile.

---

## 4. Footguns (read before editing)

- **Read the generated files, but do not hand-edit them.** Generated per-rank code lives at `.nnscaler/_parallel_modules/__main__/{ModelName}/_/gencode{rank}.py`. Reading `gencode{rank}.py` is often the fastest way to understand what the compiler actually produced — which stream a collective lands on, where syncs sit, what the lifecycle looks like — and is highly recommended when debugging or optimizing. But any patch there is erased on the next compile; fix the generator (codegen / schedule / execplan) instead.
- **Stale compile cache silently invalidates results.** `compile(..., override=False)` reuses prior gencode and the autodist solution with no warning. When you change anything in `nnscaler/graph/`, `nnscaler/execplan/`, `nnscaler/codegen/`, `nnscaler/autodist/`, `nnscaler/policies.py`, or `nnscaler/ir/adapter/`, you must invalidate:
  - set `override=True` in `compile()` / `ComputeConfig`, **and/or**
  - delete `.nnscaler/` (gencode), **and/or**
  - delete `~/.cache/nnscaler/autodist/{version}/{GPU}/` (profile DB — required when op cost characteristics change).
- **Verify a change took effect by diffing the new `gencode{rank}.py` against the previous one.** If they are byte-identical, the cache was not invalidated.
- **Don't change op signatures without re-registering** the custom op — see [docs/source/register_custom_op.md](docs/source/register_custom_op.md).
- **Don't disable correctness checks to make tests or perf numbers go green.** Validate loss/grad parity on a small config (single-GPU vs. parallel) before claiming a feature or perf win.

---

## 5. Testing and measurement entry points

- **Unit / integration tests:** [tests/](tests/) — per-subsystem layout (`codegen/`, `graph/`, `runtime/`, `autodist/`, `parallel_module/`, …). Run with `pytest tests/<area>`.
- **RVD primitives correctness:** [utility/test_rvd_prim.py](utility/test_rvd_prim.py) with `--prims all`.
- **Comm latency micro-bench:** `python -m nnscaler.profiler.benchmark_comm`.
- **Benchmark scripts:** [benchmark/](benchmark/) (`benchmark_ring_attn.py`, `benchmark_zigzag_attn.py`, …).
- **Reference training entry:** [examples/llama/train.py](examples/llama/train.py).

When measuring performance, use the **smallest config that reproduces the phenomenon** — short iteration loops dominate the quality of the optimization process.

---

## 6. Existing documentation worth knowing about

- [docs/source/parallel_module.md](docs/source/parallel_module.md) — `ParallelModule` / `ComputeConfig`
- [docs/source/register_custom_op.md](docs/source/register_custom_op.md) — custom op registration
- [docs/source/dimops.md](docs/source/dimops.md) — dim-annotated ops
- [docs/source/partition_constraints_guide.md](docs/source/partition_constraints_guide.md) — constraining the autodist solver
- [docs/source/trainer.md](docs/source/trainer.md), [docs/source/pytorch_lightning.md](docs/source/pytorch_lightning.md) — trainer integrations
- [docs/source/troubleshooting.rst](docs/source/troubleshooting.rst)
- [docs/source/control_flow.md](docs/source/control_flow.md), [docs/source/einops.md](docs/source/einops.md), [docs/source/verify_op.md](docs/source/verify_op.md)

---

## 7. Maintainer note

This file describes **stable architecture and routing**, not APIs. When a layer in §2 changes shape (not just internals), update the corresponding row and any affected skill. Stale guidance is worse than none — agents will confidently edit the wrong layer.
