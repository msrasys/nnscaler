# Offload, recompute and stream integration

This experimental branch starts from the deployed NNScaler `yileiyang/vl_pp`
revision and combines independently developed features for integration testing.
All new feature switches must remain disabled by default.

## Pinned sources

- production: `14c46e62da1c0f0233fab9dcf492197fff5cda0d`
- offload_original_head: `07479ae3f3068241d5883048806a1e9f3699da37`
- offload_snapshot: `69e1ade1656a2e0408130d007e0b74131311eab1`
- offload_merge_base: `f04321bab19a4f5a430cb46108940c375470a98e`
- overlap: `95ae10df0b09d8600c26eaeaa2115fad6054b82c`

The offload snapshot includes its validated, previously uncommitted recompute
merge and cleanup. The original source worktree and index are unchanged.

## Acceptance

Compare the deployed two-repository baseline against integration with all
features disabled, then cover all eight combinations of recompute merging,
CPU activation offload and module streams. Use isolated Python environments
and generated-code directories; keep the same stock DeepEP build, inputs,
initial weights and training settings.

Require first-step gradient-norm difference below 1e-5, multi-step numerical
checks, checkpoint save/resume checks, and warmed-up performance and F/B-memory
measurements. Compile/unit tests do not substitute for GPU training validation.

GPU work must wait until the overlap chat completes and the GPUs are released.
Maintain disk and host-memory limits and remove only this experiment's
regenerable artifacts after preserving verification evidence.

Status: CPU integration validated; GPU validation pending.

## Integration decisions

- Preserve deployed optimizer and cross-entropy implementations rather than
  importing unrelated older performance changes.
- Keep one activation scope around each scheduled forward, including stream
  fragments. Preserve the final offload-group boundary across those fragments.
- Recompute composition stops at distinct stream invocations/readiness domains.
- DeepEP runtime settings have one CLI registration and agree between buffer
  allocation and collectives; both named fields and five-value arrays load.
- Use stock DeepEP. External NIC/RDMA benchmark prototypes are not installed
  as default activation transfer backends.

CPU validation: NNScaler combined suite 253 passed/83 CUDA-skipped; LLM
configuration/policy suite 570 passed; the generated eight-combination CPU
semantic matrix plus stream-boundary checks 10 passed/8 CUDA-skipped;
DeepEP configuration follow-up 21 passed. CPU stream serialization tests
validate graph/autograd composition, not CUDA concurrency.

Checkpoint follow-up: all eight CPU combinations also pass a saved model and
Adam-state round trip followed by another update; configuration serialization
preserves every feature switch. CUDA concurrency/resume checks are collected
but remain gated on overlap chat completion.

The combined compiler uses trace metadata version 6. Packages generated before
activation/stream integration are regenerated even when their individual
feature settings happen to match, preserving the deployed lifetime fixes.

CUDA component validation: 104 tests passed, including all eight combinations
and model/Adam checkpoint continuation. The first test attempt incorrectly
loaded Adam's CPU step scalar onto CUDA; loading the checkpoint on CPU and
letting optimizer.load_state_dict place each value fixed the test.

## Default recompute compatibility

The deployed OpPlan policy skips single-operation checkpoints. Preserve that
behavior with `recompute_single_ops=False`; explicitly select singletons when
using them, or enable adjacent-region merging. The expanded validation matrix
sets `--recompute_single_ops` in every arm so its comparisons change only the
three features. The original-recipe all-off compatibility pair leaves it off.

The first real-model integration check exposed stale calls to a deleted MoE
forced-storage-release helper. Private owners now leave scope normally rather
than resizing aliased storage. dsilu buffer reuse is disabled for retained or
higher-order graph tasks; 45 real MoE kernel checks passed, including repeated
backward and borrowed-storage preservation.

## Disabled checkpoint fast path

With CPU offload, adjacent merging and singleton selection all disabled, code
generation imports the deployed `torch.utils.checkpoint` directly. The saved
input alias adapter is selected only by those explicit activation features.
Module-stream-only execution preserves its original checkpoint context path.
A runtime regression rejects accidental entry into the new adapter when off.
CPU checks48passed/8CUDA-skipped; isolated CUDA matrix22passed.
