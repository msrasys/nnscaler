# F/B action scheduling validation

The implementation adds configurable local-action lookahead and scope/group
transfer units. It is not tied to layer names, tensor shapes or a particular
model. See [the scheduling contract](activation_action_schedule.md) for precise
parameter semantics and cache migration.

## Correctness

* 212 runtime/integration tests passed with CUDA enabled, including explicit
  scope commit boundaries, non-pipeline group compatibility, shared aliases,
  budget retry, oversize-value progress, producer delays and lifecycle cleanup.
* Eight code-generation combinations passed, checking paired before/after
  action positions, valid producer/backward keys and no inference prefetch hooks.
* All eight pipeline offload/input-release/output-release combinations passed.
* Real Attention/MoE PP2/VPP4 comparisons passed five updates each for scope/auto
  lookahead 0, 1 and 4, and group lookahead 4. Full gradients and updated
  parameters were checked, not just logged loss. These test groups overlap.

The plan can be reused with a new lookahead configuration on the next iteration.
Legacy generated offload configs are marked as an older schedule ABI, preventing
silent reuse of fixed-one-action code; raw trace compatibility is retained.
Failure cleanup and tests with cyclic GC disabled cover the metadata lifetimes.

## Controlled local-action replay

The independent probe replays two local virtual stages and two microbatches:
`F(a0), F(b0), F(a1), B(a0), F(b1), B(b0), B(a1), B(b1)`.
Every configuration transfers exactly **512 MiB each direction per iteration**,
in four 128 MiB scopes, with identical compute and checksum work. No CPU-budget
fallback or residual live/pending/speculative bytes occurred.

One B200, physical GPU 1, CPU affinity at its NUMA node 0, with an exclusive
reservation over all eight GPUs. Each case uses three warmups and sixteen timed
iterations. The following cases fence only the compute stream at action ends;
future D2H/H2D copies can continue. This makes action availability explicit and
does not represent full-model throughput.

| Scope lookahead, local F/B actions | Wall ms | Backward read interval ms | Whole-step GPU allocated peak MiB | Diagnostic boundary peak MiB |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 26.790 | 12.543 | 672.1 | 32.1 |
| 1 | 19.590 | 5.027 | 800.1 | 160.1 |
| 2 | 18.138 | 3.622 | 928.1 | 288.1 |
| 4 | 18.047 | 3.551 | 928.1 | 416.1 |
| 8 | 17.994 | 3.522 | 928.1 | 416.1 |

The speculative window is 512 MiB. Earlier prefetch reduces latency here, but
the benefit plateaus around two actions while more data can remain on GPU.
Producer completion clips the effective lead, so increasing N cannot make data
available before it exists. Diagnostic boundary values are sampled separately;
they do not substitute for the allocator peak.

Backward read intervals include full checksum reductions and CPU submission
gaps as well as readiness waits. The reductions create temporary workspaces,
which are included in whole-step GPU peaks. Those intervals must not be labeled
as pure DMA stall time or used to claim that all transfer overhead disappeared.

## Why the controls are separate

With N=4, reducing the speculative window from 512 to 64 MiB changed wall time
from 18.047 to **24.184 ms**, and the read interval from 3.551 to **9.402 ms**.
Whole-step peak allocation fell to **672.1 MiB**. Both cases reported zero demand
loads: a prefetch queued just before a read can still leave exposed waiting.

Group mode with an entry window of one group measured 26.561 ms. Increasing its
entry coverage to both groups, while retaining the same N=4 and byte window,
measured **17.471 ms**. This is the matched-coverage comparison with scope mode;
the one-group result must not be presented as evidence that group submission is
intrinsically slow. Smaller forward submissions can start D2H earlier inside F,
while scope mode provides the requested F/B logical unit by default.

Without compute-action fences, N=0 and N=4 measured **18.473 / 17.988 ms**.
Host submission can already run ahead of GPU execution, so its effective window
differs. The larger gain in the fenced experiment cannot be extrapolated to all
training loops. Actual scheduling, transfer volume, byte limits and CPU/GPU
submission behavior all matter.

## Resource and source audit

The replay's pinned pool was 384 MiB with action fences and 448 MiB without them,
under a 2 GiB CPU cap. Intermediate backwards release buffers for later forwards.
Including numerical and code-generation checks, the largest process-tree RSS
was about **8.27 GiB**, sampled single-GPU use stayed below **2772 MiB**, and disk
free space stayed above **847 GiB**. No full model checkpoint was written.

The final frozen candidate is `v4` under `/tmp/activation-schedule-20261030`.
Its Python source hashes match the final working implementation. All validation
commands completed successfully, resource guards reported no stop condition,
and no owned processes remained. Small test-generated caches are cleaned after
archiving evidence; frozen source and compact records remain for review.

Configurations, plans, raw samples, traces, compressed logs and resource/source
records are in the [evidence manifest](../activation-store-results/20261030/action-schedule/manifest.json).
The probe is `benchmark/benchmark_activation_schedule.py`; use `--lookahead`,
`--prefetch-mib`, `--unit`, `--entry-groups` and `--sync-actions` to reproduce the
separate effects. None of these comparisons changes model-region annotations.
