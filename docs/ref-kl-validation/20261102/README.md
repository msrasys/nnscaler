# Reference KL memory validation (2026-11-02 UTC)

Base revisions: llm-train `7234c269adcda0b1cd5424570bbd291d5b9bb4f0`
(main) and nnscaler `349449e342e4cd6b5b53bd1a8cebecc8241b4737`
(yileiyang/vl_pp). All runs use the real `nnscaler_train_vl.py` entry point.

## Workload and agreement

Six local B200s, YOCO-MoE-15B-A6B-L4, PP3/VPP4 `333|333|333|342`,
CP2/EP2/DP1, MBS1, 16K global/8K local tokens, 12 microbatches/update,
BF16/MXFP8, Muon, MTP1, Block recomputation, first-stage frozen Vision
with trainable projector, ref_kl coefficient 10, subset masks and cross-file
packing. There are 20 optimizer updates including 5 warmup updates.
The real fixed text/VL mixture produced 413 images in 240 microbatches;
all six ranks saw identical input hashes and matching RNG initialization.
The frozen reference snapshots the initial 15B student (no compatible
150B reference checkpoint can be used in a 15B test).

Every compared run has exactly equal reported loss, KL loss, MTP loss,
gradient norm, and learning rate on every rank at every step. Full model
and optimizer content fingerprints agree at steps 0, 1, and 20. The frozen
reference buffer fingerprints remain unchanged at steps 1 and 20.
This is stronger than a single-step loss check, but does not include full
per-element gradient dumps or a distributed checkpoint-resume experiment.

## Memory and time

CUDA peaks are the trainer's allocator metrics captured before its reset,
not the observer's post-reset counters. Table uses the maximum allocated
peak across all ranks and steps. Time is rank 0 median train_wall over
steps 6–20; hashing instrumentation occurs outside the timed iteration.

| Change | Peak allocated GiB | Median step seconds |
| --- | ---: | ---: |
| Unchanged baseline | 68.5020 | 8.3113 |
| llm-train forward-only quantization/private-buffer release | 66.5039 | 8.4716 |
| nnscaler non-gradient boundary input release | 67.2364 | 8.3442 |
| Both changes | 65.2383 | 8.3540 |

Combined reduction: 3.264 GiB (4.76%). This measurement demonstrates
lower allocation, **not a throughput speedup**. Forward-only reference
quantization trades repeated quantization for shorter layout lifetimes.
One paired measurement cannot establish small timing differences as a
stable regression/improvement. Reserved/NVML memory differs from allocated;
per-rank values and sampled resource bounds are recorded in the JSONs.

## Lifecycle

The existing paired block already runs the reference under no_grad, computes
a per-layer target and feeds it to student KL. Large reference intermediates
are not all intentionally held for a full reference backward. The changes
remove two other owners: the step-long quantized-weight cache and the
executor's boundary-input list. Original reference weights are unchanged,
and hidden/K/V required by later reference layers and router targets needed
by student backward are retained. The next reference layer needs the preceding
hidden state, so it cannot be discarded before that use.

## Reproduction and limits

`VL_PP_PRESET=ref_kl` in `scripts/vl_pp_debug.sh` exposes the production-aligned
local defaults. Source `scripts/k8s_env.sh` before running it and supply a
readable `DATA_PATH` plus the Vision checkpoint. This preset intentionally
keeps the existing default debug preset unchanged. The preserved `harness/`
files are exact validation instrumentation; their original task root was
`/data/tmp/ref-kl-opt-20261103`. To rerun them elsewhere, copy them under a
scratch `scripts/`, supply the `data/mixture.json` and source shards, and map
its source-data path in run_15b.py. JSON plan fields record the exact original
CLI and deviations. Compressed audits contain hashes/metrics, not data payloads.

The local test matches 8K tokens per CP rank, VPP4, 12 microbatches and major
kernel settings of the 150B/256K/CP32/EP32/DP16 YAML. It does not reproduce
150B expert sizes, pretrained routing, cross-node network cost or DP16. It
uses deterministic attention, smaller prefetch queues and a shorter warmup.
Do not extrapolate its memory savings or step time to the production job.

Generated initial weights/shared input files were removed after successful
runs. Heavy production compile output uses a separate task-owned PVC directory;
resource guards protect host/container memory, GPU usage and free disk space.
