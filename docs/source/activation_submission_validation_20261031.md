# Independent activation submission: validation status

This change separates three mechanisms: local F/B scheduling, forward submission
boundaries/byte targets, and storage DMA chunks. The old defaults retain their
behavior. Scope restore can now use group submission, and accumulated scope saves
can flush at closed regions once a byte target is reached. The store still owns
whole immutable storages and checks versions; DMA chunks do not partition alias
ownership or promise a smaller whole-tensor allocation.

`activation_batch.py` owns forward and restore queues. The graph adapter only
creates scopes/groups and installs backward hooks; the transport only handles
stream dependencies, copy calls and backpressure. Public observation optionally
adds bounded consumer-stream timing events. Without that option, the restore path
adds no timing events or readiness queries. Failure after partial copy submission
drains the affected stream before unfenced host storage can be reused.

## Completed CPU checks

With `scripts/k8s_env.sh` sourced from llm-train and `CUDA_VISIBLE_DEVICES=''`:

- NNScaler: **77 passed, 88 CUDA cases skipped** across `test_activation_batch.py`,
  `test_activation_transfer.py`, `test_activation_schedule.py`,
  `test_activation_store.py`, `test_cpu_offloading.py`, `test_activation_memory.py`.
- llm-train: **283 passed, 8 CUDA cases skipped** across activation configuration,
  CLI parsing, profiler, training memory and lifetime-ledger tests.

The new CPU checks exercise real queue/admission code with controlled transfer
completions: early group submission with scope restore, delayed scope submission
with group restore, bounded batches including oversized storages, dead/budget-
declined saves, weak-reference cleanup with cyclic GC disabled, duplicate aliases,
budget retries and renewed producer events after flush. Byte-buffer tests check
chunk tails, storage offsets and destination aliasing. Timing tests check deferred
completion, nested observers, independent bounds and absence of host synchronization.
Submission-failure tests check that queued work drains before buffers are reused.
Serialized configs with missing new fields preserve ABI 1 and legacy defaults.

Two existing training-memory profiler test doubles used an obsolete constructor;
they now accept the public `stores` argument and continue checking span/stop order.

## Prepared, not yet GPU-validated

Eight new CUDA combinations cover scope/group restoration × scope/group submission
× whole/chunked DMA, including gradients through transposed views, early D2H,
empty pools and exported wait samples. Existing delayed-producer, multiple-stream,
alias/version, PP/VPP and lifetime tests remain required before performance runs.
The scheduling probe exposes `--forward-unit`, `--commit-mib` and
`--copy-chunk-mib`; its separate diagnostic trace now includes dependency intervals.

No GPU task, full-model compilation or parameter sweep was launched for this change.
The October 30 performance report belongs to the previous implementation. New
timing intervals include event overhead and host submission gaps, can overlap on
different streams, and must not be summed into global training stall time.

The next full-model target is the text YOCO-MoE-15B-A6B-L4 PP2×EP4/VPP8 workload,
8 accumulated microbatches, with paired recompute and Block-offload configurations.
Its microbatch/sequence-capacity and matched throughput/memory tables are specified
in llm-train's `docs/activation-offload-validation-plan.md`. Parameters will be
chosen later; this revision contains no model-specific retention exceptions.

Compressed CPU logs and file fingerprints are in
[the evidence manifest](../activation-store-results/20261031/submission/manifest.json).

## Capacity-test preparation follow-up

Restore destination allocation now precedes dependency submission. An allocation
failure has not queued a read of the host buffer and propagates without draining
unrelated restores; a partial DMA failure still drains before buffer reuse.
Whole-storage and chunked copies share the same allocation/copy boundary.
Two CPU failure-injection cases cover this distinction; the runtime suite now
reports **79 passed, 88 CUDA cases skipped**. GPU validation remains pending the
other session's complete handoff. Evidence is in
[the follow-up manifest](../activation-store-results/20261031/allocation-failure/manifest.json).
