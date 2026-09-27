# Operator partition verification

`verify_graph_operations.py` checks that every partition allowed by an
operator annotation preserves the single-device result and input gradients.
It currently tests two-way partitioning along one dimension at a time.

```bash
python utility/verify_ops/verify_graph_operations.py \
  --graph /path/to/graph.ckp \
  --outdir /path/to/verification-cache
```

The command exits nonzero when a partition differs from the reference or
cannot be verified.

The same checks can be requested while compiling:

```python
parallelize(
    model,
    dummy_forward_args,
    policy,
    compute_config,
    verify_annotations="static",  # "off", "static", "used", or "all"
)
```

`off` is the default. `static` checks partition feasibility without executing
distributed code. `used` dynamically verifies only partitions selected by the
policy, while `all` dynamically verifies every feasible two-way partition.
Dynamic verification must run before `torch.distributed` is initialized, for
example in an AOT compile step with `load_module=False`.

Stateful custom operators can register importable, zero-argument lifecycle
callbacks:

```python
@nnscaler.register_op(
    "l n -> l",
    verify_setup_fn=reset_runtime_state,
    verify_state_fn=snapshot_runtime_state,
)
def stateful_op(x):
    ...
```

`verify_setup_fn` runs before the reference and distributed executions.
`verify_state_fn` returns the full logical state after execution. The verifier
requires that state to match the reference on every rank, so this contract is
for replicated state. Operators with intentionally sharded state need a
specialized verifier or must remain unpartitioned.
