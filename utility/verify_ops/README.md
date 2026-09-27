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
