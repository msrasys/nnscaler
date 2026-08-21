# Extending Parallelism Within and Across Scale Units

This guide explains two useful parallel patterns that extend nnScaler's normal plan-level parallelism:

1. **Simulated data parallelism inside one scale unit**: most of the model runs with tensor parallelism (TP), while a slow, non-partitionable module is replicated across the ranks in each scale unit. To avoid redundant computation, its input batch is sharded across those ranks, and the outputs are gathered before the model resumes tensor-parallel execution. (We also support dp cross multiple scale units, but that looks not very useful in practice, so we do not document it here.)

2. **Context parallelism across scale units with expert parallelism inside each plan**: `plan_ngpus` remains small enough to partition expert weights sensibly, while several scale units cooperate on one long sequence.

Runnable references:

- `tests/cli/test_simulated_dp.py`
- `tests/cli/trainer_args_simulated_dp.yaml`
- `tests/cli/trainer_args_simulated_dp_dp_sharded.yaml`
- `tests/cli/test_cp_ep.py`
- `tests/cli/trainer_args_cp_ep.yaml`

This document assumes basic PyTorch knowledge. It explains the nnScaler-specific pieces from the beginning: rank topology, dataset sampling, trace inputs, custom Autograd boundaries, policies, parameter reducers, and validation.

## 1. Concepts You Need First

### 1.1 `plan_ngpus`, `runtime_ngpus`, and scale units

`plan_ngpus` is the number of GPUs described by one compiled nnScaler plan. `runtime_ngpus` must be divisible by `plan_ngpus`. nnScaler then repeats the plan across scale units:

```text
num_scale_units = runtime_ngpus // plan_ngpus
```

For example:

```yaml
compute_config:
  plan_ngpus: 2
  runtime_ngpus: 4
```

produces this layout:

| Scale unit | Global ranks |
|---|---|
| 0 | `[0, 1]` |
| 1 | `[2, 3]` |

The same two-device plan runs in both units. Plan device 0 expands to global ranks `[0, 2]`, and plan device 1 expands to global ranks `[1, 3]`.

## 2. Pattern One: Simulated DP Inside a Scale Unit

### 2.1 When to use it

Assume most of a model works well with TP, but one submodule is unsuitable for partitioning because it has dynamic shapes, complex control flow, or an unsupported implementation. Replicating that submodule normally makes every rank in a scale unit repeat the same expensive work.

In this pattern, custom Autograd Functions shard and gather the batch only around the slow module. The nnScaler policy continues to describe TP partitioning for the surrounding modules and leaves the slow custom operator replicated.

The desired model flow is:

```text
TP pre-module
    -> split batch inside the scale unit
    -> opaque slow module, one batch shard per rank
    -> all-gather batch inside the scale unit
    -> TP post-module
```

The slow module remains fully replicated. Only its input data is divided.

### 2.2 What each rank reads from the dataloader

Trainer constructs its default sampler with:

```text
num_replicas = runtime_ngpus // plan_ngpus
sampler_rank = global_rank // plan_ngpus
```

For `plan_ngpus=2` and `runtime_ngpus=4`, this distributes data by scale unit:

| Rank | Scale unit | Dataloader microbatch | Slow-module shard |
|---|---|---|---|
| 0 | 0 | A | First half of A |
| 1 | 0 | A | Second half of A |
| 2 | 1 | B | First half of B |
| 3 | 1 | B | Second half of B |

Important properties:

- ranks 0 and 1 use the same sampler stream and therefore receive the same sample indices and batch contents;
- ranks 2 and 3 use another shared sampler stream;
- A and B are different portions of the global training batch;
- no custom sampler is required for this pattern.

In the reference test, `shuffle: false`, dataset size is 8, and `micro_batch_size` is 4. The first step therefore maps samples `[0,1,2,3]` to ranks 0/1 and samples `[4,5,6,7]` to ranks 2/3. With shuffling enabled, rely on sampler-stream equality rather than fixed indices.

For example:

```yaml
micro_batch_size: 4
global_batch_size: 8
```

Each scale unit receives four samples. After the manual split, each slow-module replica processes two samples.

Trainer uses this consistency rule:

```text
global_batch_size
    = micro_batch_size
    * (runtime_ngpus / plan_ngpus)
    * grad_accumulation_steps
```

### 2.3 What the trace input must look like

Trace the original model interface using the **complete microbatch seen by one scale unit**:

```text
[micro_batch_size, sequence, hidden] = [4, S, H]
```

Do not trace with the final rank-local shape `[2, S, H]`.

The reasons are:

1. the original module receives a complete scale-unit microbatch;
2. the custom chunk operator's fake function describes the rank-local output `[2, S, H]`;
3. nnScaler needs both layouts to construct correct forward and backward IR.

This custom boundary performs an equal `torch.chunk`, so the traced batch dimension must be divisible by the chunk group size. This is a requirement of this implementation, not a general nnScaler tracing rule.

#### Variant: the slow module is first

If the slow module is the first model operation, the dataloader can provide its batch shards directly. In the reference `dp_sharded` mode:

- every rank receives a disjoint slice of its scale unit's sampler batch;
- `micro_batch_size=4` remains the logical batch seen by one scale unit;
- each runtime dataloader yields `4 / plan_ngpus = 2` samples per rank, and gathering those slices reconstructs the original scale-unit batch in order;
- `dummy_sample_gen_fn` supplies the same rank-local shape `[2,S,H]` for tracing;
- the slow module therefore traces and executes with `[2,S,H]`;
- a scale-unit all-gather after the slow module restores `[4,S,H]` before TP begins.

No entry collective or runtime chunk is needed. The exit wrapper uses the standard all-gather `fake_fn`, which expands the traced rank-local batch from `[2,S,H]` to the logical `[4,S,H]` shape.

The reducer settings are otherwise the same as the standard pattern: enable `reducer_replicated_params`, isolate the slow parameters with `param_clss_fn`, and set their bucket's `reducer_nreplicas=1`. The reference test verifies that these parameters use the all-rank reducer while preserving the sum of complementary rank-local gradients.

### 2.4 Create the scale-unit communication groups

The custom operators call collectives directly. Create every required process group on every rank, in the same order, before model execution:

```python
from nnscaler.runtime.device import DeviceGroup


def init_scale_unit_groups(trainer):
    cfg = trainer.train_args.compute_config
    group_size = cfg.plan_ngpus
    world_size = torch.distributed.get_world_size()

    if world_size != cfg.runtime_ngpus:
        raise ValueError("world size and runtime_ngpus do not match")
    if world_size % group_size:
        raise ValueError("runtime_ngpus must be divisible by plan_ngpus")

    for first_rank in range(0, world_size, group_size):
        ranks = tuple(range(first_rank, first_rank + group_size))
        DeviceGroup().get_group(ranks)
```

Reference it from Trainer YAML:

```yaml
init_env_fn: your_module.init_scale_unit_groups
```

For `plan=2, runtime=4`, this creates `[0, 1]` and `[2, 3]`.

Note: generally nnscaler has already created the plan-level process groups for TP. We have this manual initializer just for safety.

### 2.5 Find the current scale-unit ranks

```python
def scale_unit_ranks(group_size):
    rank = torch.distributed.get_rank()
    first_rank = rank // group_size * group_size
    return tuple(range(first_rank, first_rank + group_size))
```

Examples:

```text
rank 0 or 1 -> [0, 1]
rank 2 or 3 -> [2, 3]
```

### 2.6 Entry boundary: chunk forward, all-gather backward

```python
class ScaleUnitChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group_size):
        ctx.ranks = scale_unit_ranks(group_size)
        return chunk(x, dim=0, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad, dim=0, ranks=ctx.ranks), None
```

Forward behavior for microbatch A:

```text
rank 0: A[0:2]
rank 1: A[2:4]
```

Backward must all-gather because the pre-module consumed the complete batch before the boundary. It therefore expects the complete input gradient.

### 2.7 Exit boundary: all-gather forward, chunk backward

```python
class ScaleUnitAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group_size):
        ctx.ranks = scale_unit_ranks(group_size)
        return all_gather(x, dim=0, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad):
        return chunk(grad, dim=0, ranks=ctx.ranks), None
```

Forward gathers the slow-module outputs back into the complete batch.

The registered output of `scale_unit_all_gather` is a replicated full tensor. nnScaler therefore adapts every downstream gradient back to that full layout before invoking `ScaleUnitAllGather.backward`. Once its preconditions hold, the backward `chunk` is correct regardless of the downstream TP layout.

For example, the reference post linear is partitioned over output features. Each TP rank computes only one contribution to the gradient of the replicated linear input:

```text
full input gradient = sum(rank-local input-gradient contributions)
```

nnScaler consequently inserts `identity_allreduce`: its forward is identity, and its backward performs that sum. `ScaleUnitAllGather.backward` then receives the complete gradient and selects the batch shard belonging to the current rank. If the post linear is replicated instead, its local input gradient is already complete and nnScaler inserts no adapter at this boundary. Other TP layouts may require a different adapter, but the gradient delivered to `ScaleUnitAllGather.backward` must still match its replicated full output.

Do not manually add another all-reduce at this boundary. That would reduce the gradient twice in layouts where nnScaler already generated the reduction.

`ScaleUnitAllGather` itself is not valid for uneven or ragged batch shards. For example, suppose data-dependent filtering leaves rank 0 with output shape `[1,S,H]` and rank 1 with `[2,S,H]`. The runtime `all_gather` expects equal local shapes, the registered annotation assumes an output batch of `group_size * b`, and backward cannot recover the shards with equal `chunk`. Such a module needs a variable-size gather with saved per-rank offsets, plus a backward operation that slices using those offsets.

### 2.8 Register communication wrappers with fake functions

Distributed communication cannot run during tracing. Register plain wrapper functions and provide lightweight `fake_fn` implementations:

```python
def fake_chunk(x, group_size):
    if x.shape[0] % group_size:
        raise ValueError("batch size must be divisible by group_size")
    return x.chunk(group_size, dim=0)[0]


def fake_all_gather(x, group_size):
    return torch.cat([x] * group_size, dim=0)


@nnscaler.register_op(
    '(group_size b) s^ h^ -> b s^ h^',
    fake_fn=fake_chunk,
)
def scale_unit_chunk(x, group_size):
    return ScaleUnitChunk.apply(x, group_size)


@nnscaler.register_op(
    'b s^ h^ -> (group_size b) s^ h^',
    fake_fn=fake_all_gather,
)
def scale_unit_all_gather(x, group_size):
    return ScaleUnitAllGather.apply(x, group_size)
```

The fake function need not reproduce runtime values or communication, but it must preserve output metadata:

- output shape;
- output dtype;
- output `requires_grad` state.

If the activation input already requires gradients, cloning it preserves the required output metadata:

```python
def fake_slow_block(x, *weights):
    return x.clone()
```

At model entry, dataloader tensors normally do not require gradients. For a registered opaque operation with trainable weight inputs, return a fresh tensor whose `requires_grad` flag is set when it is created:

```python
def fake_slow_block(x, *weights):
    return torch.randn_like(x, requires_grad=True)
```

The fake values themselves are irrelevant. Because the weights are explicit registered-op inputs, marking the output differentiable is sufficient for nnScaler to create their gradient metadata and reducers. Do not set `requires_grad=True` on an existing non-leaf `x.clone()` result; PyTorch rejects changing that flag on non-leaf tensors. If the fake output has `requires_grad=False`, backward IR stops at the opaque operator and the required reducers or activation-gradient adapters are not generated.

Register a module-level callable wrapper. If the operation uses module parameters, pass their tensors explicitly as operator arguments; the registered callable itself is not a parameter-owning `nn.Module`. See `tests/cli/test_simulated_dp.py`.

### 2.9 Place the boundaries around only the slow region

```python
def forward(self, data):
    x = self.pre(data['data'])

    x = scale_unit_chunk(x, group_size=self.scale_unit_size)
    x = self.dynamic_block(x)
    x = scale_unit_all_gather(x, group_size=self.scale_unit_size)

    output = self.post(x)
    return loss_fn(output, data['target'])
```

The pre- and post-modules remain under the normal TP policy. Only the opaque slow region is enclosed by the manual boundaries.

### 2.10 Do not partition the slow module in the policy

The slow module must remain opaque and fully replicated. A policy can still partition the surrounding modules:

```python
def policy(graph, cfg):
    for node in get_pas_ops(graph):
        if ProjectionModule not in node.module_class_chain:
            continue
        if node.fn == torch.nn.functional.linear:
            yield OpPlan(node, partition=OpPartition(input=1, dim=0))
        else:
            yield OpPlan(node, partition='auto')
```

Do not return a partitioned `OpPlan` for the slow custom operator.

### 2.11 Configure slow-module parameter gradients

The slow-module parameters are complete replicas on every rank.

In ordinary replicated execution, ranks inside a scale unit see the same batch, so their parameter gradients are duplicates. In simulated DP, those ranks see complementary batch shards, so their gradients are complementary contributions that must be summed.

Enable reducers for replicated parameters:

```yaml
compute_config:
  reducer_replicated_params: true
```

Classify all slow-module parameters into a separate bucket and override its post-reduction divisor:

```python
from nnscaler.runtime.adapter.reducer import ParamBucketConfig


def param_clss_fn(parameter_fqn):
    if parameter_fqn.startswith('dynamic_block.'):
        return ParamBucketConfig(reducer_nreplicas=1)
    return ParamBucketConfig()
```

`reducer_nreplicas` is the divisor applied after the bucket's gradient collective. With a SUM reducer:

- `1` preserves the sum of complementary batch-shard gradients;
- `plan_ngpus` averages duplicate gradients from normal replicated execution.

It does not change reducer ranks or collective type. Values greater than one require SUM reduction semantics. Distinct resolved `ParamBucketConfig` values force parameters into different buckets, preventing normal replicated parameters from sharing the simulated-DP divisor.

Trainer YAML:

```yaml
optimizer:
  type: torch.optim.Adam
  param_clss_fn: your_module.param_clss_fn
  args:
    lr: 0.001
```

### 2.12 Minimal Trainer configuration

The model, policy, group initializer, and parameter classifier must be importable by fully qualified name:

```yaml
compute_config:
    plan_ngpus: 2
    runtime_ngpus: 4
    use_end2end: true
    reducer_replicated_params: true

run_mode: run
pas_policy: your_package.simulated_dp_policy
micro_batch_size: 4
global_batch_size: 8       # 4 * (4 / 2), with one accumulation step
init_env_fn: your_package.init_scale_unit_groups

model:
    type: your_package.Model
    args:
        scale_unit_size: $(compute_config.plan_ngpus)

optimizer:
    type: torch.optim.Adam
    param_clss_fn: your_package.simulated_dp_param_clss_fn
    args:
        lr: 0.001
```

Dataset and dataloader settings remain model-specific. The default Trainer sampler is intentional for this pattern.

### 2.13 Simulated-DP checklist

- [ ] Ranks in one scale unit receive the same sampler indices and batch contents.
- [ ] Different scale units receive different microbatches by default.
- [ ] Trace uses the complete scale-unit microbatch.
- [ ] Batch size is divisible by `plan_ngpus`.
- [ ] The slow module is not partitioned by the policy.
- [ ] Every fake output preserves `requires_grad`.
- [ ] Slow-module parameters are complete replicas.
- [ ] Slow parameters occupy buckets with `reducer_nreplicas=1`.
- [ ] Generated forward/backward code has the expected adapter direction and communication ranks between the output gather and downstream TP.
- [ ] Multi-step model and optimizer checkpoints match an unsharded baseline within an appropriate floating-point tolerance.

## 3. Pattern Two: CP Across Scale Units with EP Inside Each Plan

### 3.1 Target topology

In this pattern, custom Autograd Functions split and gather sequence shards across scale units. Within each plan, `TransformRule` and `OpPlan` describe the paired sequence and expert-weight partition used by the EP operator.

The reference files exercise two related topologies:

| Validation | `plan_ngpus` | `context_parallel_size` | `runtime_ngpus` | CP groups |
|---|---:|---:|---:|---|
| Live distributed test | 2 | 4 | 4 | `[0,1,2,3]` |
| Live distributed test (when eight GPUs are available) | 2 | 4 | 8 | `[0,1,2,3]`, `[4,5,6,7]` |
| CPU static compile test | 2 | 4 | 8 | `[0,1,2,3]`, `[4,5,6,7]` |

The eight-rank topology is the more complete example:

```text
EP = plan_ngpus = 2
CP = 4
runtime_ngpus = 8
```

CP and EP reuse ranks in this design. This is not an orthogonal `CP x EP = 8` mesh. `context_parallel_size` means the number of ranks that cooperate on one logical input, independent of the total runtime world size.

Every four contiguous ranks cooperate on one input:

| CP group | Input | EP units |
|---|---|---|
| `[0, 1, 2, 3]` | A | `[0,1]`, `[2,3]` |
| `[4, 5, 6, 7]` | B | `[4,5]`, `[6,7]` |

Within each CP group, the sequence is ultimately split four ways:

| Rank | Sequence shard | Expert shard |
|---|---|---|
| 0 | First quarter of A | experts `[0:2]` |
| 1 | Second quarter of A | experts `[2:4]` |
| 2 | Third quarter of A | experts `[0:2]` |
| 3 | Fourth quarter of A | experts `[2:4]` |
| 4 | First quarter of B | experts `[0:2]` |
| 5 | Second quarter of B | experts `[2:4]` |
| 6 | Third quarter of B | experts `[0:2]` |
| 7 | Fourth quarter of B | experts `[2:4]` |

Weights are partitioned only two ways by EP, while sequence activations are partitioned four ways by CP. This prevents expert weights from being split as finely as `runtime_ngpus`.

The live `runtime_ngpus=4` test has only the first CP group. All four ranks cooperate on one input stream. The eight-rank topology is always checked by a CPU static compile test that verifies both independent CP groups, generated shapes, expert slices, and reducer groups. When eight GPUs are available, a separate live test executes its collectives and compares training checkpoints with the unsharded baseline.

### 3.2 What each rank reads from the dataset

Trainer passes a scale-unit rank to the sampler:

```text
scale_unit_rank = global_rank // plan_ngpus
```

For `runtime=8, plan=2`, the sampler sees scale-unit ranks 0, 1, 2, and 3.

A custom sampler must implement this mapping for the eight-rank topology:

```text
scale units 0 and 1 -> input A
scale units 2 and 3 -> input B
```

Therefore:

```text
global ranks [0, 4) -> A
global ranks [4, 8) -> B
```

The mapping can be derived as follows:

```python
scale_units_per_cp_group = context_parallel_size // expert_parallel_size
num_cp_groups = num_scale_units // scale_units_per_cp_group
cp_group_rank = scale_unit_rank // scale_units_per_cp_group
```

A practical sampler wraps `DistributedSampler`:

```python
self.sampler = torch.utils.data.DistributedSampler(
    dataset,
    num_replicas=num_cp_groups,
    rank=cp_group_rank,
    shuffle=shuffle,
)
```

Do not use Trainer's default sampler for this pattern. The default sampler gives every scale unit different data, so two units cannot cooperate on the same long sequence.

For the live `runtime=4` test there are only two scale units, both in the same CP group. The inner `DistributedSampler` therefore has one replica, and both scale units read the same sample stream. For `runtime=8`, scale units 0/1 map to CP sampler replica 0 and scale units 2/3 map to replica 1.

### 3.3 What the trace input must look like

Trace with the complete sequence seen before the outer CP boundary:

```text
[micro_batch_size, full_sequence_length, hidden]
```

For example:

```text
[2, 128, 16]
```

Do not trace with the final rank-local `[2, 32, 16]` shape.

The shape transition is:

```text
Trace input                              [B, 128, H]
Outer cross-scale chunk                  [B,  64, H]
EP TransformRule sequence partition      [B,  32, H]
```

The outer fake chunk divides by `CP / EP`, which is the number of scale units cooperating on one CP input. The plan-level EP transform then divides by `EP`. Together they produce `S / CP`.

### 3.4 Required process groups

For `runtime=8, CP=4, EP=2`:

These are four distinct communication meshes. Do not use one mesh as a substitute for another.

#### EP all-to-all groups

```text
[0,1], [2,3], [4,5], [6,7]
```

These groups never cross a scale-unit boundary.

#### Complete CP context groups

Used by attention or other context-dependent collectives:

```text
[0,1,2,3], [4,5,6,7]
```

#### Outer cross-unit sequence groups

Used by entry chunk and exit all-gather:

```text
[0,2], [1,3], [4,6], [5,7]
```

Each group contains the same EP lane from every scale unit participating in one CP input.

#### Reducer groups for identical expert shards

Generated by nnScaler's scale reducer:

```text
experts [0:2] -> [0,2,4,6]
experts [2:4] -> [1,3,5,7]
```

These span all data replicas that own the same expert-weight shard. They are generated by nnScaler's reducer setup; they are not created by the manual group-initialization loop used for the custom collectives.

Every manually created process group must be created by all ranks in the same deterministic order.

A minimal initializer is:

```python
def init_cp_ep_groups(trainer):
    cfg = trainer.train_args.compute_config
    ep_size = cfg.plan_ngpus
    cp_size = trainer.train_args.model.args['context_parallel_size']
    world_size = torch.distributed.get_world_size()

    if world_size % cp_size:
        raise ValueError("runtime_ngpus must be divisible by CP size")
    if cp_size % ep_size:
        raise ValueError("CP size must be divisible by EP size")

    # Contiguous plan-sized groups for EP dispatch/combine.
    for first_rank in range(0, world_size, ep_size):
        DeviceGroup().get_group(
            tuple(range(first_rank, first_rank + ep_size))
        )

    for first_rank in range(0, world_size, cp_size):
        # Complete CP group for attention/context communication.
        DeviceGroup().get_group(
            tuple(range(first_rank, first_rank + cp_size))
        )

        # Same-EP-lane groups for outer sequence chunk/gather.
        for ep_lane in range(ep_size):
            DeviceGroup().get_group(
                tuple(range(
                    first_rank + ep_lane,
                    first_rank + cp_size,
                    ep_size,
                ))
            )
```

nnScaler initializes reducer groups separately from the generated weight layout.

### 3.5 Outer cross-unit sequence boundaries

The entry boundary does not immediately split on the complete four-rank CP group. It first splits between scale units at the same EP lane:

```text
[0,2]: input A -> rank 0 first half, rank 2 second half
[1,3]: input A -> rank 1 first half, rank 3 second half
```

The EP policy then partitions each half again inside `[0,1]` and `[2,3]`, yielding four sequence quarters.

The custom Autograd semantics are:

```text
Entry: forward chunk(sequence), backward all-gather(sequence)
Exit:  forward all-gather(sequence), backward chunk(sequence)
```

In the reference policy, nnScaler first reconciles the plan-level EP output layout for the scale-unit-local consumer. The manual all-gather then combines the two scale-unit halves into the complete sequence. The exact nnScaler adapter is layout-dependent; inspect generated code if the surrounding policy changes.

### 3.6 The custom EP operator must partition sequence and experts together

Register the runtime operator with this transform rule:

```python
rule = TransformRule(
    [DimopSplit.D(1), DimopSplit.D(0)],
    [DimopSplit.D(1)],
)
```

| Tensor | Rule | Result |
|---|---|---|
| Activation `[B,S,H]` | `D(1)` | Partition sequence across EP ranks |
| Expert weight `[E,O,H]` | `D(0)` | Partition experts across the same ranks |
| Output `[B,S,O]` | `D(1)` | Preserve the corresponding sequence shard |

Selecting expert-weight dimension 0 activates the complete transform rule:

```python
def policy(graph, cfg):
    for node in get_pas_ops(graph):
        if node.fn == routed_expert:
            yield OpPlan(node, partition=OpPartition(input=1, dim=0))
```

When the runtime custom operator is called, nnScaler has already partitioned both the activation sequence and expert weights. The operator must not split sequence again.

### 3.7 EP dispatch, local expert computation, and combine

Assume:

```text
EP size = 2
Local experts per rank = 2
Rank-local sequence length = 32
```

The toy router interprets the 32 tokens as four equal routing blocks of eight tokens:

```text
[destination rank 0 / local expert 0 / 8 tokens,
 destination rank 0 / local expert 1 / 8 tokens,
 destination rank 1 / local expert 0 / 8 tokens,
 destination rank 1 / local expert 1 / 8 tokens]
```

The implementation then performs these steps:

1. Interpret the sequence as `[destination_rank, local_expert, tokens, hidden]`.
2. Flatten it back to sequence so each destination rank owns one contiguous message.
3. Run all-to-all inside the local EP group.
4. Interpret received data as `[source_rank, local_expert, tokens, hidden]`.
5. Permute it to `[local_expert, source_rank, tokens, hidden]`.
6. Concatenate source blocks and run each local expert.
7. Reverse the permutation.
8. Run a second all-to-all to return outputs to their source ranks.
9. Recover the original rank-local token-block order.

The test omits a real router. It assumes consecutive token blocks are already ordered by destination expert. A real MoE must:

- compute expert assignments;
- permute tokens by destination expert;
- communicate variable split sizes when routing is uneven;
- perform the inverse permutation after combine.

The `nnscaler.runtime.adapter.nn.alltoall_alltoall` operation used by the test has an autograd implementation whose backward performs the inverse exchange. Do not assume an arbitrary raw all-to-all API is differentiable.

### 3.8 Simulating a globally context-dependent attention operation

`GlobalContextMix` is not an attention implementation. It keeps only the properties relevant to CP testing:

- every local output depends on the complete sequence;
- forward requires communication over one CP group;
- backward requires communication over the same CP group.

It computes the equivalent of:

```text
output = local_x + global_sum(local_x) / (
    full_sequence_length * context_replicas
)
```

The communication group is `[0,1,2,3]` for input A or `[4,5,6,7]` for input B. Independent inputs never communicate with one another.

The non-CP baseline deliberately skips the outer sequence chunk. The custom sampler still gives every scale unit inside the CP group the same complete input. After the plan-level EP sequence split, the two scale units hold duplicate EP shards, so the context sum contains two copies and the denominator uses `context_replicas=2`.

This baseline interpretation depends on the sampler invariant. It is valid in the live `runtime=4` test because both scale units belong to the single CP group and map to the same sampler replica. If the sampler gives those units different data, `context_replicas=2` is incorrect.

Replace this toy operation with ring attention, zigzag attention, or another CP-aware attention implementation in a real model. The outer sequence boundaries, sampler, and reducer design remain applicable.

### 3.9 Why the residual temporarily restores a larger sequence shard

You do not need to add communication for correctness; nnScaler generates the required adapters. This section matters only when optimizing communication or activation memory.

Consider one reference block:

```python
output = x + routed_expert(x, ...)
```

After the outer cross-scale chunk, each scale unit holds half of the sequence, so `x` has shape `[B, 64, H]`. The two branches then use different layouts:

```text
                            residual branch: x [B, 64, H]
                           /                              \
x [B, 64, H]                                                     add [B, 64, H]
                           \                              /
                            expert branch:
                            chunk [B, 64, H] -> [B, 32, H]
                            routed_expert    -> [B, 32, H]
                            all-gather       -> [B, 64, H]
```

The `routed_expert` transform rule applies only to the expert branch. It splits that branch across the two EP ranks, but the residual branch remains `[B, 64, H]`. Because `torch.add` requires matching layouts, nnScaler gathers the expert output back to `[B, 64, H]` before the addition.

The next block repeats the process: its expert branch is split from `64` to `32`, then gathered back to `64` for the residual addition. Thus every `routed_expert` computes on `S/CP=32`, while tensors at residual boundaries use the larger scale-unit-local shape `S/(CP/EP)=64`. The complete global sequence `S=128` is restored only by the final outer all-gather.

This is correct but adds one gather and one split around each residual boundary. To avoid that overhead, the residual path and addition must also use a compatible `S/CP` partition, or the complete block must be represented by one custom operator whose transform rule preserves that partition.

### 3.10 Weight reducer configuration

CP alone is not a reason to override `reducer_nreplicas`. This value is a structural divisor applied after a bucket's gradient collective: it accounts for identical gradient contributions computed more than once. It does not choose the reducer ranks, and it should not be used to implement the global mean over distinct samples or tokens.

Apply the following rule independently to each weight bucket:

- if ranks compute the same weight-shard gradient from the same data and computation, those contributions are duplicates and the post-reduction sum must be divided by the duplicate count;
- if ranks compute contributions from disjoint samples, sequence shards, or routed tokens, those contributions are complementary and must remain summed at this stage;
- if nnScaler's generated reducer already represents the plan-level TP or EP layout correctly, inherit its `nreplicas` value instead of overriding it;
- override the bucket only when manual communication changes the duplicate count in a way that the generated plan cannot express.

The module type is not the deciding factor. A non-attention weight outside the manually sharded CP region is unaffected and should keep its existing reducer configuration. A non-attention MLP or expert weight inside that region may receive complementary token shards, while an attention implementation may synchronize its own weight gradients internally. Inspect the actual gradient ownership and generated reducer rather than applying one setting to all attention or all non-attention parameters.

In the reference CP+EP graph, expert weights are partitioned by the plan. nnScaler already generates SUM reducers with `nreplicas=1` for corresponding expert shards, so the CP run does not need a `param_clss_fn` override. For the eight-rank topology, the reducers are:

```text
experts [0:2] -> ranks [0,2,4,6]
experts [2:4] -> ranks [1,3,5,7]
```

For the live four-rank topology, the corresponding groups are `[0,2]` and `[1,3]`.

The test-only non-CP baseline is different: it deliberately repeats one complete input in every scale unit of a CP group. Only that baseline overrides the generated divisor:

```text
reducer_nreplicas = scale_units_per_cp_group = CP / EP
```

This averages the deliberately duplicated baseline gradients and exists only for correctness comparison. It is not a general CP or EP setting.

### 3.11 Trainer global-batch accounting

Trainer defines:

```text
global_batch_size
    = micro_batch_size
    * (runtime_ngpus / plan_ngpus)
    * grad_accumulation_steps
```

This counts scale-unit microbatches. It does not directly count unique CP inputs.

When the sampler assigns one unique microbatch to every CP group, the number of unique samples represented in one micro-step is:

```text
micro_batch_size * (runtime_ngpus // context_parallel_size)
```

This formula is conditional on the sampler invariant; it is not a replacement for Trainer's global-batch validation.

Several independent settings affect normalization at different stages:

- `loss_reduction` controls aggregation of reported loss outputs;
- `grad_reduction` controls Trainer's gradient multiplier after output aggregation;
- Trainer reducer pre-hooks compensate for its scale-factor reduction path;
- `reducer_nreplicas` divides a reducer bucket after that bucket's collective.

The reference CP run uses `optimizer.grad_reduction: sum` and inherits the generated `nreplicas=1` for EP weight shards. Only its deliberately duplicated non-CP baseline overrides `reducer_nreplicas`.

For a real training job, decide whether the intended loss is a sum or mean over unique samples or tokens. Configure `grad_reduction`, `loss_reduction`, `grad_reduce_divisor`, or a custom `aggregate_outputs_fn` accordingly. Do not assume Trainer's scale-unit count is automatically the desired CP sample denominator.

### 3.12 CP+EP configuration constraints

The following must hold:

```text
runtime_ngpus % context_parallel_size == 0
context_parallel_size % plan_ngpus == 0
num_experts % plan_ngpus == 0
sequence_length % context_parallel_size == 0
```

The fixed-size toy router also requires:

```text
local_sequence % (EP * local_experts) == 0
```

Because `local_experts = num_experts / EP` in this test, this is equivalent to `local_sequence % num_experts == 0`. It is a constraint of the toy equal-block reshape, not a general EP requirement. A real dynamic router should use variable split-size all-to-all and does not need equal token counts per expert.

### 3.13 Minimal Trainer configuration

For the eight-rank topology with two independent four-rank CP groups:

```yaml
vars:
    context_parallel_size: 4

compute_config:
    plan_ngpus: 2           # EP size
    runtime_ngpus: 8
    use_end2end: true

run_mode: run
pas_policy: your_package.cp_ep_policy
micro_batch_size: 2
global_batch_size: 8      # 2 * (8 / 2), with one accumulation step
init_env_fn: your_package.init_cp_ep_groups

model:
    type: your_package.Model
    args:
        expert_parallel_size: $(compute_config.plan_ngpus)
        context_parallel_size: $(vars.context_parallel_size)
        runtime_ngpus: $(compute_config.runtime_ngpus)
        use_context_parallel: true

optimizer:
    type: torch.optim.Adam
    grad_reduction: sum
    args:
        lr: 0.001

dataset_sampler:
    type: your_package.ContextGroupSampler
    train_args:
        context_parallel_size: $(vars.context_parallel_size)
        expert_parallel_size: $(compute_config.plan_ngpus)
    val_args:
        context_parallel_size: $(vars.context_parallel_size)
        expert_parallel_size: $(compute_config.plan_ngpus)
```

`global_batch_size=8` is Trainer's scale-unit accounting value. With this sampler, the number of unique samples in the micro-step is `micro_batch_size * runtime_ngpus / context_parallel_size = 4`. Review Section 3.11 before choosing mean-versus-sum semantics for a production loss.

EP weights are partitioned by the plan, so nnScaler automatically creates cross-scale reducers for matching shards. Neither `param_clss_fn` nor `reducer_replicated_params` is required for those partitioned expert weights in this CP configuration.

### 3.14 CP+EP checklist

- [ ] `plan_ngpus` equals the desired EP weight-partition size.
- [ ] `context_parallel_size` describes ranks per logical input.
- [ ] The sampler replicates data inside a CP group and shards data between CP groups.
- [ ] Trace uses the complete sequence before the outer CP boundary.
- [ ] Outer chunk/gather groups never mix independent CP input groups.
- [ ] The transform rule partitions both sequence and expert dimensions.
- [ ] Runtime EP all-to-all stays inside one scale unit.
- [ ] Context-dependent communication stays inside one complete CP group.
- [ ] Expert-shard reducers span all data replicas owning that shard.
- [ ] Existing reducer divisors are preserved unless manual sharding changes the number of duplicate gradient contributions.
- [ ] Static generated-code checks confirm rank-local sequence and weight shapes.
- [ ] Real multi-GPU execution is tested when enough devices are available.

## 4. Common Errors and Debugging

### 4.1 Process-group deadlock

Symptoms include hanging in `new_group` or at the first collective.

Check that:

- every rank executes the same group-creation loops;
- groups are created in the same order;
- no rank skips initialization because of a conditional branch;
- a communication group does not cross independent CP input groups by mistake.

### 4.2 Trace shape differs from runtime shape

Check that:

- trace input is the complete scale-unit-local input;
- fake functions return the intended rank-local shapes;
- symbolic annotation parameters are passed as integer keyword arguments;
- fake outputs preserve runtime dtype and `requires_grad`.

### 4.3 Gradients are divided too much or too little

Inspect every relevant bucket:

```python
for reducer in model.reducers:
    for bucket in reducer.buckets:
        print(reducer.ranks, bucket.nreplicas)
```

Use these rules:

- repeated computation of identical data: divide by the repetition count;
- complementary shards contributing to one loss: preserve the sum with divisor `1`;
- do not use both a parameter hook and a reducer to all-reduce the same gradient dimension.

### 4.4 Results do not match at `1e-8`

Full-batch and sharded GEMMs may use different float32 accumulation orders. Mathematical equivalence does not imply bitwise equality.

The reference test uses `atol=rtol=1e-6` for multi-step checkpoint comparison. Choose tolerances from a reproducible baseline for the model and dtype being tested; do not use a larger tolerance to hide a systematic scaling error.

### 4.5 Not enough GPUs to validate a topology

The reference static test combines `load_module=False` with `@replace_all_device_with('cpu')` to generate rank-specific modules without loading or running them on GPUs. It then inspects:

- selected custom-operator input and weight shapes captured from the transformed graph;
- expert-weight slices on each rank;
- reducer ranks emitted in generated code;
- the number of generated `gencodeN.py` files;
- sampler mappings for all scale-unit ranks.

`tests/cli/test_cp_ep.py::test_cp4_ep2_runtime8_static` demonstrates this workflow.

A compile-only check cannot validate NCCL communication, collective ordering, runtime buffers, or numerical correctness. Run a real distributed test before production deployment.

## 5. Minimum Validation Procedure

1. **Single-device reference**: verify the original model's forward and backward results.
2. **One-plan baseline**: run only the original TP or EP policy.
3. **Enable manual boundaries**: compare forward outputs and the boundary shapes that matter to the new layout.
4. **Inspect generated code**: verify adapter direction and communication ranks.
5. **Inspect reducers**: verify rank groups, bucket membership, and `nreplicas`.
6. **Train for multiple steps**: compare model and optimizer checkpoints, not only one loss value.
7. **Add focused gradient checks when needed**: the reference tests compare checkpoint state and selected metadata; they do not compare every intermediate or every parameter gradient directly.
8. **Expand runtime topology**: compile statically first if hardware is unavailable, then run on the real topology.

## 6. Choosing a Pattern

Use simulated DP inside a scale unit when:

- only one submodule cannot use TP;
- samples are independent inside that submodule;
- ranks in one unit should process different batch samples.

Use CP across scale units with EP inside the plan when:

- one sequence is too long for the existing plan;
- increasing `plan_ngpus` would partition expert weights too finely;
- several scale units should cooperate on one sequence;
- expert or tensor parallel weight partitioning should remain inside each plan.

Both patterns follow the same principle:

> Manually express only the runtime data layout that lies outside the nnScaler plan. Keep weight and activation partitions that fit inside the plan in the nnScaler policy.
