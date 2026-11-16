# Concurrent module streams

`ComputeConfig.module_streams` schedules independent module branches on named
CUDA streams inside a compiled segment. Model `forward` code does not need to
change. PyTorch autograd executes each backward operator on its forward stream
and synchronizes tensor dependencies between streams.

```python
from nnscaler import ComputeConfig, ModuleStreamRule

config = ComputeConfig(
    plan_ngpus=8,
    runtime_ngpus=8,
    module_streams=(ModuleStreamRule(
        module='*.shared',
        stream='shared_expert',
        launch_before='*.dispatch',
        split_before='down_proj',
        resume_before='*.combine',
    ),),
)
```

The same configuration is available to existing launchers through a JSON
environment variable. Its value is resolved when `ComputeConfig` is constructed
and stored in the serialized configuration. Changing it invalidates generated
code reuse, while allowing reuse of the unmodified traced graph.

```bash
export NNSCALER_MODULE_STREAMS='[{"module":"*.shared","stream":"shared_expert","launch_before":"*.nnscaler_moe_dispatch","split_before":"down_proj","resume_before":"*.nnscaler_moe_combine"}]'
```

`module` matches qualified module paths (including all descendant operators).
`launch_before` matches an operator signature in the enclosing module invocation.
Repeated invocations of the same module, such as a universal transformer loop,
are scheduled separately. `split_before` names a child module relative to the
selected module; the suffix starts before `resume_before`. Omit both split fields
to launch the whole branch. Without a launch anchor, a branch stays in its
original position and can overlap subsequent independent modules.

The compiler checks tensor and object dependencies before moving computation.
It preserves complete recompute regions and rejects splits through their
boundaries, ambiguous anchors and known mutations/random operations. Rules are
local to each pipeline segment and never move operations between stages. ZeRO
0/1 is supported. Selecting a module is an assertion that it and its custom
operators are pure: hidden mutable state, external side effects, RNG use inside
custom operators, and unrepresented alias mutations are not safe to reorder.

Each fork waits on the caller's submitted work. The caller waits on a region's
completion event only at the first consumer, and drains remaining events when
leaving the segment. Storage is recorded on consuming streams; opaque owners
are retained through completion. These operations do not synchronize the host.
Streams are cached per device and name; completion events are per invocation,
so pipeline microbatches cannot overwrite another invocation's dependency.

For MoE, launching shared-expert up/gate projections before dispatch and its
down projection before combine can cover communication with independent GEMMs.
Launching before dispatch also makes computation available while a dynamic
DeepEP dispatch obtains its receive counts on the host. This follows the shared
expert schedule in [Kimi K3, Figure 11 and Section 5.2.1](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf).
Kimi's MoonEP additionally uses balanced static shapes; module scheduling alone
does not remove DeepEP's receive-count host synchronization.

Performance must be measured with both unprofiled training wall time and GPU
timelines. Sum interval unions per GPU instead of summing kernel durations across
streams. Report the baseline dispatch/combine fraction, communication with no
concurrent useful computation, and the achieved step-time reduction. Concurrent
kernels can contend for SMs and bandwidth; timeline overlap alone does not
establish that all communication latency has been hidden.
