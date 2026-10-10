# Paralleling a Module

nnScaler can transform a `torch.nn.Module` into a parallel module, which is a specialized version of `torch.nn.Module` capable of running across multiple GPUs or nodes. This process hides the complexity of distributed training and inference from the user.

Currently, we support three kinds of parallelism: data parallelism, tensor parallelism and pipeline parallelism. We can also combine them for better performance.

Data parallelism and tensor parallelism can be supported for any module, but pipeline parallelism is only supported for end2end modules for scheduling reason.

An end2end module is a module which satisfies:
- the first argument of `module.forward` is the data sample, and other arguments should have default value, and should never be used in `module.forward` function.
- the first return value of `module.forward` is the loss (scalar tensor)

The above restrictions are necessary for the pipeline parallelism to work. Of course, you can still use the parallel module without pipeline parallelism for end2end modules.

## Weight initialization

By default, nnScaler saves the original model's weights during compilation and
loads them when creating each parallel model. To avoid this weight-file I/O, use
`full` or `shard`.

### Choose a strategy

Set `ComputeConfig.param_init_strategy` to one of these strings, or use the
equivalent `nnscaler.ParamInitStrategy` constant:

| Strategy | How runtime initialization works |
| --- | --- |
| `"file"` / `FILE` (default) | Load parameters and buffers from `fullmodel.pt.*` and `npbuffer.pt`. |
| `"full"` / `FULL` | Construct the full original model for initial weights. Save non-persistent buffers in `npbuffer.pt` and load them directly on resume. |
| `"shard"` / `SHARD` | Use `__shard__init__` if defined; otherwise capture the constructor and replay only the tensors needed by this rank. |

Choose `full` to avoid `fullmodel.pt.*` while keeping ordinary constructor behavior.
Choose `shard` to reduce runtime initialization memory, with the
[capture limitations](#automatic-capture) described below. Only `shard` avoids both
`fullmodel.pt.*` and `npbuffer.pt`; generated code and metadata are still required.

The parser APIs use a single `save_level` argument with `AttrSaveLevel`
(imported from `nnscaler.graph.parser`):

| Level | Saved attribute files |
| --- | --- |
| `NONE` | None |
| `M` | `dist_param_map.pt` |
| `N` | `npbuffer.pt` |
| `F` | `fullmodel.pt.*`, including parameters and both persistent and non-persistent buffers |
| `MN` | `dist_param_map.pt` and `npbuffer.pt` |
| `ALL` (default) | All three categories |

`parallelize` selects `ALL` for `file`, `MN` for `full`,
and `M` for `shard`.
Construction uses the original class, or the `module_fn` supplied to `parallelize`.

On first initialization, `full` checks that each local non-persistent buffer
matches its saved slice in `npbuffer.pt` bitwise (after value-partition scaling).
A mismatch raises an error without overwriting the model's values. This check is
independent of the CLI replica check and also applies to supplied model instances.
Use `broadcast_strategy="all"` or make this file available on every node yourself;
`no_weights` excludes it.

```python
import nnscaler

config = nnscaler.ComputeConfig(
    plan_ngpus=2,
    runtime_ngpus=2,
    param_init_strategy=nnscaler.ParamInitStrategy.SHARD,
    param_init_seed=1234,
)
```

**An existing model instance takes priority.** For `full` and `shard`,
`parallelize(original_model, ...)` uses that instance's current values, without
reconstructing it, reseeding it, or calling `__shard__init__`. The same applies to
`GeneratedModel(init_module=original_model)`. Hook validation and binding in
`parallelize` are unchanged.

**All strategies still construct a full real model during compilation.**
These options change runtime initialization, not tracing.

### Automatic capture

With `shard` and no user hook, nnScaler records the original constructor's tensor
operations, then replays the tensors needed by this rank. It copies each local
slice into the parallel model before releasing the temporary result.

This is **selective full-tensor initialization**, not direct shard generation:
materializing a tensor still allocates its full shape and any dependencies.
Use a [shard hook](#write-a-shard-initializer) when even one full tensor is too
large. Capture uses per-operation random streams, so its values need not match
eager `full` initialization with the same seed.

#### Supported operations

Ordinary non-mutating operators are captured generically, including arithmetic,
`sin`, `cos`, `tril`, matrix multiplication and reductions. Views and mutations
require explicit support:

| Category | Supported operations |
| --- | --- |
| Factories and sampling | `empty`, `zeros`, `ones`, `full`, `arange`, `linspace`, `rand`, `randn`, `randint`, `randperm`, like/new factories, `normal`, `bernoulli`, `poisson`, `multinomial` |
| Views | `detach`, `alias`, `view`, `_unsafe_view`, `transpose`, `t`, `permute`, `slice`, `select`, `unsqueeze`, `squeeze`, `expand`, `unbind` (including tensor iteration) |
| Copies | `clone`, copying `to`, `copy_` |
| In-place initialization | `fill_`, `zero_`, `uniform_`, `normal_`, `random_`, `bernoulli_`, `exponential_`, `geometric_`, `log_normal_`, `cauchy_`, `erfinv_`, scalar-bound `clamp_` |
| In-place arithmetic | `add_`, `sub_`, `mul_`, `div_` with scalar operands on contiguous tensors or views |
| Whole Python functions | `torch.nn.init.trunc_normal_` on PyTorch 2.12+ |

Except for the last row, the table uses ATen operator names; composite APIs work
only if their underlying operations are supported.

PyTorch 2.12+ skips `trunc_normal_` on meta tensors. nnScaler captures its Python
helper as one seeded, in-place operation and replays the installed PyTorch
implementation on CPU. PyTorch 2.0-2.11 retains operator-level capture.
The helper is patched only inside the capture context and restored on exit,
including exceptional exits; nested contexts restore the enclosing patch.
Internally, `PyFunction` supplies the same schema and tags as dispatched operators.
It uses the existing supported operation names to select the op, write or view
path, including their existing mutation restrictions and dependency handling.
Each dispatch mode receives the `PyFunction` object with that mode temporarily
popped. Calling `func(...)` forwards to the next lower mode; once the stack is
empty, the function executes (or returns the meta target for a write).
This is explicit internal registration, not interception of arbitrary Python functions.

Important limits:

- `out=` overloads, noncontiguous mutations, tensor-operand in-place arithmetic,
  and unlisted alias operations such as `split` and `diagonal` are unsupported.
- Data-dependent operations such as `.item()` and `nonzero`, and operations
  without a meta implementation, use a logged CPU fallback during capture.
  This can allocate full tensors. Recipes are replayed later; constructor
  branches are decided during capture, not reevaluated as dynamic branches.
- Replay requires CPU implementations. Custom operators must accurately declare
  mutation, aliasing and randomness; external side effects are not replay-safe.
  For seeded custom operators, replay seeds Python, NumPy, PyTorch CPU and the
  current CUDA device from the captured node seed, restoring caller RNG states
  afterward. NumPy uses the seed modulo `2**32`.
- Tensor literals and external tensor data can retain real storage. Keep external
  inputs unchanged until initialization finishes, and initialize every value
  before reading it; `empty` contents remain unspecified.
- Explicit random generators contribute their initial seed, not their current
  state or device. Avoid concurrent initialization with other users of
  process-global RNG or default-dtype settings.

The following `torch.nn.init` functions fail automatic capture with PyTorch
**2.10.0+cu128**, even for ordinary contiguous floating-point parameters:

| Function | Unsupported underlying operation |
| --- | --- |
| `eye_` | `torch.eye(..., out=tensor)` uses an `out=` overload. |
| `orthogonal_` | The wide-matrix path uses the metadata mutation `t_()`; the tall-matrix path fails at `q *= ph`, a noncontiguous, tensor-operand in-place multiplication after QR. |
| `sparse_` | Indexed assignment uses `aten.index_put_`. |

Their deprecated aliases without the trailing underscore have the same
limitations. Noncontiguous initialization targets are also unsupported, even
for otherwise supported initializers such as `uniform_` and `normal_`.

If capture cannot handle a constructor, use `full`, `file`, a supplied initialized
instance, or a shard hook. Capture does not silently reconstruct a full eager
model on failure.

#### Compatibility snapshot

Constructor capture/replay was tested with PyTorch **2.10.0+cu128**,
Transformers **4.57.6**, and torchvision **0.25.0**:

| Models | Configuration | Result |
| --- | --- | --- |
| torchvision ResNet-18 / ResNet-50 | Standard models, `weights=None` | Passed |
| HF `GPT2LMHeadModel` / `LlamaForCausalLM` | One layer, hidden size 16, two heads, vocabulary 32, sequence capacity 16 | Passed, including causal-mask / RoPE buffers |
| HF `ViTModel` | One layer, hidden size 16, two heads, image size 16, patch size 8 | Passed, including truncated-normal initialization |

These probes checked all parameter/buffer shapes, dtypes, finite values, seeded
repeatability, eager buffer equality and CPU RNG restoration. They do **not**
certify pretrained loading, full distributed execution, other configurations or
equality with eager random weights.

### Write a shard initializer

Define `__shard__init__` on the original model class to replace automatic capture.
It must be a staticmethod or classmethod callable with one positional metadata
map. Additional optional arguments are allowed.

Signature (shown as a staticmethod):

```python
from typing import Dict, Iterable, Tuple, Union
import torch
from nnscaler.runtime.module import AttrMeta

@staticmethod
def __shard__init__(
    attr_meta_map: Dict[str, AttrMeta],
) -> Union[Iterable[Tuple[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    ...
```

The map contains only the parameters and buffers requested by this rank.
Its keys are **generated attribute names**; each `AttrMeta` provides `orig_name`,
the full `shape`, local `slicers`, `sub_shape`, `dtype` and `val_chunks`.

For example, this hook computes an `arange` weight's local slice directly from
global coordinates, without allocating the full weight:

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(64.0).reshape(8, 8))

    @staticmethod
    def __shard__init__(attr_meta_map):
        for name, meta in attr_meta_map.items():
            if meta.orig_name != "weight":
                raise ValueError(f"Unexpected attribute: {meta.orig_name}")
            row_slice, col_slice = meta.slicers
            rows = torch.arange(*row_slice.indices(meta.shape[0]), dtype=meta.dtype)
            cols = torch.arange(*col_slice.indices(meta.shape[1]), dtype=meta.dtype)
            value = rows[:, None] * meta.shape[1] + cols[None, :]
            assert value.shape == meta.sub_shape
            yield name, value

    def forward(self, x):
        return x @ self.weight
```

The hook must follow three rules:

1. **Return exactly the requested tensors.** Yield each `(name, tensor)` once,
   with real data of `meta.sub_shape` and `meta.dtype`. Yield values **before**
   division by `val_chunks`; nnScaler applies that scaling. Invalid outputs raise
   errors rather than falling back to capture.
2. **Stream to save memory.** nnScaler copies each tensor before advancing the
   iterator, so reusable scratch-buffer views are safe. A dictionary return is
   also accepted, but retains all its tensors until consumed.
3. **Make values independent of the requested subset and order.** Pipeline ranks
   and checkpoint resume can request different maps. For random initialization,
   use stable per-attribute seeds and global element coordinates, not one random
   stream consumed in map order or Python's randomized `hash()`. Replicas and
   overlapping slices must agree; values need not match the eager constructor.

### Seeds and replica checking

`param_init_seed` defaults to `1234` and must be an integer in `[0, 2**32)`.
It is independent of the CLI training `seed`. Constructor and hook execution,
including generator iteration, runs in an isolated seed scope covering Python,
NumPy, PyTorch CPU and the current CUDA device. Caller RNG states are restored,
including on failure. Other CUDA devices and custom generators are not seeded.
Constructors and hooks must still be deterministic and rank-independent.

The CLI trainer enables `debug.param_init_check` by default to compare hashes of
replicated weights with **bitwise equality**. This is not a check of overlapping
but different shards, nor is it run by direct `parallelize` calls. See
[Trainer Debug Config](./trainer.md#debug-config) for its scope and limitations,
and [Trainer Compute Config](./trainer.md#compute-config) for a YAML example.

### Loading, resume and generated-code reuse

- **Loading:** Use `parallelize` to attach the original factory or shard hook to
  a generated class, or pass `init_module` when constructing it. `full` and
  automatic capture require tensors to be available by their original attribute
  names, including tensor attributes promoted to buffers by tracing. For
  forward-local or global constants synthesized by the tracer, use `file` or
  supply them through a shard hook.
- **Checkpoint resume:** `init_params=False` initializes only non-persistent
  buffers, leaving parameters and persistent buffers for checkpoint loading.
  Both `file` and `full` read `npbuffer.pt`, without constructing the original
  model or invoking a hook. `shard` receives a buffer-only map and initializes
  those buffers and any dependencies.
- **Cache reuse:** Switching between `file` and a non-file strategy, or changing
  `param_init_seed`, requires retracing (`gen_reuse: moo` or a fresh generated-code
  directory). `full` to `shard` with the same seed can reuse the graph; the reverse
  requires retracing to create `npbuffer.pt`. Missing `npbuffer.pt` also requires
  retracing for `file` and `full`. `debug.param_init_check` does not affect generated code. If generated
  code is already imported, use a fresh process or instance name; `moo` cannot
  replace an imported module.

## Examples

- Example 1: Parallelize the whole module

```python
import torch
from nnscaler import parallelize, ComputeConfig, build_optimizer

class LLM(torch.nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, x):
        ...

llm_sample_input = ...              # dummy input will be used to do tracing
pas_policy = ...                    # the PAS policy, you can use autodist pas
compute_config = ComputeConfig(
    plan_ngpus=...,
    runtime_ngpus=...,
    use_zero=...,
    ...,
)                                   # compute environment config
ParallelizedLLM = parallelize(
    LLM,
    {'x': llm_sample_input},
    pas_policy,
    compute_config,
)
```

- Example 2: Parallelize submodules.

In this case, for non-paralle modules, they are replicated inside unit, and run data parallelism across units. See more details about unit in [Compute Config](./trainer) section.

```python
import torch
from nnscaler import parallelize, ComputeConfig, build_optimizer

class HeavyModule(torch.nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, x):
        ...

class ParallelizedLLM(torch.nn.Module):
    def __init__(self, ...):
        ...
        # use parallelize to convert submodules
        heavy_module_sample_input = ...     # dummpy input will be used to do tracing
        pas_policy = ...                    # the PAS policy, you can use autodist pas
        compute_config = ComputeConfig(
            plan_ngpus=...,
            runtime_ngpus=...,
            use_zero=...,
            ...,
        )                                  # compute environment config
        self.heavy_module = parallelize(
            HeavyModule(),
            {'x': heavy_module_sample_input},
            pas_policy,
            compute_config,
        )
        # you can add other submodules here
        ...

    def forward(self, x, ...):
        # call other submodules
        ...
        x = self.heavy_module(x)
        ...
        # call other submodules
        return x
```

For both example 1 & 2, you can train/infer that module in multiple GPUs/Nodes just like a normal `torch.nn.Module`:

```python
# do inference exactly the same way
def infer(model: ParallelizedLLM, x):
    model.eval()
    with torch.inference_mode():
        return model(x)


# do training exactly the same way
# except you need to patch your optimizer to support distributed training via build_optimizer
def train(model: ParallelizedLLM, data):
    loss_fn = ...
    # build_optimizer function will help to create a distributed optimizer
    optimizer = build_optimizer(model, ...)

    for i, (x, y) in enumerate(data):
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

- Example 3: Parallelize end2end module.
```python
class End2EndMLP(nn.Module):
    def __init__(self):
        init_random()
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(8):
            self.layers.append(nn.Linear(16, 16, bias=False))
        self.loss_fn = nn.BCELoss()

    def forward(self, data: Dict[str, torch.Tensor]):
        x = data['data']
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(x)
        loss = self.loss_fn(x, data['target'])
        return loss

    llm_sample_input = {'data': ..., 'target': ...}  # dummy input will be used to do tracing
    pas_policy = ...                    # the PAS policy, you can use autodist pas
    compute_config = ComputeConfig(
        plan_ngpus=...,
        runtime_ngpus=...,
        use_zero=...,
        use_end2end=True,
        ...,
    )                                   # compute environment config
    ParallelizedPipelinedLLM = parallelize(
        LLM,
        {'data': llm_sample_input},
        pas_policy,
        compute_config,
    )
```

For end2end modules, you can't use `Module.forward`.
Instead, you must use `ParallelModule.train_step` and `ParallelModule.infer_step` to train/infer the module.

```python
def infer(model: ParallelizedPipelinedLLM, data):
    model.eval()
    with torch.inference_mode():
        return model.infer_step(data)


def train(model: ParallelizedPipelinedLLM, data):
    # build_optimizer function will help to create a distributed optimizer
    optimizer = build_optimizer(model, ...)

    for i, x in enumerate(data):
        model.train()
        losses = model.train_step(x)
        optimizer.step()
        optimizer.zero_grad()
```

## BroadcastGenFilesStrategy

The broadcast strategy for new generated files.
Please note reused files (i.e., matched by `ReuseType`) are never broadcasted.

The generated files include:
1. config file: compute config (`compute_config.pt`)
2. trace files: graph dump (`graph.ckp`), forward args dump (`forward_args.pkl`), origin module metadata (`origin_module_metadata.pt`), init weights (`fullmodel.pt.*`), param name mapping (`dist_param_map.pt`)
3. code: generated code files (`gencode*.py`)

```python
class BroadcastGenFilesStrategy(Enum):
    NONE = 'none'
    ALL = 'all'
    NO_WEIGHTS = 'no_weights'
    CODE = 'code'
```

1. `NONE`: nothing will be broadcasted.

    You need to do it by yourself or the generated files are saved in a shared directory (like azure blob).

2. `ALL`: broadcast all new generated files to all nodes (Recommended).

    This is useful when you want to run the same code on all nodes.
    Please note the init weight files can be huge.

3. `NO_WEIGHTS`: broadcast all new generated files except init weights (`fullmodel.pt.*`) (Only for experts).

    Without weights, you can only construct the parallel module with `init_params=False`.
    You can then:
    - Safe way: use `broadcast_weights` to get the weights from the workers who have init weights. By default rank 0 will run the `parallelize` and store all the generated files. So if local world size is bigger than `plan_ngpus`, you can use `broadcast_weights` to get the weights from workers on node 0.
    - Risky way: load the weights from a checkpoint file with `module.load_state_dict`, `load_merged_state_dict` or `load_deduped_state_dict`.

    Please note: the non-persistent buffers will remain uninitialized after loading the checkpoints,
    because they are not saved in the state dict.
    You still need to set `init_params=True` to make sure non-persistent buffers are initialized if you want to initialize weights by loading a checkpoint.

4. `CODE`: broadcast the new generated code (`gencode*.py`) and `compute_config.pt` only. It's your responsibility to make sure other necessary files are available on all nodes.

Here are some guidelines to choose the strategy:

1. When restarting a training and there is a successful previous run: As we have a previous run, the compiling process has been done before. So there will be no new generated files and no broadcast will happen no matter what this option is. Please make sure the reuse flag of `parallelize` is `MATCH`, so we can ensure the generated code is the same as the previous run.

2. When training a model from scratch. If there is only one node, `none` is good enough.
If there are multiple nodes, here are some strategies:

a. If use `none`, the user should run `parallelize(..., load_module=False, ..)`, and then copy all files to all nodes manually, so all nodes have the same files. Then the user loads the module by running `parallelize(..., load_module=True, ..)`.

b. If they are using a NAS-like device to save generated files, and the upload/download speed is fast in the cluster, they can also use `none`, and just run `parallelize(..., load_module=True, ..)` to do the training.

c. If use `all`, then user can just run `parallelize(..., load_module=True, ..)` safely. (Remember to set `nccl` communication timeout to a very large value to tolerate the duration of this `nccl` broadcast). This is the most recommended way.

d. If use `no_weights`, then user can run `parallelize(..., load_module=True, init_module_params=rank<plan_ngpus, ..)`. After the module is loaded, the user should call `broadcast_weights(plan_ngpus)` manually to synchronize the module weights before training (note all submodules have the same `plan_ngpus`). Here is an example:
```python
class Module(torch.nn.Module):
    ...
plan_ngpus = ...
rank = torch.distributed.get_rank()
local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', default=1))
assert local_world_size < plan_ngpus

parallel_module = parallelize(Module(), ..., load_module=True, init_module_params=rank<plan_ngpus, broadcast_strategy='no_weights', ...)

broadcast_weights(parallel_module, plan_ngpus)
# now the module is ready to train
```
`no_weights` option is only suggested for experts, because you must be very careful to make it right
when there are non-persistent buffers in the module.

e. Currently `code` option is provided just for completeness. Do not suggest users to use.

## Module Parallelization

We have `parallelize` function to convert a `torch.nn.Module` to a `ParallelModule`.
```python
def parallelize(
    module_or_module_class: Union[torch.nn.Module, Type[torch.nn.Module]],
    dummy_forward_args: Dict[str, Any],
    pas_policy: Union[str, Callable[[IRGraph, ComputeConfig], IRGraph], Callable[[IRGraph, ComputeConfig], Iterable[OpPlan]]],
    compute_config: ComputeConfig,
    *,
    gen_savedir: Union[str, Path] = './.nnscaler',
    reuse: Union[ReuseType, str] = ReuseType.MATCH,
    instance_name: Optional[str] = None,
    load_module: bool = True,
    module_dtype: Optional[torch.dtype] = None,
    module_fn: Optional[Callable[[], torch.nn.Module]] = None,
    init_module_params: bool = True,
    build_module_buckets: bool = True,
    broadcast_strategy: Union[str, BroadcastGenFilesStrategy] = 'none',
) -> Union[None, ParallelModule, Type[ParallelModule]]:
```
It has the following parameters:

- `module_or_module_class` (`Union[torch.nn.Module, Type[torch.nn.Module]]`): the module or module class to be compiled. Please note if the input is a module object, we will return a `ParallelModule` object. If the input is a module class, we will return a `ParallelModule` class.

- `dummy_forward_args` (`Dict[str, Any]`): the dummy input for the module forward.
The keys are the argument names of `Module.forward` function,
and the values are the dummy input for the arguments.
The dummy forward args will be used to trace the module.
Please note the module can't be parallelized if `Module.forward` has positional-only arguments.

- `pas_policy` (`Union[str, Callable[[IRGraph, ComputeConfig], IRGraph], Callable[[IRGraph, ComputeConfig], Iterable[OpPlan]]]`): the pas (partition-assign-schedule) policy, which describes how to place all computations across devices.
You need either pass a builtin PAS policy name or a custom policy function which should take an `IRGraph` and a `ComputeConfig` as input, and return a new `IRGraph` or an iterable of `OpPlan`.

 We have 6 builtin PAS policies: `dp`, `tp`, `pp`, `data`, `hybrid`, and `autodist`. Please note all builtin PAS policies except `autodist` are only for test purpose. The `autodist` policy is the recommended policy for most cases.
 For details, please refer to [PAS Policies](./trainer) section.

- `compute_config` (`ComputeConfig`): the environment resource

- `reuse` (`ReuseType`): specify which part can be reused.

- `gen_savedir` (`Union[str, Path]`): the directory to save generated code

- `instance_name` (`Optional[str]`): the instance name of the generated module. If it is `None`, will use the default name `_`.

- `load_module` (`bool`): whether to load the generated module or module class after parallelization is done.
Currently the module can only be loaded in `torchrun` environment. So you can do the parallelization in any environment (with `load_module` unset), and load the module in `torchrun` environment.

- `init_module_params` (`bool`): If true, when we construct the module, all its parameters are initialized with the same value as when we traced.
Otherwise, they will be empty tensors.
This parameter will be passed to the module constructor,
so it is only used when `module_or_module_class` is a module object, and `load_module` is true.
See more details in the `ParallelModule APIs` section.

- `build_module_buckets` (`bool`): For parallel module, parameters that need to synchronize will be grouped into buckets for more efficient communication.
If true, the grouping process will be done in `__init__`.
If false, you should call `build_buckets()` manually before using the module.
This parameter will be passed to the module constructor,
so it is only used when `module_or_module_class` is a module object, and `load_module` is true.
Leave it as true unless you have a specific reason to defer bucket building (e.g., when using a hybrid optimizer with `param_clss_fn`).

- `module_dtype` (`Optional[torch.dtype]`): the dtype of the module. Keep the module as it is if it is None.

- `module_fn` (`Optional[Callable[[], torch.nn.Module]]`): the function to create the module. Will use `__init__` if it is None. This parameter is only used when `module_or_module_class` is a module class.

- `broadcast_strategy` (`Union[str, BroadcastGenFilesStrategy]`): the broadcast strategy for new generated files.

Note:

1. This function can be used to convert both module object and module class to parallel module or parallel module class.
Among key-value arguments,
`module_fn` and `module_dtype` control how to create the module object.
whereas `init_module_params` controls how to load parallel module object after parallelization is done.

2. If you want to save multiple instances of the same module (with different configurations),
you can specify the `instance_name` to distinguish them.

3. `load_module` flag should be used with `broadcast_strategy`. See more details in the `BroadcastGenFilesStrategy` section.

4. if `reuse` is not set to `ReuseType.MATCH`,
the generated code in outdir will be removed EVEN IF the code generation fails in this call.

5. For `broadcast_strategy`, please note that the broadcast will only be done in `torchrun` environment, and will throw an error if `torch.distributed` is not initialized and `broadcast_strategy` is not `NONE`.


## Optimizer Creation

We have `build_optimizer` to build an optimizer for distributed training.
```python
def build_optimizer(
    module: torch.nn.Module,
    optimizer_fn: Union[Type[OptimizerT], Callable[..., OptimizerT]],
    compute_config: Optional[ComputeConfig] = None,
    param_clss_fn: Optional[Callable[[str], Any]] = None,
    **kwargs,
) -> OptimizerT:
```
It has the following parameters:
- `module` (`torch.nn.Module`): the module to be optimized
- `optimizer_fn` (`Union[Type[torch.optim.Optimizer], Callable[..., torch.optim.Optimizer]]`):
    It can be the optimizer class or optimizer factory function.
    The first parameter of the `optimizer_fn` should be the module parameters.
- `compute_config` (`Optional[ComputeConfig]`):
    The config will be used to generate communication reducer.
    If it is None, default configuration will be used when creating reducer for non-parallel modules.
- `param_clss_fn` (`Optional[Callable[[str], Any]]`):
    A function that maps original full-qualified parameter names to their class IDs.
    Required when using a hybrid optimizer; the return value must be a `tuple[int, int]` of `(optimizer_index, param_group_index)`.
- `**kwargs`: the kwargs will be passed to `optimizer_fn`.

To support distributed training, in the function we need to hook 4 places (which we have done for you in `build_optimizer`. That's why you should use `build_optimizer` to create optimizer):

1. optimizer constructor:
    the parameters of optimizer will not be the same with the parameters of the module if we use zero.
    So we need to replace the parameters of optimizer with `ParallelModule.parameters_for_optimizer`.

2. `optimizer.step()`:
    we need to call `optimizer.sync_shard_grad()` to sync the gradients of the module before `optimizer.step()`.
    In zero mode, we have to call `ParallelModule.gather_params()` after `optimizer.step()`

3. `optimizer.zero_grad()`:
    We need to call `ParallelModule.zero_grad()` after `optimizer.zero_grad()`

`build_optimizer` will patch optimizer for you. Besides the above patches, we also add several utility functions/variables to optimizer:

1. `sync_shard_grad`: Sync the shard gradients of the module from nodes with same shard to the optimizer.
Please note the gradients are `None` until `optimizer.sync_shard_grad()` is called.
This function is called in optimizer's pre-step hook.  You need to manually call it in two cases:
    - If you want to access the gradients before `optimizer.step()`.
    - When closure is used in optimizer.step(). In this case, optimizer's pre-step hook will be called before `train_step`, so no gradients are synced.

2. `scale_grads`: Scale the gradients of the module by multiplying a factor. This function is useful to avoid overflow when the gradients are large. Please note you can only call this function **after** `sync_shard_grad`, because the gradients are `None` until `sync_shard_grad` is called.

3. `clip_gnorm`: Clip the gradients with global norm, and return the global gnorm value, it will sync grads across devices if necessary. This function is useful to avoid gradient explosion.

4. `register_reducer_pre_hook`, `register_reducer_post_hook`: Register pre/post hooks to reducers which will be applied before/after gradient synchronization. These hooks will apply to all the reducers (including `_non_parallel_module_reducer`) in the optimizer.

You can use `register_reducer_pre_hook` and `register_reducer_post_hook` to do some operations before/after gradient synchronization. Not all parameters are managed by reducers, so it is tricky to use them. Actually we don't encourage you to use these functions.

Here is one example (Assume we calculate loss with sum) showing how to carefully scale down the gradient locally and scale up the gradient after reduce. This is useful to avoid overflow when the gradients are large:.

```python
num_scale_units = ...
optimizer.register_reducer_pre_hook(lambda reducer, grad: grad.div_(num_scale_units)) # scale down with factor num_scale_units before reduce
optimizer.register_reducer_post_hook(lambda reducer, grad: grad.mul_(num_scale_units) # scale up with factor num_scale_units after reduce
```

5. `_non_parallel_module_reducer`: The reducer for the modules which are not parallelized. It is used to sync the parameters in those modules across units.

## ParallelModule APIs

The `ParallelModule` is a subclass of `torch.nn.Module`. It has the following APIs:

1.constructor
```python
def __init__(self, init_params=True, build_buckets=True):
    ...
```
- `init_params` (`bool`): whether to initialize the module parameters with the values they had at trace time. Set to `False` if you plan to load from a checkpoint instead.
- `build_buckets` (`bool`): whether to build communication buckets immediately. Set to `False` when you need to call `build_buckets()` manually later (e.g., for hybrid optimizers with `param_clss_fn`).

As noted before, in most cases you still need to set `init_params=True` to make sure non-persistent buffers are initialized if you want to initialize weights by loading a checkpoint.


2.`train_step`
```python
def train_step(self,
    samples: List[Any],
    is_dummy_batch: Optional[List[bool]] = None,
    scale_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> List[Any]:
    ...
```
The training step function. It should be called in the training loop.
Please note:
    1. This function is only supported in end2end mode.
    2. Gradient accumulation is done in this function.
        You shouldn't do it outside this function,
        because gradients will be cleared at the beginning of this function.

It has the following arguments:
- `samples` (`List[Any]`): a list of samples.
        If pipeline is used, it must have the same length as configured in the pas policy.
- `is_dummy_batch` (`Optional[List[bool]]`): indicates whether each micro-batch is dummy.
- `scale_fn` (`Optional[Callable[[torch.Tensor], torch.Tensor]]`): the function to scale the loss.

Returns a list of outputs for the samples.

3.`infer_step`
```python
def infer_step(self, samples: List[Any]) -> List[Any]:
    ...
```
The inference step function. It should be called in the inference loop.
Only supported in end2end mode.
The input is a list of samples, and returns a list of outputs for the samples. If pipeline is used, it must have the same length as configured in the pas policy.

4.`build_buckets`
```python
def build_buckets(self, param_clss: Optional[dict[torch.nn.Parameter, Any]] = None):
    ...
```
Build communication buckets for the model reducers. Must be called exactly once before using the module if `build_module_buckets=False` was passed to `parallelize()`.

- `param_clss` (`Optional[dict[torch.nn.Parameter, Any]]`): parameter-to-class mapping produced by `param_clss_fn`. Used to put parameters with different optimizer or param groups into separate buckets.

5.`sleep`
```python
def sleep(self) -> Self:
    ...
```
Move all parameters and buffers to CPU and release contiguous reducer memory. Unlike `nn.Module.cpu()`, attribute references are unchanged. Useful for temporarily freeing GPU memory when the module is not in use.

6.`wake_up`
```python
def wake_up(self, device: Optional[Union[int, torch.device]] = None) -> Self:
    ...
```
Move all parameters and buffers back to GPU and reallocate reducer memory. This is the reverse of `sleep()`.

## Checkpoint support

You can save/load the checkpoints for parallel modules.
Each rank will save/load its own checkpoint just like a normal module.

Note: The only exception is the non-persistent buffers, which will remain uninitialized after loading the checkpoints, because they are not saved in the state dict. To make sure all the buffers are initialized, you must initialize the module with `init_params=True`.

You can also merge the checkpoints from different ranks into a single checkpoint.
We call it a merged checkpoint. The merged checkpoint can be loaded by the original module directly.
So you can easily share the checkpoint with the original module.

On the other hand, a lot of weights/state in the module and the optimizer will be the same across ranks in parallel training. So we can save a lot of space by deduplicating the state dicts before saving them to disk.

We provide two functions to help you save/load the merged checkpoint for the parallel module,
and two other functions to help you save/load the deduped state dicts for the parallel module.

### `merge_state_dicts`
```python
def merge_state_dicts(
    module_state_dicts: List[Dict[str, Any]],
    optimizer_state_dicts: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
```

Merge a list of per-rank state dicts into a single full state dict.
Note: Only Adam/Muon-like optimizers are supported for merging.

The state dicts do not need to be in rank order; the function will sort them internally using the rank stored in each state dict.

Please Note:
    We don't guarantee the devices of tensors in the merged state dict will be uniform.
    The device of each tensor can be one of:
        1. `'cpu'` (for tensors originating from parallel module merging)
        2. the device of the tensor in the original state dict (for non-parallel module tensors)
    When loading state dicts from file, use `torch.load(..., map_location='...')` to unify devices.


### `load_merged_state_dict`
```python
def load_merged_state_dict(
    module: torch.nn.Module,
    module_state_dict: Dict[str, Any],
    optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    *,
    device: Union[str, torch.device] = None
) -> None:
```
Load the merged state dicts to the module, and optionally the optimizer, to a specified device.

Please note the `device` parameter. If it is None, `torch.cuda.current_device()` will be used. If you want to load the state dict to a specific device, you can set it to the device you want.


### `deduped_state_dict`

In parallel training, many weights/states in the module and optimizer are identical across ranks. We can save significant disk space by deduplicating state dicts before saving. Each part of a logical tensor is saved only at the first rank it appears.

```python
def deduped_state_dict(
    module: torch.nn.Module,
    optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
```

Returns the deduped `(module_state_dict, optimizer_state_dict)` for the current rank. Ranks that are not responsible for a particular shard will have those entries omitted.

### `load_deduped_state_dict`

This is the reverse of `deduped_state_dict`. It assumes the distributed plan is unchanged. The loading process is:
1. Each rank reads its own (partial) state dict.
2. Replicated weights are broadcasted inside the dedup group so that each group has the full parameters.
3. The first scale unit broadcasts weights to other units via `broadcast_weights`.

```python
def load_deduped_state_dict(
    module: torch.nn.Module,
    module_state_dict: Dict[str, Any],
    optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    *,
    device: Union[str, torch.device] = None
) -> None:
```

## Dataset

We use the same dataset/dataloader as pytorch. For example, you can use `torch.utils.data.DistributedSampler` to create a distributed sampler.

`ParallelModule`s running in the same unit should use the same input, and will get the same output. `ParallelModule`s running in different units should use different input and will get different output (similar to data parallelism). The gradients of all parameters will be synced across all the devices automatically.

Take `torch.utils.data.DistributedSampler` for example, you can create the sampler like this:
```python
def create_distributed_sampler(dataset):
    return torch.utils.data.DistributedSampler(
        dataset=dataset,
        num_replicas=compute_config.runtime_ngpus // compute_config.plan_ngpus,
        rank=torch.distributed.get_rank() // compute_config.plan_ngpus,
        ...,
    )
```

## self.training support

To parallelize the training process, we firstly need to trace the module and get a static computational graph.

A common problem with static graph is that it is impossible to handle control flow.

But on the other hand, `self.training` is very common used in module forward method.
So we add a very limited support for `self.training` in tracing.

Please note that user code is flattened and transformed into a single `ParallelModule` at runtime, so `training` is a global module state, and we don't support the case that user want to set a sub-module's training to True but remaining modules to False.

## Some Details on Integration with Trainer

There are two ways to use `ParallelModule` with Trainer:

1. Pipeline Parallelism with End2End Module (Data Parallelism/Tensor Parallelism can also be used here): You must use `ParallelModule.train_step` and `ParallelModule.infer_step` (which are wrappers of `_train_step`/`_infer_step` from gencode of `ExecutionPlan`) to train/infer the module. The PAS policy must have pipeline parallelism, and the compute config must set `use_end2end=True`.

2. Pure Non-Pipeline Parallelism (Data Parallelism/Tensor Parallelism) : You can use `ParallelModule` just like a normal `torch.nn.Module`, i.e., call `ParallelModule.forward` to do forward, and use `build_optimizer` to create optimizer for the module. `ParallelModule.train_step` and `ParallelModule.infer_step` are also available, which are just a wrapper of `ParallelModule.forward`. The PAS policy must not have pipeline parallelism.

We can distinguish the above two ways by checking `ParallelModule.use_scheduler` flag.

In the following, we will refer to the first way as "PP", and the second way as "Non-PP" for better readability.

### Gradient Accumulation Support

Gradient accumulation is done with two runtime flags: `RuntimeFlag.skip_zero_grad` and `RuntimeFlag.skip_reducer`.

In PP mode, both flags are managed directly in generated code of `ExecutionPlan`, and you don't need to care about them. The codegen will automatically set the flags according to the micro-batch index and the accumulation steps.

In Non-PP mode, If you use `ParallelModule.forward` directly, you need to manually set the flags in the training loop for gradient accumulation by `nnscaler.utils.accum_mode`. If you use `ParellelModule.train_step`, the flags will be automatically set in `train_step` according to the micro-batch index and the accumulation steps, so you don't need to care about them.


### Gradient Reduction Support

In the end of `train_step`, we need to sync the gradients across devices. The way we sync gradients is different in PP and Non-PP mode.

We will always call `optimizer.sync_shard_grad()` to sync the gradients before `optimizer.step()`, but in end2end model, the `sync_shard_grad` is a no-op because the gradients are already synced in the codegen (`_train_step`), whereas in Non-end2end mode, the `sync_shard_grad` will do the real synchronization.

In Non-PP mode, to support multiple calls of `optimizer.sync_shard_grad()`, `ParallelModule` will keep track whether the gradients are synced or not with `self._sync_grad_required` flag, and only sync the gradients when `self._sync_grad_required` is True. So you can call `optimizer.sync_shard_grad()` multiple times without worrying about it.

We also support async gradient reduction via `compute_config.use_async_reducer`. In this case, the gradient reduction will be kicked off once the gradients are ready, and `optimizer.sync_shard_grad()` will wait for the reduction to be done if it is called before the reduction is done.

When we combine async reduction with gradient accumulation, The time of kicking off gradient reduction becomes a problem. The current implementation is reusing `RuntimeFlag.skip_reducer` flag to control when to kick off the reduction. It is not ideal because `RuntimeFlag.skip_reducer` is originally designed for gradient accumulation, and it is not compatible when overlapping is used. So in overlapped scenarios, we must not use async reduction. We will improve it in the future.
