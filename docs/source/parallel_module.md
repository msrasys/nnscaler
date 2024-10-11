# Parallel Module

nnScaler can parallelize a `torch.nn.Module` to a parallel module.
A parallel module is a special `torch.nn.Module` but runs in multiple gpus/nodes.
All the complexity of distributed training/inferring is hidden from the user.

Currently we support three kinds of parallelism: data parallelism, tensor parallelism and pipeline parallelism (model parallelism). We can also combine them to get the best performance.

Data parallelism and tensor parallelism are support for all kinds of module, but pipeline parallelism is only supported for end2end modules for scheduling reason.

An end2end module is a module which satisfies:
- the first argument of `module.forward` is the data sample, and every other argument should have default value, and use its default value in `module.forward` function.
- the first return value of `module.forward` is the loss (scalar tensor)

The above restrictions are necessary for the pipeline parallelism to work. Of course, you can still use the parallel module without pipeline parallelism for end2end modules.

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

In this case, for non-paralle modules, they are replicated inside unit, and run data parallelism across units. See more details about unit in [Compute Config](###ComputeConfig) section.

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

## APIs

### ComputeConfig
The configuration of the compute environment. It is a dataclass with the following fields:
```python

@dataclass(frozen=True)
class ComputeConfig:
    plan_ngpus: int
    runtime_ngpus: int

    constant_folding: bool = False
    trace_strategy: Literal['cpu', 'cuda', 'meta', 'cuda_run_cpu_offload', 'reuse_cache'] = 'cuda_run_cpu_offload'

    use_zero: bool = False
    zero_ngroups: int = 1
    zero_use_reduce_scatter: bool = False

    inference_only : bool = False
    use_end2end: bool = False

    use_async_reducer: bool = False
    reducer_bucket_cap_mb: Optional[float] = None

    pas_config: Dict[str, Any] = field(default_factory=dict)
    user_config: Dict[str, Any] = field(default_factory=dict)
```
We can categorize the fields into 4 categories:

1. Trace configuration
    - `constant_folding`: whether to enable constant folding when generating code.
    When it is true, all non-tensor non-input values will be folded into the generated code.

        For example, if user's code contains following snippet, and `bsz=1`, `num_heads=32`, `len=1024`, `hidden_dim=128` at tracing.
        ```python
            bsz, num_heads, len, hidden_dim = x.size()
            x = x.view(bsz * num_heads, len, hidden_dim)
        ```
        The code (graph) is folded into the following format

        ```python
            y = x.view(32, 1024, 128)
        ```

        Constant folding is helpful to simplify the input program,
        and can make the compiling process faster and reduce the communication cost at runtime.
        However, user should make sure that inputs at runtime share a same schema (including shape) with tracing and correspond to a same computation graph.
        Errors may be raised at runtime when this assumption is broken.
    - `trace_strategy`: how to execute the functions during trace.
    Five strategies are supported:
        1. `cpu`: Execute all functions on cpu device, model weights and intermediate results are on cpu device.
        2. `cuda`: Execute all functions on cuda device, model weights and intermediate results are on cuda device. This strategy is recommended if the model can inference on single gpu.
        3. `meta`: Execute all functions on meta device, model weights are on cpu and intermediate results are on meta device. For more information about meta device type, please view https://pytorch.org/docs/stable/meta.html.
        4. `cuda_run_cpu_offload`: Try to execute all functions on cuda, and retry to execute the function on cpu as backup if OOM is catched, model weights and intermediate results are on cpu. This strategy is recommanded for most case if the model is too large to inference on single gpu.
        5. `reuse_cache`: Compared to `cuda_run_cpu_offload` strategy, maintains a map from function signatures to output values. The cached output is returned when the signature of the function that generates it has been executed. Same signature means the funtions are the same and have almost the same inputs (for tensor type input, just check if they have same tensor meta data[shape, dtyep, requires_grad, stride, memory_format, ...], and don't check the value). This strategy is an experimental strategy to speedup the large-model-large-input case, and have risk to trace an incorrect graph if the signature defined here can not distinguish the differnet functions used in the model, for example, torch.nonzero will always return the same result if the input have same meta data but different value. We have plan to continue improve this strategy to handle most these kind of data dependence cases, but please note that the risk is still inevitable.
2. Compute environment configuration
    - `plan_ngpus`: the number of gpus to be used as a unit. The model is partitioned (TP or PP) within a unit, and then data parallelism is applied across multiple units. So every `plan_ngpus` devices holds the whole model. Furthermore, assume we have two workers, and their ranks are `rank1` and `rank2`:
        1. if `rank1 // plan_gpus == rank2 // plan_ngpus`, then they are in the same unit.
        2. If `rank1 % plan_ngpus == rank2 % plan_ngpus`, then the portion of model hold on both gpus are exactly the same.
    - `runtime_ngpus`: the number of gpus to be used in runtime. It should be a multiple of `plan_ngpus`, which means we have `runtime_ngpus // plan_ngpus` units in runtime, and the data parallelism is `runtime_ngpus // plan_ngpus`.
    Please note all modules must have the same `plan_ngpus` and `runtime_ngpus`.
3. Code generation feature configuration
    - `use_zero`: whether to use zero. If it is true, the generated code will use zero1 to do distributed training.
    - `zero_ngroups`: the number of groups to be used in zero.
    - `zero_use_reduce_scatter`: whether to use reduce scatter in zero. If it is true, the gradients will be reduced by reduce scatter in zero.

       Please note
        - Reduce scatter is only available when `zero_ngroups` is 1. when `zero_ngroups` > 1, you should set it to `False`, or an error will be raised.
        - In some cases, it can introduce parity issue. So use it with caution.
    - `inference_only`: whether to generate code for inference only. If it is true, the generated code can not be used to train the model.
    - `use_end2end`: whether to use end2end training. For the requirement of end2end, see the description above.
    - `use_async_reducer`: whether to use async reducer.
        If it is true, the gradients will be reduced asynchronously.
        Please note this only works when `use_end2end` is true.
    - `reducer_bucket_cap_mb`: the bucket capacity of the reducer.
        If it is `None` or `0`, the default value will be used, which is
        - 25MB for async, the same default value with pytorch ddp implementation
        - no limit for sync

        Please note this only works when `use_end2end` is true.
    - `pas_config`: the configuration for the PAS policy (partition-assign-schedule policy, which describes how to place all computations across devices. For details, please refer to [PAS Policies](#pas-policies)).
    It is a dictionary, and will be used by the PAS policy.
    Please note different PAS will have different configurations,
    You can also put any other settings that can affect code generation here. but please prefix the keys with `_` to avoid conflicts with PAS configurations.
    - `user_config`: the user configuration, which is used to decide whether skipping compiling and reusing the previously traced graph.

Note:
1.  You can put any custom configurations in `user_config`. The assumption is different `user_config` should generate different graph/code. So if the user config is changed, we will regenerate the graph/code automatically. Here are some examples:

    - Example 1: save module configuration
        ```python
        class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            ...
            if module_config.use_3d:
            ...
        ```
        here we can set `user_config` to `{'use_3d': module_config.use_3d}`,
        and we can be sure different use_3d config will never use the same graph (and eventually the generated code).

    - Example 2: save file stats
        If you want to track all related file stats (just like traditional compilers do),
        you can save the md5 of the files to save some bytes:
        ```python
        import hashlib
        h = hashlib.md5()
        for f in Path('./src').glob('**/*.py'):
        with open(f, 'rb') as f:
            h.update(f.read())
        compute_config = {
            ....,
            user_config: {
                'files_md5': h.hexdigest()
            }
        }
        ```
2.  If some settings doesn't affect tracing/graph generation, but do affect code generation, you can put them in `pas_config`. Please prefix the keys with `_` to avoid conflicts with predefined PAS configurations. One typical example is you can put the name of selected PAS policy in `pas_config`, so changing PAS policy will regenerate code but the graph will be reused.

    ```python
    compute_config = ComputeConfig(
        ...
        pas_config={
            '_pas_name': ...,
            # PAS policy specific configurations
            ...
        },
    )
    ```

### ReuseType

The reuse policy for the existing generated code. It is an enum with the following values:

```python
class ReuseType(Enum):
    MATCH = 'match'
    OVERRIDE = 'override'
    MOO = 'moo'
    GRAPH = 'graph'
```
We call it a `match` when the `ComputeConfig` is the same with the previous run.

1. `MATCH`: Reuse if match, error if not match, generate if no previous gerenated code exists.
2. `OVERRIDE`: Nothing will be reused. Everything will be regenerated.
3. `MOO`: `MOO` is short for 'match or override'. It will reuse if match, generate if not match or no previous generated code exists.
4. `GRAPH`: Reuse graph only if match, generate otherwise.

### BroadcastGenFilesStrategy

The broadcast strategy for new generated files.
Please note we never broadcast reused files (i.e., specified by `ReuseType`.).

```python
class BroadcastGenFilesStrategy(Enum):
    NONE = 'none'
    ALL = 'all'
    NO_WEIGHTS = 'no_weights'
    CODE = 'code'
```

1. `None`:  nothing will be broadcasted.

    You need to do it by yourself or the generated files are save in a shared directory (like azure blob).

2. `ALL`: broadcast all the generated files to all nodes (Recommended).

    This is useful when you want to run the same code on all nodes.
    please note the init weight files can be huge.

3. `NO_WEIGHTS`: broadcast all except init weights (Only for experts).

    Without weights, you can only construct the parallel module with `init_params=False`.
    You can then
    - Safe way: you can use `broadcast_weights` to get the weights from the workers who have init weights. By default rank 0 will run the `parallelize` and store all the generated files. So if local world size is bigger than plan_ngpus, you can use `broadcast_weights` to get the weights from workers on node0.
    - Risk Way: Load the weights from a checkpoint file with `module.load_state_dict`, `load_merged_state_dict` or `load_deduped_state_dict`.

    Please note: the non-persistent buffers will remain uninitialized after loading the checkpoints,
    because they are not saved in the state dict.
    To make sure all the buffers are initialized,  you still need to set `init_params=True` to make sure non-persistent buffers are initialized if you want to initialize weights by loading a checkpoint.

4. `CODE`: broadcast the new generated code only (Not recommeneded)
    It's your responsibility to make sure other necessary files are available on all nodes.

Here are some guidelines to choose the strategy:

1. When restarting a training and there is a successful previous run: As we have a previous run, the compiling process has been done before. So there will be no new generated files and no broadcast will happen no matter what this option is. Please be sure the reuse flag of `parallelize` is `MATCH`, so we can make sure the generated code is the same with the previous run.

2. When training a model from scratch. If there is only one node, `none` is good enough.
If there are multiple nodes, here are some strategies:

a. If use `none`, the user should run `parallelize(..., load_module=False, ..)`, and then copy all files to all nodes manually, so all nodes have the same files. Then the user load the module by running `parallelize(..., load_module=True, ..)`.

b. if they are using a NAS-like device to save generated files, and the upload/download speed is fast in the cluster, they can also use `none`, and just run `parallelize(..., load_module=True, ..)` to do the training.

c. If use `all`, then user can just run `parallelize(..., load_module=True, ..)` safely. (remember to set `nccl` communication timeout to a very big value to tolerate the duration of this `nccl` broadcast). This is the most recommended way.

d. If use `no_weights`. then user can run `parallelize(..., load_module=True, init_module_params=rank<plan_ngpus, ..)`. After module is loaded, the user should call `broadcast_weights(plan_ngpus)` manually to synchronize the module weights before training (Note all submodules have the same `plan_ngpus`). Here is an example:
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

### Module Parallelization

We have `parallelize` function to Convert a torch.nn.Module to a ParallelModule.
```python
def parallelize(
    module_or_module_class: Union[torch.nn.Module, Type[torch.nn.Module]],
    dummy_forward_args: Dict[str, Any],
    pas_policy: Callable[[IRGraph, ComputeConfig], IRGraph],
    compute_config: ComputeConfig,
    *,
    gen_savedir: Union[str, Path] = './.nnscaler',
    reuse: Union[ReuseType, str] = ReuseType.MATCH,
    instance_name: Optional[str] = None,
    load_module: bool = True,
    module_dtype:  Optional[torch.dtype] = None,
    module_fn: Optional[Callable[[], torch.nn.Module]] = None,
    init_module_params: bool = True,
    broadcast_strategy: Union[str, BroadcastGenFilesStrategy] = 'none',
) -> Union[None, ParallelModule, Type[ParallelModule]]:
```
It has the following parameters:

- `module_or_module_class` (`Union[torch.nn.Module, Type[torch.nn.Module]]`): the module or module class to be compiled. Please note if the input is a module object, we will return a `ParallelModule` object. If the input is a module class, we will return a `ParallelModule` class.

- `dummy_forward_args` (`Dict[str, Any]`): the dummy input for the module forward.
The keys are the argument names of `Module.forward` function,
and the values are the dummy input for the arguments.
The dummy forward args will be used to trace the module.
Please note the module can't be parallelize if `Module.forward` has positional-only arguments.

- `pas_policy` (`Union[str, Callable[[IRGraph, ComputeConfig], IRGraph]]`): the pas (partition-assign-schedule) policy, which describes how to place all computations across devices.
You need either pass a builtin PAS policy name or a a custom policy function which should take an `IRGraph` and a `ComputeConfig` as input, and return a new `IRGraph` with the PAS policy applied.
 We have 6 builtin PAS policies: `dp`, `tp`, `pp`, `data`, `hybrid`, and `autodist`. Please note all builtin PAS policies except `autodist` are only for test purpose. The `autodist` policy is the recommended policy for most cases.
 For details, please refer to [PAS Policies](#pas-policies) section.

- `compute_config` (`ComputeConfig`): the environment resource

- `reuse` (`ReuseType`): specify which part can be reused.

- `gen_savedir` (`Union[str, Path]`): the directory to save generated code

- `instance_name` (`Optional[str]`): the instance name of the generated module. If it is `None`, will use the default name `_`.

- `load_module` (`bool`): whether to load the generated module or module class after parallelization is done.
Currently the module can only be loaded in `torchrun` environment. So you can do the parallelization in any environment (with `load_module` unset), and load the module in `torchrun` environment.

- `init_module_params` (`bool`): If true, when we construct the module, all its parameters are initialized with the same value with when we traced.
Otherwise, they will be empty tensor.
This parameter will be passed to the module constructor,
so it is only used when `module_or_module_class` is a module object, and `load_module` is true.
See more details in the `ParallelModule APIs` section.

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

5. For `broadcast_strategy`, Please note that the broadcast will only be done in `torchrun` environment, and will throw an error if `torch.distributed` is not initialized and `broadcast_strategy` is not `NONE`.


### Optimizer Creation

We have `build_optimizer` to build an optimizer for distributed training.
```python
def build_optimizer(
    module: torch.nn.Module,
    optimizer_fn: Union[Type[OptimizerT], Callable[..., OptimizerT]],
    *args,
    **kwargs,
) -> OptimizerT:
```
It has the following parameters:
- `module` (`torch.nn.Module`): the module to be optimized
- `optimizer_fn` (`Union[Type[torch.optim.Optimizer], Callable[..., torch.optim.Optimizer]]`):
    It can be the optimizer class or optimizer factory function.
    The first parameter of the `optimizer_fn` should be the module parameters.
- *args: other args for `optimizer_fn` besides module parameters.
- **kwargs: the kwargs will pass to `optimizer_fn`

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

### ParallelModule APIs

The `ParallelModule` is a subclass of `torch.nn.Module`. It has the following APIs:

1. constructor
```python
def __init__(self, init_params=True):
    ...
```
You can use `init_params` to control whether to initialize the module parameters with the module parameters' values when we trace it. You can set it to `False` if you don't want to.

As noted before, in most cases, you still need to set `init_params=True` to make sure non-persistent buffers are initialized if you want to initialize weights by loading a checkpoint.


2. `train_step`
```python
def train_step(self,
    samples: List[Any],
    is_dummy_batch: Optional[List[bool]],
    scale_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
) -> List[Any]:
    ...
```
The training step function. It should be called in the training loop.
Please note:
    1. This function is only supported in end2end mode.
    2. Gradient accumulation is done in this function.
        You shouldn't do it outside this function,
        because `zero_grad` will be called in the beginning of this function

It has the following arguments:
- `samples` (`List[Any]`): a list of samples.
        if pipeline is used, it must have the same length as configured to pas policy.
- `is_dummy_batch` (`Optional[List[bool]]`): indicates whether the each micro-batch is dummy
- `scale_fn` (`Optional[Callable[[torch.Tensor], torch.Tensor]]`): the function to scale the loss

And it will return a list of outputs for the samples.

3. `infer_step`
```python
def infer_step(self, samples: List[Any]) -> List[Any]:
    ...
```
The inference step function. It should be called in the inference loop.
The input is a list of samples, and returns a list of outputs for the samples. If pipeline is used, it must have the same length as configured to pas policy.

### PAS Policies

Writing a pas policy can be very hard and error-prone. So we provide 6 builtin PAS policies to help you. `dp`, `tp`, `pp`, `data`, `hybrid`, and `autodist`. Please note only `autodist` policy is the recommended policy for most cases, and all other PAS policies are mainly test purpose only.

The configuration of the PAS policy should be passed in the `pas_config` of `ComputeConfig` as a dictionary.

1. `dp`: data parallelism. It will replicate the module across all devices, and run data parallelism across all devices. It requires the `plan_ngpus` must be 1 and no configurations

2. `tp`: tensor parallelism + data parallelism. It will do tensor parallelism inside a scale unit, and run data parallelism across scale units. It has only one configuration:
    - seed: the random seed for choose the partition dimension. Default is `1`

3. `pp`: pipeline parallelism + data parallelism.
It will do model parallelism inside a scale unit,
and run data parallelism across scale units.
It requires the `use_end2end` be true.
It has two configurations `pipeline_nmicros` and `pipeline_scheduler`.
See `hybrid` policy for more details.

4. `data`: tensor parallelism on batch dimension. It has no configurations.

5. `hybrid`: pipeline parallelism + tensor parallelism + data parallelism.
It will do model parallelism and tensor parallelism(on 0 dimension) inside a scale unit,
and run data parallelism across scale units.
It requires the `use_end2end` to be true. It has the following configurations.
    - `pipeline_nstages`: the number of stages in the pipeline. Default is `plan_ngpus`. Optional.
    - `pipeline_nmicros`: the number of microbatches in the pipeline. Required.
    - `pipeline_scheduler`: the scheduler name for the pipeline. Current we support four schedulers in training `1f1b`/`1f1b_plus`/`gpipe`/`chimera_direct` (4 stages pipeline only), and one scheduler in inference `infer_pipe`. Default is `1f1b`. Optional.

6. `autodist`: the recommended policy for most cases. Currently it only support Adam-like optimizers. It will automatically choose the best partition for you by balancing the memory usage and speed. It has the following configurations.
    - `update_freq (int)`: the update frequency when training the module. Default is 1. Optional.
    - `mem_constraint (float)`: The memory constraint in each device in GB. Optional.
    - `task_name (str)`: The name of the current task to distinguish runs. Optional.
    - `use_fp16 (bool)`: Whether you use `fp16`. Default is `False`. Optional.
    - `use_memory_efficient_fp16` Whether you use memory efficient fp16 optimizer. Default is `False`. Optional.
    - `use_bf16`: Whether you use `bf16`. Default is `False`. Optional.
    - `use_memory_efficient_bf16`: Whether you use memory efficient bf16 optimizer. Default is `False`. Optional.
    - `re_profile (bool)`: If set to `True`, the computation profiling results will be overridden. Please note reprofiling will take some time. Optional.
    - `verbose (bool)`:  Whether to print verbose information. Optional.
    - `load_plan_path (str)`: The path to the plan file to load. If specified, the plan will be loaded from the file instead of searching. Optional.
    - `save_plan_path (str)`: The path to the plan file to save. Optional.
    - `partition_constraints_path (str)`: The path to the partition constraints file. Optional.
    - `recompute_modules (str)`: The module names to recompute, separated by `,`. For example, `module1,module2`. Optional.
    - `pipeline_pivots (str)`: The module names to pivot the pipeline, separated by `,`. For example, if `module1,module2` is specified, stages searched by pipeline solver only start from either `module1` or `module2`. Optional.
    - `use_apex_fused_adam_v2`: If set to `True`, the apex fused adam v2 optimizer will be used. Default is `False`. Optional.
    - `explore_pipeline`: If set to `True`, autodist will try pipeline parallelism to find the best partition plan
    (but the selected partition plan is not necessarily pipeline parallelism).
    - `pipeline_scheduler`: The scheduler name for the pipeline. Please note currently `1f1b` is the only supported scheduler in `autodist`. Default is `1f1b`. Optional.
    - `parallel_profile`: If set to `True`, autodist will profile operators in parallel by using available gpus. Default is `True`. Optional.
    - `max_partition_degree`: Max degree when partitioning an operator / node. When pipeline parallelism is enabled to explore (`explore_pipeline` is True), user can change the value to constrain the plan to be composed of stages that span on less or equal to `max_partition_degree` devices (recommend to set `max_partition_degree` to the number of devices in a node to avoid inter-node communication, but should be be no more than `plan_ngpus`). Default is `plan_ngpus`. Optional.
    - `transient_mem_coef`: In autodist, a heuristic is used to estimate the transient memory size: `transient_mem_size = opt_transient_coef * (1st_largest_infer_mem + 2nd_largest_infer_mem)`. This formula is useful in many cases, but it may be too strict when some operators consume or generate a large tensor (>= 4GB). In this case, you can set `transient_mem_coef` to a smaller value to relax the constraint. Default is `2`. Optional.

 You can also put any other settings that can affect code generation here. but please prefix the keys with `_` to avoid conflicts with predefined keys.

Here is an example:
```python
compute_config = ComputeConfig(
    plan_ngpus=...,
    runtime_ngpus=...,
    use_zero=...,
    pas_config={
        '__pas_name': ...,   # addtional configurations that can affect code generation.
        'update_freq': ...,
        'mem_constraint': ...,
        'task_name': ...,
        'use_fp16': ...,
        'use_memory_efficient_fp16': ...,
        'use_bf16': ...,
        'use_memory_efficient_bf16': ...,
        're_profile': ...,
        'verbose': ...,
        'load_plan_path': ...,
        'save_plan_path': ...,
        'partition_constraints_path': ...,
        'recompute_modules': ...,
        'pipeline_pivots': ...,
        'use_apex_fused_adam_v2': ...,
    },
)
```

### Checkpoint support

You can save/load the checkpoints for parallel modules.
Each rank will save/load its own checkpoint just like the normal module.

Note: The only exception is the non-persistent buffers, which will remain uninitialized after loading the checkpoints, because they are not saved in the state dict. To make sure all the buffers are initialized, you must initialize the module with `init_params=True`.

You can also merge the checkpoints from different ranks to a single checkpoint.
We call it a merged checkpoint. The merged checkpoint can be loaded by original module directly.
So you can easily share the checkpoint with the original module.

On the other hand, a lot of weights/state in the module and the optimizer will be the same in the ranks in parallel training. So we can save a lot of space by deduping the state dicts before saving them to the disk.

We provide two functions to help you save/load the merged checkpoint for the parallel module,
and two other functions to help you save/load the deduped state dicts for the parallel module.

#### `merge_state_dicts`
```python
def merge_state_dicts(
    module_state_dicts: List[Dict[str, Any]],
    optimizer_state_dicts: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
```

Merge a list of shard state dicts (one for each rank) to a single full state dict
Note: Only Adam-like optimizers are supported for merging

Please Note:
    We don't guarantee the devices of tensors are the same in the merged state dict.
    You can assume the device of the tensors in the merged state dict can be one of the following:
        1. the current device when running this function
        2. the current cuda device when running this function
        3. the device of the tensor in the original state dict
    When you load the state dict from file, you can just use `torch.load(..., map_location='...')` to unify the device of the tensors.


#### `load_merged_state_dicts`
```python
def load_merged_state_dicts(
        module: torch.nn.Module,
        module_state_dict: Dict[str, Any],
        optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        *,
        device: Union[str, torch.device] = None
) -> None:
```
Load the merged state dicts to the module, and optionally the optimizer to a specified device.

Please note the `device` parameter. If it is None, we will use `torch.cuda.current_device()` to get the current device. If you want to load the state dict to a specific device, you can set it to the device you want.


#### `deduped_state_dicts`

In parallel training, a lot of weights/state in the module and the optimizer will be the same in the ranks. So we can save a lot of space by deduping the state dicts before saving them to the disk.

```python
def deduped_state_dict(
    module: torch.nn.Module,
    optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
```

#### `load_deduped_state_dict`

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

### Dataset

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
