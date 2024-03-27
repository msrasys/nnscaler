# Parallel Module

Besides the support of end-to-end model training, Cube can also convert a `torch.nn.Module` to a parallel module. A parallel module is a special `torch.nn.Module` but runs in multiple gpus/nodes. All the complexity of distributed training/inferring is hidden from the user.

## An example

- Example 1: Parallelize the whole module

```python
import torch
from cube.parallel import parallelize, ComputeConfig, build_optimizer

class LLM(torch.nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, x):
        ...

llm_sample_input = ...              # dummpy input will be used to do tracing
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
from cube.parallel import parallelize, ComputeConfig, build_optimizer

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

## APIs

### ComputeConfig
The configuration of the compute environment. It is a dataclass with the following fields:
```python
@dataclass(frozen=True)
class ComputeConfig:
    plan_ngpus: int
    runtime_ngpus: int

    dynamic_shape: bool = True

    use_zero: bool = False
    zero_ngroups: int = 1

    user_config: Optional[Dict[str, Any]] = None
```
We can categorize the fields into 4 categories:

1. Trace configuration
    - dynamic_shape: whether to use dynamic shape or static shape.
2. Compute environment configuration
    - plan_ngpus: the number of gpus to be used as a unit. The model is partitioned (TP or PP) within a unit, and then data parallelism is applied across multiple units. So every `plan_ngpus` devices holds the whole model. Furthermore, assume we have two workers, and their ranks are `rank1` and `rank2`:
        1. if `rank1 // plan_gpus == rank2 // plan_ngpus`, then they are in the same unit.
        2. If `rank1 % plan_ngpus == rank2 % plan_ngpus`, then the portion of model hold on both gpus are exactly the same.
    - runtime_ngpus: the number of gpus to be used in runtime. It should be a multiple of `plan_ngpus`, which means we have `runtime_ngpus // plan_ngpus` units in runtime, and the data parallelism is `runtime_ngpus // plan_ngpus`.
    Please note all modules must have the same `plan_ngpus` and `runtime_ngpus`.
3. Code generation feature configuration
    - use_zero: whether to use zero. If it is true, the generated code will use zero1 to do distributed training.
    - zero_ngroups: the number of groups to be used in zero.
4. User configuration
    - user_config: the user configuration. A typical usage is deciding whether skipping compiling and reusing the previously compiled parallel module. If user_config is the same between two runs, compiling in the second run will be skipped.

Note:
1.  You can put any graph related configuration here. The assumption is different user_config should generate different graph/code. So if the user config is changed, we will regenerate the graph/code automatically. Here are some examples:

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
        here we can set `user_config={'use_3d': module_config.use_3d}`,
        and we can be sure different use_3d config will never use the same generated code.

    - Example 2: save file stats
        If you want to track all related file stats (just like traditional compilers do),
        you can do
        ```python
        user_config = {
            'file_stats': {
                str(f): os.stat(f).st_mtime_ns for f in Path('./src').glob('**/*.py')  # assume all source code is in ./src
            }
        }
        ```
        Or you can save the md5 of the files to save some bytes:
        ```python
        import hashlib
        h = hashlib.md5()
        for f in Path('./src').glob('**/*.py'):
        with open(f, 'rb') as f:
            h.update(f.read())
        user_config = {
            'files_md5': h.hexdigest()
        }
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

1. MATCH: Reuse if match, error if not match, generate if no previous gerenated code exists.
2. OVERRIDE: Nothing will be reused. Everything will be regenerated.
3. MOO: MOO is short for 'match or override'. It will reuse if match, generate if not match or no previous generated code exists.
4. GRAPH: Reuse graph only if match, generate otherwise.

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

2. `ALL`: broadcast all the generated files to all nodes.
    This is useful when you want to run the same code on all nodes.
    please note the init weight files can be huge.

3. `NO_WEIGHTS`: broadcast all except init weights.
    Without weights, you can only construct the parallel module with `init_params=False`.
    You can then
    - Load the weights from a checkpoint file with `module.load_state_dict`, `load_merged_state_dict`
        or `load_deduped_state_dict`
    - Or you can use `broadcast_weights` to get the weights from the workers in node0.
       (local world size should be bigger than plan_ngpus)

4. `CODE`: broadcast the new generated code only
    It's your responsibility to make sure other necessary files are available on all nodes.

Here are some guidelines to choose the strategy:

1. When restarting a training and there is a successful previous run: As we have a previous run, the compiling process has been done before. So there will be no new generated files and no broadcast will happen no matter what this option is. Please be sure the reuse flag of `parallelize` is `MATCH`, so we can make sure the generated code is the same with the previous run.

2. When training a model from scratch. If there is only one node, `none` is good enough.
If there are multiple nodes, here are some strategies:

a. If use `none`, the user should run `parallelize(..., load_module=False, ..)`, and then copy all files to all nodes manually, so all nodes have the same files. Then the user load the module by running `parallelize(..., load_module=True, ..)`.

b. if they are using a NAS-like device to save generated files, and the upload/download speed is fast in the cluster, they can also use `none`, and just run `parallelize(..., load_module=True, ..)` to do the training.

c. If use `all`, then user can just run `parallelize(..., load_module=True, ..)` safely. (remember to set `nccl` communication timeout to a very big value to tolerate the duration of this `nccl` broadcast)

d. If use `no_weights`. then user can run `parallelize(..., load_module=True, init_module_params=rank<plan_ngpus, ..)`. After module is loaded, the user should call `broadcast_weights(plan_ngpus)` manually to synchronize the module weights before training (Note all submodules have the same `plan_ngpus`).
```python
class Module(torch.nn.Module):
    ...
plan_ngpus = ...
rank = torch.distributed.get_rank()
local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', default=1))
assert local_world_size < plan_ngpus

parallel_module = parallelize(Module(), ..., load_module=True, init_module_params=rank<plan_ngpus, broadcast_strategy='no_weights, ...)

broadcast_weights(parallel_module, plan_ngpus)
# now the module is ready to train
```

e. Currently `code` option is provided just for completeness. Do not suggest users to use.

### Module Parallelization

We have `parallelize` function to Convert a torch.nn.Module to a ParallelModule.
```python
def parallelize(
    module_or_module_class: Union[torch.nn.Module, Type[torch.nn.Module]],
    dummy_input: dict,
    pas_policy: Callable[[IRGraph, ComputeConfig], IRGraph],
    compute_config: ComputeConfig,
    *,
    cube_savedir: Union[str, Path] = './.cube',
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

- `dummy_input` (`dict`): the dummy input for the module

- `pas_policy` (`Callable[[IRGraph, ComputeConfig], IRGraph]`): the pas policy, which describes how to place all computations across devices. You can use `autodist` to do the pas automatically in the most efficient way.

- `compute_config` (`ComputeConfig`): the environment resource

- `reuse` (`ReuseType`): specify which part can be reused.

- `cube_savedir` (`Union[str, Path]`): the directory to save generated code

- `instance_name` (`Optional[str]`): the instance name of the generated module. If it is `None`, will use the default name `_`.

- `load_module` (`bool`): whether to load the generated module or module class after parallelization is done.
Currently the module can only be loaded in `torchrun` environment. So you can do the parallelization in any environment (with `load_module` unset), and load the module in `torchrun` environment.

- `init_module_params` (`bool`): If true, when we construct the module, all its parameters are initialized with the same value with when we traced.
Otherwise, they will be empty tensor.
This parameter will be passed to the module constructor,
so it is only used when `module_or_module_class` is a module object, and `load_module` is true.

- `module_dtype` (`Optional[torch.dtype]`): the dtype of the module. Keep the module as it is if it is None.

- `module_fn` (`Optional[Callable[[], torch.nn.Module]]`): the function to create the module. Will use `__init__` if it is None. This parameter is only used when `module_or_module_class` is a module class.

- `broadcast_strategy` (`Union[str, BroadcastGenFilesStrategy]`): the broadcast strategy for new generated files.

Note:

1. This function can be used to convert both module object and module class to cube module or cube module class.
Among key-value arguments,
`module_fn` and `module_dtype` control how to create the module object.
whereas `init_module_params` controls how to load cube module object after parallelization is done.

2. If you want to save multiple instances of the same module (with different configurations),
you can specify the `instance_name` to distinguish them.

3. `load_module` flag should be used with `broadcast_strategy`. See more details in the `BroadcastGenFilesStrategy` section.

4. if `reuse` is not set to `ReuseType.MATCH`,
the generated code in outdir will be removed EVEN IF the code generation fails in this call.

5. For `broadcast_strategy`, Please note that the broadcast will only be done in `torchrun` environment, and will throw an error if `torch.distributed` is not initialized and `broadcast_strategy` is not `NONE`.

After the module is converted, you can use it to create module object by calling it like a module class.
The module class is defined like:
```python
class GenModule(cube.runtime.module.ParallelModule):
    def __init__(self, init_params=True):
        super().__init__()
        ...
    ...
```
So you can use `init_params` in `__init__` to control whether to initialize the module parameters.
For example, if you don't want to intialize module params:
```python
module = GenModule(init_params=False)
```


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

To support distributed training, in the function we need to hook 4 places:

1. optimizer constructor:
    the parameters of optimizer will not be the same with the parameters of the module if we use zero.
    So we need to replace the parameters of optimizer with `CubeModule.parameters_for_optimizer`.

2. `optimizer.step()`:
    we need to call `optimizer.sync_shard_grad()` to sync the gradients of the module before `optimizer.step()`.
    In zero mode (not supported yet), we have to call `CubeModule.gather_params()` after `optimizer.step()`

3. `optimizer.zero_grad()`:
    We need to call `CubeModule.zero_grad()` after `optimizer.zero_grad()`

`build_optimizer` will patch optimizer for you. Besides the above patches, we also add several utility functions/variables to optimizer:

1. `sync_shard_grad`: Sync the shard gradients of the module from nodes with same shard to the optimizer. This function is called in optimizer's pre-step hook. But If you want to access the gradients before `optimizer.step()`(for example, you need gnorm),  you need to call this function manually.

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

### Checkpoint support

You can save/load the checkpoints for parallel modules.
Each rank will save/load its own checkpoint just like the normal module.

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

## TODOs

1. Pipeline parallelism is not supported yet.
