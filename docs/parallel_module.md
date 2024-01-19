# Parallel Module

Besides the support of end-to-end model training, Cube can also convert a `torch.nn.Module` to a parallel module. A parallel module is a special `torch.nn.Module` but runs in multiple gpus/nodes. All the complexity of distributed training/inferring is hidden from the user.

## An example

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

    reducer_op: str = 'sum'
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
3. Code generation feature configuration
    - use_zero: whether to use zero. If it is true, the generated code will use zero1 to do distributed training.
    - zero_ngroups: the number of groups to be used in zero.
    - reducer_op: the reducer operation for the gradients. It can be `sum`, `mean`, `min`, `max`, `avg`.
4. User configuration
    - user_config: the user configuration. A typical usage is deciding whether skipping compiling and reusing the previously compiled parallel module. If user_config is the same between two runs, compiling in the second run will be skipped.

Note:
1.  `reducer_op` represents which `torch.distributed.ReduceOp` is used when reduce gradients
    by torch.distributed.all_reduce or torch.distributed.reduce_scatter

    In some cases, you may want to firstly divide the local gradients, and then use torch.distributed.ReduceOp.SUM to get the final the gradients.
    You can achieve that speical mean with `optimizer.register_reducer_pre_hook` by setting `reducer_op` to `sum` and divide the local gradients with the following code:
    ```python
    def _mean_hook(reducer, grad):
      if reducer.reduce_op == torch.distributed.ReduceOp.SUM:
        grad.div_(reducer.ranks)
    optimizer.register_reducer_pre_hook(_mean_hook)
    ```
2.  You can put any graph related configuration here. The assumption is different user_config should generate different graph/code. So if the user config is changed, we will regenerate the graph/code automatically. Here are some examples:

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
3. MOO: MOO is short for 'match or override'. It will reuse if match, generate if not match or no previous gerenated code exists.
4. GRAPH: Reuse graph only if match, generate otherwise.


### Module Conversion

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
) -> Union[None, ParallelModule, Type[ParallelModule]]:
```
It has the following parameters:

- module_or_module_class (Union[torch.nn.Module, Type[torch.nn.Module]]): the module or module class to be compiled. Please note if the input is a module object, we will return a `ParalleModule` object. If the input is a module class, we will return a `ParalleModule` class.

- dummy_input (dict): the dummy input for the module

- pas_policy (Callable[[IRGraph, ComputeConfig], IRGraph]): the pas policy, which describes how to place all computations across devices. You can use `autodist` to do the pas automatically in the most efficient way.

- compute_config (ComputeConfig): the environment resource

- reuse (ReuseType): specify which part can be reused.

- cube_savedir (Union[str, Path]): the directory to save generated code

- instance_name (Optional[str]): the instance name of the generated module. If it is `None`, will use the default name.

- load_module (bool): whether to load the generated module or module class after conversion is done.
Currently the module can only be loaded in torchrun environment. So you can do the conversion in any environment (with `load_module` unset), and load the module in torchrun environment.

- init_module_params (bool): If true, when we construct the module, all its parameters are initialized with the same value with when we traced.
Otherwise, they will be empty tensor.
This parameter will be passed to the module constructor,
so it is only used when `module_or_module_class` is a module object, and `load_module` is true.

- module_dtype (Optional[torch.dtype]): the dtype of the module. Keep the module as it is if it is None.

- module_fn (Optional[Callable[[], torch.nn.Module]]): the function to create the module. Will use `__init__` if it is None. This parameter is only used when `module_or_module_class` is a module class.

Note:

1. This function can be used to convert both module object and module class to cube module or cube module class.
Among key-value arguments,
`module_fn` and `module_dtype` control how to create the module object.
whereas `init_module_params` controls how to load cube module object after conversion is done.

2. If you want to save multiple instances of the same module (with differnt configurations),
you can specify the `instance_name` to distingish them.

3. Currently you must use a shared file system to share the generated files (like mounted Azure Blob).
Or you can unset `load_module` flag, and manually copy the generated files to other nodes.
After all nodes have the generated files, you can call `parallelize()` again with `load_module` flag set.

4. if reuse is not set to ReuseType.MATCH,
the generated code in outdir will be removed EVEN IF the code generetion fails in this call.

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
- module (torch.nn.Module): the module to be optimized
- optimizer_fn (Union[Type[torch.optim.Optimizer], Callable[..., torch.optim.Optimizer]]):
    It can be the optimizer class or optimizer factory function.
- *args: the args will pass to `optimizer_fn`
- **kwargs: the kwargs will pass to `optimizer_fn`

To support distrubted training, in the function we need to hook 4 places:

1. optimizer constructor:
    the parameters of optimizer will not be the same with the parameters of the module if we use zero.
    So we need to replace the parameters of optimizer with `CubeModule.parameters_for_optimizer`.

2. `optimizer.step()`:
    we need to call `optimizer.sync_shard_grad()` to sync the gradients of the module before `optimizer.step()`.
    In zero mode (not supported yet), we have to call `CubeModule.gather_params()` after `optimizer.step()`

3. `optimizer.zero_grad()`:
    We need to call `CubeModule.zero_grad()` after `optimizer.zero_grad()`

`build_optimizer` will patch optimizer for you. Besides the above patches, we also add several utility functions to optimizer:

1. `sync_shard_grad`: Sync the shard gradients of the module from nodes with same shard to the optimizer. This function is called in optimizer's pre-step hook. But If you want to access the gradients before `optimizer.step()`(for example, you need gnorm),  you need to call this function manually.

2. `register_reducer_pre_hook`, `register_reducer_post_hook`: Register pre/post hooks to reducers which will be applied before/after gradient synchronization.

### Dataset

We use the same dataset/dataloader as pytorch. For example, you can use `torch.utils.data.DistributedSampler` to create a distributed sampler.

`ParallelModule`s running in the same unit should use the same input, and will get the same output. `ParallelModule`s runing in different units should use different input and will get different output (similar to data parallelism). The gradients of all parameters will be synced across all the devices automatically.

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
1. When ParallelModule is a submodule of another Module, Pytorch DDP is not supported yet.
2. Pipeline parallelism is not supported yet.
