# Paralleling a Module

nnScaler can transform a `torch.nn.Module` into a parallel module, which is a specialized version of `torch.nn.Module` capable of running across multiple GPUs or nodes. This process hides the complexity of distributed training and inference from the user.

Currently, we support three kinds of parallelism: data parallelism, tensor parallelism and pipeline parallelism. We can also combine them for better performance.

Data parallelism and tensor parallelism can be supported for any module, but pipeline parallelism is only supported for end2end modules for scheduling reason.

An end2end module is a module which satisfies:
- the first argument of `module.forward` is the data sample, and other arguments should have default value, and should never be used in `module.forward` function.
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

We will always call `optimizer.sync_shard_grad()` to sync the gradients before `optimizer.step()`, but in PP mode, the `sync_shard_grad` is a no-op because the gradients are already synced in the codegen, whereas in Non-PP mode, the `sync_shard_grad` will do the real synchronization.

In Non-PP mode, to support multiple calls of `optimizer.sync_shard_grad()`, `ParallelModule` will keep track whether the gradients are synced or not with `self._sync_grad_required` flag, and only sync the gradients when `self._sync_grad_required` is True. So you can call `optimizer.sync_shard_grad()` multiple times without worrying about it.

We also support async gradient reduction via `compute_config.use_async_reducer`. In this case, the gradient reduction will be kicked off once the gradients are ready, and `optimizer.sync_shard_grad()` will wait for the reduction to be done if it is called before the reduction is done.

When we combine async reduction with gradient accumulation, The time of kicking off gradient reduction becomes a problem. The current implementation is reusing `RuntimeFlag.skip_reducer` flag to control when to kick off the reduction. It is not ideal because `RuntimeFlag.skip_reducer` is originally designed for gradient accumulation, and it is not compatible when overlapping is used. So in overlapped scenarios, we must not use async reduction. We will improve it in the future.
