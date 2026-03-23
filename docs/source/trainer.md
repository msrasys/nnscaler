# Native Trainer

`nnScaler` provides a `Trainer` class for training and evaluating model
parallelization. Let's start from an example to demonstrate how to
parallelize a model using the `parallelize` API. Next, we'll illustrate
how to train the model across multiple GPUs using the provided dataset
and optimizer.

## Arguments

All the arguments are defined in `TrainerArgs` class. Here is the
definition of `TrainerArgs`:

``` python
@dataclass
class TrainerArgs:
    init_module: Optional[str] = None
    vars: Dict[str, Any] = field(default_factory=dict)
    compute_config: ComputeConfig = None

    gen_savedir: str = './.nnscaler'
    gen_reuse: str = 'auto'
    pas_policy: str = 'autodist'
    broadcast_strategy: str = 'all'
    instance_name: Optional[str] = None
    run_mode: str = 'run'
    dummy_sample_gen_fn: Optional[Callable[['TrainerArgs'], Any]] = None
    dummy_sample_post_process_fn: Optional[Callable[['TrainerArgs', Any], Any]] = None
    tracing_from_weights: str = None

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    dataset_sampler: Optional[DatasetSamplerConfig] = None
    lr_scheduler: Optional[LRSchedulerConfig] = None
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    log: List[LogConfig] = field(default_factory=list)
    hook: Union[HookConfig, HookMapConfig, None] = None

    debug: DebugConfig = field(default_factory=DebugConfig)

    precision: Union[str, Dict[_TENSOR_TYPE, _PRECISION_TYPE], None] = None

    micro_batch_size: int = 1
    global_batch_size: Optional[int] = None
    grad_accumulation_steps: Optional[int] = None

    max_epochs: Optional[int] = None
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None

    val_every_n_train_steps: Optional[int] = None
    val_every_n_epochs: Optional[int] = 1

    enable_progress_bar: bool = True
    log_progress_every_n_train_steps: Optional[int] = 100

    seed: Optional[int] = None
    init_env_fn: str = None
```

The design philosophy of `Trainer` arguments is: The classes(or factory
functions) of components(model/optimizer/etc) and their arguments are
provided in the `TrainerArgs` class (functions/types are passed as fully
qualified names), and we are responsible for creating them.

For example, you can tell me how to create a model by providing the
model type and its arguments in `ModelConfig` class.

Please note some of the arguments of components are set automatically,
and you should not set them manually. For example, arguments `dataset`,
`num_replicas` and `rank` of the dataset sampler are set automatically
by the `Trainer` class. Those 3 arguments passed in the
`DatasetSamplerConfig.train_args/val_args` (if any) will be ignored.

``` python
'dataset': {
    'type': 'SomeDataset',
    'train_args': {
        ...
    },
    'val_args': {
        ...
    }
}
'dataset_sampler': {
    'type': 'SomeDatasetSampler',
    'train_args': {
        'num_replicas': ...,  # this will be ignored
        'dataset': ...,       # this will be ignored
        'rank': ...,          # this will be ignored
        ...
    },
    'val_args': {
        'num_replicas': ...,  # this will be ignored
        'dataset': ...,       # this will be ignored
        'rank': ...,          # this will be ignored
        ...
    },
}
```

If any argument type is a class, you can pass it as a dict, and add a
special key `__type` to specify the class type.

For example, if the module `__init__` takes `ModelConfig` object

``` python
class SomeModule(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        ...
```

You can pass the `model_config` as

``` python
{
    'type': 'SomeModule',
    'args': {
        'model_config': {
            '__type': 'ModelConfig',
            # arguments to create ModelConfig
        }
    }
}
```

We also use `ast.literal_eval` to guess the type of the string
arguments, You can skip it by passing a dict with `__value_type` and
`value` keys. For example, you want a number to be a str, you can use

``` python
{
    '__value_type': 'str',
    'value': '1'
}
```

Internally we will get the final value with `__value_type(value)`.

If any of the arguments are dynamic (for example, the `instance_name` you want to use depends on other arguments), you can use `__type` to specify the function to generate the argument. For example, you can specify `instance_name` as

``` python
{
  ...
  'instance_name': {
      '__type': 'fqn.generate_instance_name',
      'arg1': ...,
      'arg2': ...,
  }
  ...
}
```

Note: `__type` and `__value_type` are done during config parsing, resolved references (`$!(...)`) are not supported in the arguments of `__type` and `__value_type`.

### Component Configs

- `model` (`ModelConfig`): The model to be trained. You need to provide
  the model type and its arguments in `ModelConfig` class. Here is the
  definition of `ModelConfig`:

  ``` python
  @dataclass
  class ModelConfig:
      type: str = None
      args: Dict[str, Any] = field(default_factory=dict)
      parallel_modules: list[ModuleParallelizeConfig] = field(default_factory=list)
  ```

  - `type` (`str`): The model type. Note: It can't be a factory
    function.

  - `args` (`Dict[str, Any]`): The arguments of the model's `__init__`
    function.

  - `parallel_modules` (`List[ModuleParallelizeConfig]`): The sub
    modules to be parallelized. If this is not empty, these modules will
    be parallelized instead of the whole model. i.e. sub modules (in the
    list of `parallel_modules`) in the model will be replaced with
    parallelized version. The parallelized sub modules must be unique by
    type.

    Note: When parallel_modules is not empty, pipeline parallelism is
    not supported as the model is not end-to-end parallelized any more.

  ``` python
  @dataclass(frozen=True)
  class OptionalComputeConfig:
      constant_folding: Optional[bool] = None
      trace_strategy: Optional[str] = None
      use_zero: Optional[bool] = None
      zero_ngroups: Optional[int] = None
      zero_use_reduce_scatter: Optional[bool] = None
      use_async_reducer: Optional[bool] = None
      reducer_bucket_cap_mb: Optional[float] = None

      pas_config: Optional[Dict[str, Any]] = None
      user_config: Optional[Dict[str, Any]] = None
  ```

This is an optional version of the `ComputeConfig`. Please refer to
`ComputeConfig` for more
information.

 ``` python
 @dataclass
 class ModuleParallelizeConfig:
     type: str = None
     args: Optional[Dict[str, Any]] = None
     forward_args_gen_fn: Optional[Callable[['TrainerArgs'], dict[str, Any]]] = None
     forward_args_post_process_fn: Optional[Callable[['TrainerArgs', dict[str, Any]], dict[str, Any]]] = None
     tracing_from_weights: str = None
     tracing_from_weights_prefix: str = None

     # For the following config, If None, the config of the trainer_args will be used
     compute_config: Optional[OptionalComputeConfig] = None
     gen_savedir: Optional[str] = None
     gen_reuse: Optional[str] = None
     pas_policy: Optional[str] = None
     broadcast_strategy: Optional[str] = None
     instance_name: Optional[str] = None
     precision: Union[str, Dict[_TENSOR_TYPE, _PRECISION_TYPE], None] = None
 ```

 - `type` (`str`): The sub model type to be parallelized. Note: It
   can't be a factory function.
 - `args` (`Optional[Dict[str, Any]]`): The arguments of the model's
   `__init__` function. If `run_mode` is `'compile'` and `args` is not
   `None`, we can parallelize submodules without creating the whole
   model when this arugment is provided.
 - `forward_args_gen_fn` (`str`): The full qualified name of the
   function to generate dummy forward args. Its type should be
   `Callable[[TrainerArgs],Dict[str, Any]]`. The function should return
   a dict of dummy forward args for the model. This is required.
 - `forward_args_post_process_fn` (`str`): The full qualified name of
   the function to post process the dummy forward args. Its type should
   be `Callable[[TrainerArgs, Dict[str, Any]], Dict[str, Any]]`. The
   input tensors have already been moved to GPU and converted to
   `input_dtype`. You can use this to further process the sample, for
   example, to mark some dims of tensors as dynamic.
 - `tracing_from_weights` (`str`): The path to the weights to be loaded
   when tracing(compiling) the model. It is only used in tracing to
   serve as the initial state dict of the model. Default is `None`.
 - `tracing_from_weights_prefix` (`str`): the prefix in the state dict
   (loaded from `trainer_args.tracing_from_weights`) to be used for
   tracing. Please note `trainer_args.tracing_from_weights` must be set
   if you want to use this, and `tracing_from_weights` and
   `tracing_from_weights_prefix` shouldn't be set at the same time.
 - `compute_config` (`Optional[OptionalComputeConfig]`): The compute
   config for the parallelized module. The merged config with the
   compute config of the `trainer_args.compute_config` will be used.
 - `gen_savedir` (`Optional[str]`): The directory to save the generated
   files. If None, the config of the trainer_args will be used. You can
   find more information below.
 - `gen_reuse` (`Optional[str]`): The reuse strategy of the generated
   code. If None, the config of the trainer_args will be used. You can
   find more information below.
 - `pas_policy` (`Optional[str]`): The policy of parameter
   partitioning. If None, the config of the trainer_args will be used.
   You can find more information below.
 - `broadcast_strategy` (`Optional[str]`): The strategy of broadcasting
   the model. If None, the config of the trainer_args will be used. You
   can find more information below.
 - `instance_name` (`Optional[str]`): The instance name of the trainer.
   If None, the config of the trainer_args will be used. You can find
   more information below.
 - `precision`
   (`Union[str, Dict[_TENSOR_TYPE, _PRECISION_TYPE], None]`): The
   precision of the model. If None, the config of the trainer_args will
   be used. You can find more information below.

Please Note:

1.  The parallelization is per-module-type, which means one
    module type can only be parallelized once. Moreover, the initial weights
    of the parallelized modules with the same type are all the same.

    So if you want to parallelize a module multiple times (with different
    arguments or different inital weights), you need to create an alias
    for it.

    For example, if you want to parallelize a module named `SomeModule` twice, you can create an alias for it:

    ``` python
    class SomeModuleAlias(SomeModule):
        pass
    ```

2. The initial weights of the whole model will be different when sub
   module parallelization is enabled, since parallelization process
   will change the `rng_state` of torch.

    To make the initial weights of the whole model the same as the
    original model, We recommend to save the initial weights of the
    original model and load them before training.

- `optimizer` (`OptimizerConfig`): The optimizer to be used.

  ``` python
  @dataclass
  class OptimizerConfig:
      type: str = None
      args: Dict[str, Any] = field(default_factory=dict)
      clip_gnorm: float = 0.0

      param_clss_fn: Optional[Callable[[str], Any]] = None

      loss_reduction: str = 'mean'
      grad_reduction: str = 'mean'
      grad_reduce_divisor: Optional[float] = None
      aggregate_outputs_fn: str = None
  ```

  - `type` (`str`): The optimizer type or factory function. Please note
    the first parameter of the optimizer constructor must be the model
    parameters.
  - `args` (`Dict[str, Any]`): The arguments of the optimizer.
  - `clip_gnorm` (`float`): The maximum norm value for gradient
    clipping. 0.0/None means no clipping.
  - `param_clss_fn` (`Optional[Callable[[str], Any]]`): A function
    to classify parameters. It takes a parameter name and returns a
    classification. Parameters with the same classification will be
    grouped together. When set, bucket building will be delayed.
  - `loss_reduction` (`str`): The reduction method for loss. It can be
    `mean` (average the loss over all micro-batches), `sum` (sum the
    loss of all micro-batches), or `per-token-mean` (average the loss
    over all tokens). Default is `mean`. When `per-token-mean` is used,
    you need to specify `aggregate_outputs_fn`. Please note in
    validation stage, this configuration is ignored and the loss is
    always averaged over all batches.
  - `grad_reduction` (`str`): The reduction method for gradients. It can
    be `mean` (average the gradients over all micro-batches), `sum` (sum
    the gradients of all micro-batches), `per-token-mean` (average the
    gradients over all tokens). Default is `mean`. Please note if
    `per-token-mean` is used, you need to specify
    `aggregate_outputs_fn`, which will return the number of tokens.
  - `grad_reduce_divisor` (`Optional[float]`): The divisor applied to
    gradients before all-reduce. If not set, the default divisor is
    `runtime_ngpus / plan_ngpus`. We divide the gradients to avoid
    overflow. However, if the gradients are in high precision or the user
    has known the range of the gradients, a smaller divisor can be set to
    improve accuracy. The gradients will be recovered by multiplying
    the divisor after all-reduce and before the optimizer step.
  - `aggregate_outputs_fn` (`str`): The function to aggregate the
    outputs of the model. It is required when `grad_reduction` is
    `per-token-mean`. Its signature should be
    `def aggregate_outputs(self, loss_outputs, sync_group) -> AggregatedOutputs`,
    where `loss_outputs` is a list of outputs of the model, and
    `sync_group` is the `torch.distributed.ProcessGroup` to sync with.
    The function should return an `AggregatedOutputs` object, which
    defines as:

  ``` python
  @dataclass
  class AggregatedOutputs:
      # the aggregated loss as a sum
      loss_sum: float = None
      # number of mini batches
      num_batches: int = None
      # number of tokens (necessary when grad_reduction is 'per-token-mean')
      num_tokens: Optional[int] = None
      # any other custom outputs
      aggregated_outputs: Any = None

      @classmethod
      def aggregate(cls,
          loss_outputs: list,
          sync_group: torch.distributed.ProcessGroup,
          loss_fn: Callable[[Any], torch.Tensor],
          ntokens_fn: Callable[[Any], torch.Tensor] | None = None,
      ) -> 'AggregatedOutputs':
          ...
  ```

  `AggregatedOutputs` also provides a convenience classmethod
  `aggregate` that handles loss/token aggregation and all-reduce across
  ranks. You can use it in your `aggregate_outputs_fn`:

  - `loss_outputs`: The list of outputs from the model.
  - `sync_group`: The `torch.distributed.ProcessGroup` to sync with.
  - `loss_fn`: A function to extract the loss tensor from each output.
  - `ntokens_fn`: A function to extract the number of tokens from each
    output. Required when `grad_reduction` is `'per-token-mean'`.

- `dataset` (`DatasetConfig`): The dataset to be used.

  ``` python
  @dataclass
  class DatasetConfig:
      type: str = None
      train_args: Dict[str, Any] = field(default_factory=dict)
      val_args: Dict[str, Any] = field(default_factory=dict)
      test_args: Dict[str, Any] = field(default_factory=dict)
  ```

  - `type` (`str`): The dataset type or factory function.
  - `train_args` (`Dict[str, Any]`): The arguments of the training
    dataset.
  - `val_args` (`Dict[str, Any]`): The arguments of the validation
    dataset.
  - `test_args` (`Dict[str, Any]`): The arguments of the test
    dataset.

- `dataloader` (`DataloaderConfig`): The dataloader to be used. Please
  note we recommend to pass `drop_last=True` in the dataloader arguments
  to avoid the last batch with different sizes.

  ``` python
  @dataclass
  class DataloaderConfig:
      type: str = 'torch.utils.data.DataLoader'
      train_args: Dict[str, Any] = field(default_factory=dict)
      # default to train_args
      val_args: Dict[str, Any] = field(default_factory=dict)
      # default to train_args
      test_args: Dict[str, Any] = field(default_factory=dict)
  ```

  - `type` (`str`): The dataloader type or factory function. Please note
    the dataloader constructor must at least have 3 parameters
    `dataset`, `batch_size`, `sampler`.
  - `train_args` (`Dict[str, Any]`): The arguments (except
    `dataset`,`batch_size`, `sampler`) of the training dataloader.
    Argument `batch_size` will be set to `micro_batch_size`.
  - `val_args` (`Dict[str, Any]`): The arguments (except
    `dataset`,`batch_size`, `sampler`) of the validation dataloader.

- `dataset_sampler` (`Optional[DatasetSamplerConfig]`): The dataset
  sampler to be used. If not set, `torch.utils.data.DistributedSampler`
  will be used by default.

  ``` python
  @dataclass
  class DatasetSamplerConfig:
      type: str = 'torch.utils.data.DistributedSampler'
      train_args: Dict[str, Any] = field(default_factory=dict)
      val_args: Dict[str, Any] = field(default_factory=dict)
      test_args: Dict[str, Any] = field(default_factory=dict)
  ```

  - `type` (`str`): The dataset sampler type or factory function. Please
    note the dataset sampler constructor must at least have 3 parameters
    `dataset`, `num_replicas`, `rank`.
  - `train_args` (`Dict[str, Any]`): The arguments (except
    `dataset`,`num_replicas`, `rank`) of the training dataset sampler.
  - `val_args` (`Dict[str, Any]`): The arguments (except
    `dataset`,`num_replicas`, `rank`) of the validation dataset sampler.

  Note: `IterableDataset` does not support sampler. If you use
  `IterableDataset`, please remove `dataset_sampler` from `TrainerArgs`.

- `lr_scheduler` (`LRSchedulerConfig`): The learning rate scheduler to
  be used. This is optional.

  ``` python
  @dataclass
  class LRSchedulerConfig:
      type: str = None
      args: Dict[str, Any] = field(default_factory=dict)
      interval: str = 'epoch'
  ```

  - `type` (`str`): The learning rate scheduler type or factory
    function. Please note the first parameter of the learning rate
    scheduler constructor must be optimizer.
  - `args` (`Dict[str, Any]`): The arguments of the learning rate
    scheduler.
  - `interval` (`str`): The interval to update the learning rate. It can
    be `epoch` or `step`. Default is `epoch`.

- `log` (`List[LogConfig]`): The loggers to be used. You can provide
  multiple loggers. Currently we have two builtin loggers:
  `TensorBoardLogger` and `WandbLogger`.

  ``` python
  @dataclass
  class LogConfig:
      type: str = None
      args: Dict[str, Any] = field(default_factory=dict)
  ```

  - `type` (`str`): The logger type or factory function. For built-in
    loggers, you can use short names like `TensorBoardLogger` or
    `WandbLogger` (they will be expanded to
    `nnscaler.cli.loggers.TensorBoardLogger`, etc.).
  - `args` (`Dict[str, Any]`): The arguments of the logger.

- `hook` (`Union[HookConfig, HookMapConfig, None]`): The hooks to be
  used. You can provide a hook with a hook class or a map of hook
  functions. Please note if your `model`/`optimizer`/`lr_scheduler`
  inherit from `TrainHook`, their hook functions will be called
  automatically. The order of the hook functions called is `model` ->
  `optimizer` -> `lr_scheduler`, and hooks passed with this config is
  called in the last.

  Hook class:

  ``` python
  @dataclass
  class HookConfig:
      type: str = None
      args: Dict[str, Any] = field(default_factory=dict)
  ```

  - `type` (`str`): The hook type or factory function.
  - `args` (`Dict[str, Any]`): The arguments of the hook.

  Hook map:

  ``` python
  @dataclass
  class HookMapConfig:
      after_setup: str = None
      on_finalize: str = None

      on_train_start: str = None
      on_train_end: str = None
      on_val_start: str = None
      on_val_end: str = None

      on_epoch_start: str = None
      on_epoch_end: str = None

      on_step_start: str = None
      on_step_end: str = None

      on_train_step_start: str = None
      on_train_step_end: str = None
      on_val_step_start: str = None
      on_val_step_end: str = None

      after_aggregate_train_step_outputs: str = None
      after_aggregate_val_step_outputs: str = None

      before_zero_grad: str = None
      after_zero_grad: str = None

      before_sync_grad: str = None
      after_sync_grad: str = None

      before_gnorm_clip: str = None
      after_gnorm_clip: str = None

      before_optimizer_step: str = None
      after_optimizer_step: str = None

      before_log_train_metrics: str = None
      before_log_val_metrics: str = None

      on_load_checkpoint: str = None
      after_load_checkpoint: str = None
      on_save_checkpoint: str = None
      on_expire_checkpoint: str = None
  ```

  - `after_setup` (`str`): The hook function to be called after setting
    up the trainer. Only be called when `run_mode == 'run'`. Signature:
    `def after_setup(trainer: 'Trainer') -> None:`
  - `on_finalize` (`str`): The hook function to be called when the
    training is done. Signature:
    `def on_finalize(trainer: 'Trainer') -> None:`
  - `on_train_start` (`str`): The hook function to be called at the
    start of the training stage. Signature:
    `def on_train_start(trainer: 'Trainer') -> None:`
  - `on_train_end` (`str`): The hook function to be called at the end of
    the training stage. Signature:
    `def on_train_end(trainer: 'Trainer') -> None:`
  - `on_val_start` (`str`): The hook function to be called at the start
    of the validation stage. Signature:
    `def on_val_start(trainer: 'Trainer') -> None:`
  - `on_val_end` (`str`): The hook function to be called at the end of
    the validation stage. Signature:
    `def on_val_end(trainer: 'Trainer', val_loss: float) -> None:`
  - `on_epoch_start` (`str`): The hook function to be called at the
    start of each epoch. Signature:
    `def on_epoch_start(trainer: 'Trainer', epoch: int) -> None:`
  - `on_epoch_end` (`str`): The hook function to be called at the end of
    each epoch. Signature:
    `def on_epoch_end(trainer: 'Trainer', epoch: int) -> None:`
  - `on_train_step_start` (`str`): The hook function to be called at the
    start of each training step. Signature:
    `def on_train_step_start(trainer: 'Trainer', batches: List[Any]) -> None:`
  - `on_train_step_end` (`str`): The hook function to be called at the
    end of each training step. Signature:
    `def on_train_step_end(trainer: 'Trainer', outputs: List[Any]) -> None:`
  - `on_val_step_start` (`str`): The hook function to be called at the
    start of each validation step. Signature:
    `def on_val_step_start(trainer: 'Trainer', batches: List[Any]) -> None:`
  - `on_val_step_end` (`str`): The hook function to be called at the end
    of each validation step. Signature:
    `def on_val_step_end(trainer: 'Trainer', outputs: List[Any]) -> None:`
  - `on_step_start` (`str`): The hook function to be called at the start
    of each step. Signature:
    `def on_step_start(self, trainer: 'Trainer', epoch: int, idx: int) -> None:`
  - `on_step_end` (`str`): The hook function to be called at the end of
    each step. Signature:
    `def on_step_end(self, trainer: 'Trainer', epoch: int, idx: int, step_metrics: TrainStepMetrics, aggregated_outputs: 'AggregatedOutputs') -> None:`
  - `after_aggregate_train_step_outputs` (`str`): The hook function to
    be called after aggregating the outputs of the model in the training
    step. Signature:
    `def after_aggregate_train_step_outputs(trainer: 'Trainer', aggregated_outputs: 'AggregatedOutputs', train_loss: float) -> None:`
  - `after_aggregate_val_step_outputs` (`str`): The hook function to be
    called after aggregating the outputs of the model in the validation
    step. Signature:
    `def after_aggregate_val_step_outputs(trainer: 'Trainer', aggregated_outputs: 'AggregatedOutputs', val_loss: float) -> None:`
  - `before_zero_grad` (`str`): The hook function to be called before
    zeroing the gradients. Signature:
    `def before_zero_grad(trainer: 'Trainer') -> None:`
  - `after_zero_grad` (`str`): The hook function to be called after
    zeroing the gradients. Signature:
    `def after_zero_grad(trainer: 'Trainer') -> None:`
  - `before_sync_grad` (`str`): The hook function to be called before
    syncing the gradients between ranks. Please note this hook can't be
    triggered correctly, and you should not reply on this. Will fix it
    later. Signature:
    `def before_sync_grad(trainer: 'Trainer') -> None:`
  - `after_sync_grad` (`str`): The hook function to be called after
    syncing the gradients between ranks. Signature:
    `def after_sync_grad(trainer: 'Trainer') -> None:`
  - `before_gnorm_clip` (`str`): The hook function to be called before
    gradient clipping. Signature:
    `def before_gnorm_clip(trainer: 'Trainer') -> None:`
  - `after_gnorm_clip` (`str`): The hook function to be called after
    gradient clipping. Signature:
    `def after_gnorm_clip(trainer: 'Trainer', gnorm: torch.Tensor) -> None:`
  - `before_optimizer_step` (`str`): The hook function to be called
    before the optimizer step. Signature:
    `def before_optimizer_step(trainer: 'Trainer') -> None:`
  - `after_optimizer_step` (`str`): The hook function to be called after
    the optimizer step. Signature:
    `def after_optimizer_step(trainer: 'Trainer') -> None:`
  - `before_log_train_metrics` (`str`): The hook function to be called
    before logging the training metrics. You can use this to modify the
    training metrics before logging. Signature:
    `def before_log_train_metrics(self, trainer: 'Trainer', step_metrics: TrainStepMetrics, aggregated_outputs: 'AggregatedOutputs') -> None:`
  - `before_log_val_metrics` (`str`): The hook function to be called
    before logging the validation metrics. You can use this to modify
    the validation metrics before logging. Signature:
    `def before_log_val_metrics(self, trainer: 'Trainer', metrics: ValMetrics) -> None:`
  - `on_load_checkpoint` (`str`): The hook function to be called after
    loading the checkpoint. If you saved something with
    `on_save_checkpoint` this is your chance to restore this. Please
    note when checkpoints are merged, the custom data saved in the
    checkpoint will be collected and saved as array in merged
    checkpoint. You must handle this case. Signature:
    `def on_load_checkpoint(trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:`
  - `after_load_checkpoint` (`str`): The hook function to be called
    after setting model/optimizer/etc from checkpoint. You can use this
    to restore some states for model/optimizer/etc that are not saved in
    the checkpoint. Signature:
    `def after_load_checkpoint(trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:`
  - `on_save_checkpoint` (`str`): The hook function to be called before
    saving the checkpoint. If you want to save something, you can add it
    to the checkpoint here. Signature:
    `def on_save_checkpoint(trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:`
  - `on_expire_checkpoint` (`str`): The hook function to be called
    before expiring (deleting) a checkpoint. If you want to do something
    before a checkpoint is deleted, you can do it here. Note: only
    local-rank 0 will call this hook. Signature:
    `def on_expire_checkpoint(trainer: 'Trainer', step: int, checkpoint_dir: Path) -> None:`

### Compute Config

All compute configs are put in `compute_config` (`ComputeConfig`).
Please refer to [`ComputeConfig`](#appendix-computeconfig) for more information.

Please note only end2end mode is supported in the trainer, so you must
set `compute_config.use_end2end` to `True` to make it work.

An end2end module is a module which satisfies:

- the first argument of `module.forward` is the data sample, and every
  other argument should have default value, and use its default value in
  `module.forward` function.
- the first return value of `module.forward` is the loss (scalar tensor)

### Checkpoint Config

 ``` python
 @dataclass
 class CheckpointConfig:
     save_dir: str = './checkpoints'
     no_save: bool = False

     format: str = 'pt'
     serializer: Optional[SerializerOptions] = None

     save_type: str = 'sharded'

     save_last: bool = True
     save_best: bool = True
     symlink_best_and_last: bool = True

     every_n_train_steps: Optional[int] = None
     every_n_epochs: Optional[int] = None
     keep_last_n_checkpoints: Optional[int] = None

     resume_from: Optional[ResumeOptions] = None
 ```

Where `ResumeOptions` and `SerializerOptions` are:

 ``` python
 @dataclass
 class ResumeOptions:
     checkpoint: str = 'last'
     convert_fn: Optional[str] = None
     with_merged: Optional[bool] = None
     save_memory: bool = True
 ```

 ``` python
 @dataclass
 class SerializerOptions:
     name: str = ''
     args: Dict[str, Any] = field(default_factory=dict)
 ```

- `save_dir` (`str`): The directory to save the checkpoints.
- `no_save` (`bool`): Whether to save the checkpoints. Default is
  `False`.
- `format` (`str`): The checkpoint format. It can be `"pt"` (PyTorch
  native format) or `"safetensors"` (Safetensors format). You can also
  register new formats via `nnscaler.cli.serialization.register_format`.
  Default is `"pt"`.
- `serializer` (`Optional[SerializerOptions]`): The serialization runner
  to use. If `None`, the default serializer will be used.
  - `name` (`str`): The name of the registered serialization runner.
  - `args` (`Dict[str, Any]`): The arguments for the serialization runner.
  You can also pass a string directly (e.g. `serializer: "runner_name"`)
  as a shorthand.
- `save_type` (`str`): The type of saving checkpoint. It can be
  `sharded` or `deduped`. Default is `sharded`.
  - `"sharded"`: Each rank saves its shard of weights and optimizer
    states to a file. The checkpoint is a folder with as many files as
    the world size.
  - `"deduped"`: Each rank saves its deduped shard of weights and
    optimizer states to a file. The checkpoint is a folder with as many
    files as the world size.
  - `"merged"`: everything has been merged into a single file. Used
    internally only when you merge the checkpoint files via
    `Trainer.merge_checkpoints`
- `save_last` (`bool`): Whether to save the last checkpoint. Default is
  `True`.
- `save_best` (`bool`): Whether to save the best (lowest `val_loss`)
  checkpoint. Default is `True`.
- `symlink_best_and_last` (`bool`): Whether to use symlink (instead of
  copy) to the best and last checkpoint. Default is `True`.
- `every_n_train_steps` (`Optional[int]`): Save the checkpoint every
  `every_n_train_steps` training steps. Default is `None`. Please note
  when neither `every_n_train_steps` nor `every_n_epochs` is specified,
  `every_n_epochs` defaults to `1`. You cannot specify both at the same
  time.
- `every_n_epochs` (`Optional[int]`): Save the checkpoint every
  `every_n_epochs` epochs. Default is `None`.
- `keep_last_n_checkpoints` (`Optional[int]`): Keep the last
  `keep_last_n_checkpoints` checkpoints. If we have more than
  `keep_last_n_checkpoints` checkpoints, we will remove the oldest ones.
  Default is `None`, which means all checkpoints are kept.
- `resume_from` (`Optional[ResumeOptions]`): The checkpoint to resume
  from. It can be a string (shorthand for `ResumeOptions(checkpoint=...)`),
  or a `ResumeOptions` object. Default is `None`.
  - `checkpoint` (`str`): The path to the checkpoint to resume from. It
    can be `last`/`best`/a specific folder/file. We will not resume (nor
    report error) if `checkpoint` is `last` or `best` but the
    corresponding checkpoint does not exist. Default is `'last'`.
  - `convert_fn` (`Optional[str]`): The full qualified name of the
    function to convert the checkpoint to nnscaler format. It should be
    `Callable[[Dict[str, Any]], Dict[str, Any]]`. Only applied when
    `checkpoint` is a file.
  - `with_merged` (`Optional[bool]`): Whether to merge the checkpoint
    files when loading. `True` means will load the merged checkpoint
    (without saving). `False` means will load the sharded checkpoint
    files. `None` means will load sharded if world size is unchanged,
    and merged otherwise. Only used when `checkpoint` is a directory.
  - `save_memory` (`bool`): If the memory is limited, only load merged
    state dict in GPU 0 of each node and broadcast trimmed state dict to
    other ranks. Although slower, this saves memory. Only used when
    resuming from a merged checkpoint. Default is `True`.

Please note

1.  When the parallel plan is changed (i.e you re-trace the model with
    different configurations), the checkpoints become incompatible, and
    can't be loaded any more. You must firstly merge the checkpoints to
    a merged checkpoint with `Trainer.merge_checkpoint` and then load
    the merged checkpoint just like a regular checkpoint.

    ``` python
    def merge_checkpoint(cls, checkpoint_files: List[str], output_file: str):
    ```

    where `checkpoint_files` is a list of checkpoint files to merge, and
    `output_file` is the output file path.

2.  When a checkpoint is saved, we will run validation on the validation
    dataset and save the validation loss to the checkpoint file. The
    validation run will ignore the `val_every_n_train_steps` and
    `val_every_n_epochs` configurations. If no valid dataset is
    provided, validation is skipped and `valid_loss` is set to
    `train_loss` by default.

3.  The sharded checkpoints will contain PyTorch's RNG state, but not
    Python's or NumPy's. The checkpoint's RNG state will be resumed
    right before training start, which means the initialization stage
    will use `TrainerArgs.seed` instead. Merged checkpoints
    will discard the RNG state.

### Other configs
- `init_module` (`Optional[str]`): A module to import before deserializing
  the config. This is useful when you need to register custom types or patch functions/types before
  the config is parsed.
- `vars` (`Dict[str, Any]`): User-defined variables that can be referenced
  elsewhere in the config using `$!(vars.xxx)` or `$!{vars.xxx}` syntax.
  See the "Variable Resolution" section below for more details.
- `gen_savedir` (`str`): The directory to save the generated files.
  Default is `./.nnscaler`.
- `gen_reuse` (`str`): the reuse strategy of the generated code, it can
  be
  - `auto`: automatically decide the reuse strategy (`moo` for
    `compile`, `match` for `run`)
  - one of `match`/`override`/`moo`/`graph`. See `parallelize` API for
    more information.
- `pas_policy` (`str`): The policy of parameter partitioning. Default is
  `autodist`. You can pass builtin pas policy name or your own pas
  policy function. See `parallelize` API for more information.
- `broadcast_strategy` (`str`): The strategy of broadcasting the model.
  Default is `all`. See `parallelize` API for more information.
- `instance_name` (`Optional[str]`): The instance name of the trainer.
  Default is `None`. See `parallelize` API for more information. You can
  dynamically set the instance name, for example, to the hash of related
  files.
- `run_mode` (`str`): The run mode of the trainer. It can be `run`
  (compile and train the model in a single python script OR train from
  previous compiling results) and `compile` (only compile the model for
  code generation). Default is `run`. Please note you can only use `run`
  mode with `torchrun`. On the other hand, if you disable broadcasting
  generated files (by setting `broadcast_strategy` to `none`), you can
  run `compile` mode without `torchrun`.
- `dummy_sample_gen_fn` (`Optional[str]`): The full qualified name of
  the function to generate a dummy sample. Its type should be
  `Callable[[TrainerArgs], Any]`. The tensors in the sample will be
  moved to GPU and converted to `input_dtype` by the trainer. If not
  specified, the first batch from the training dataloader will be used.
- `dummy_sample_post_process_fn` (`Optional[str]`): The full qualified
  name of the function to post process the dummy sample. The tensors in
  the sample have already been moved to GPU and converted to
  `input_dtype`. You can use this to further process the sample, for
  example, marking some dims of tensors as dynamic.
- `tracing_from_weights` (`str`): The path to the weights to be loaded
  when tracing(compiling) the model. It is only used in tracing to serve
  as the initial state dict of the model. Default is `None`.
- `precision`(`Union[str, Dict[_TENSOR_TYPE, _PRECISION_TYPE], None]`):
  The precision of the model. It can be a `str`, which means the same
  precision for all tensors, or a `Dict[_TENSOR_TYPE, _PRECISION_TYPE]`,
  which means the precision for each tensor type. Default is `None`.
  Currently we support 3 tensor types (`param`, `buffer`, `input`) and
  three precisions (`fp32`, `fp16`, `bf16`). You can set precision to
  `none` to avoid any precision conversion.
- `micro_batch_size` (`int`): The micro batch size. Default is `1`.
- `global_batch_size` (`Optional[int]`) and `grad_accumulation_steps`
  (`Optional[int]`): You can set one of `global_batch_size` and
  `grad_accumulation_steps` option. Please note if both are set, they
  must be consistent. Default is `micro_batch_size*scaling_factor` and
  `1` respectively.
- `max_epochs` (`Optional[int]`): The maximum number of epochs to train.
  Default is `None`, which means no limit.
- `max_train_steps` (`Optional[int]`): The maximum number of training
  steps to train. Default is `None`, which means no limit.
- `max_val_steps` (`Optional[int]`): The maximum number of validation
  steps to validate. Default is `None`, which means no limit.
- `val_every_n_train_steps` (`Optional[int]`): Validate every
  `val_every_n_train_steps` training steps. Default is `None`, which
  means no validation based on training steps.
- `val_every_n_epochs` (`Optional[int]`): Validate every
  `val_every_n_epochs` epochs. Default is `1`.
- `enable_progress_bar` (`bool`): Whether to enable the progress bar.
  Default is `True`.
- `log_progress_every_n_train_steps` (`Optional[int]`): When the
  progress bar is disabled (`enable_progress_bar` is `False`), the
  frequency to print the training progress. Validation metrics will also
  be printed if set. Default is `100`.
- `seed` (`Optional[int]`): The random seed. Default is `None`.
- `init_env_fn` (`str`): The function to initialize the environment.
  Default is `None`.

### Debug Config

``` python
@dataclass
class DebugConfig:
    check_gradient_sync_cross_devices: bool = True
```

- `check_gradient_sync_cross_devices` (`bool`): Before gradient clip
  norm, check the gradient sync for the same parameter is consistent
  across devices. If ZeRO is enabled, will check the gradient across
  each ZeRO group; if ZeRO is not enabled, will check the gradient
  across each nnscaler scale unit. This helps to find bugs related to
  gradient updates during training. Default is `True`.

## CLI

You can run the trainer with the following command:

``` bash
torchrun [torchrun arguments] ${NNSCALER_HOME}/cli/train.py -f ${CONFIG_FILE} [other arguments]
```

CONFIG_FILE is the path to the configuration yaml file. It looks like
(taken from our test case)

``` yaml
compute_config:
  plan_ngpus: 4
  runtime_ngpus: 100
  constant_folding: true
  use_zero: true
  use_end2end: true

run_mode: run
pas_policy: autodist
micro_batch_size: 2
global_batch_size: 8
max_epochs: 4
max_train_steps: 10

model:
  type: tests.cli.common.MLP
  args:
    dim: 16
    nlayers: 16

optimizer:
  type: torch.optim.Adam
  args:
    lr: 0.01

dataset:
  type: tests.cli.common.SimpleDataset
  train_args:
    dim: 16
    size: 100
  val_args:
    dim: 16
    size: 10

checkpoint:
  keep_last_n_checkpoints: 30
  every_n_train_steps: 1
  save_type: deduped
```

All the arguments in the yaml file are the same as the arguments in the
`TrainerArgs` class. And they can be override with the command line
arguments. For example, you can override the `max_epochs` with
`--max_epochs 2`, or override the `model` with
`--model.args.dim 32 --model.args.nlayers 32`.

### Variable Resolution

The configuration file supports variable resolution using `$(...)` or
`${...}` syntax. This allows you to reference other values within the
configuration file. For example:

``` yaml
compute_config:
  plan_ngpus: 4
  runtime_ngpus: $(compute_config.plan_ngpus)

model:
  args:
    dim: 16
    nlayers: $(model.args.dim)
```

In this example, `runtime_ngpus` will be resolved to the value of
`plan_ngpus` (4), and `nlayers` will be resolved to the value of `dim`
(16).

If a variable is not found in the configuration, the environment
variable with the same name will be used as a fallback. Circular
references will raise an error.

You can escape the variable syntax with a backslash: `$\(` or `$\{`.

### Resolved References

In component arguments (model args, optimizer args, dataloader args,
etc.), you can use `$!(...)` or `$!{...}` syntax to reference resolved
values. This is different from variable resolution: resolved references
are evaluated at runtime, not during config parsing.

- `$!(self)` or `$!{self}`: References the `TrainerArgs` object itself.
- `$!(vars.xxx)` or `$!{vars.xxx}`: References values from the `vars`
  dictionary defined in `TrainerArgs`.

For example:

``` yaml
vars:
  my_config:
    hidden_dim: 256

model:
  type: my_module.MyModel
  args:
    config: $!(vars.my_config)
    trainer_args: $!(self)
```

### Checkpoint Distribution

You can convert a merged checkpoint or directory containing per-rank
checkpoints into sharded checkpoints suitable for distributed training
with a different number of GPUs:

``` bash
python -m nnscaler.cli.checkpoint distribute <from> <to> -f <trainer_args.yml> [other trainer args]
```

Where `<from>` is the path to the merged checkpoint file or directory
containing per-rank checkpoints, and `<to>` is the directory to save
the sharded checkpoints.

## Appendix: ComputeConfig

### ComputeConfig

The configuration of the compute environment. It is a dataclass with the
following fields:

``` python
@dataclass(frozen=True)
class ComputeConfig:
    plan_ngpus: int
    runtime_ngpus: Optional[int] = None

    constant_folding: bool = False
    trace_strategy: Literal['cpu', 'cuda', 'meta', 'cuda_run_cpu_offload', 'reuse_cache'] = 'cuda_run_cpu_offload'

    use_zero: int = 0
    zero_ngroups: int = 1
    zero_use_reduce_scatter: bool = False
    zero_param_level_sharding: bool = False

    inference_only: bool = False
    use_end2end: bool = False

    use_async_reducer: bool = False
    reducer_bucket_cap_mb: Optional[float] = None

    pas_config: Dict[str, Any] = field(default_factory=dict)
    user_config: Dict[str, Any] = field(default_factory=dict)
```

We can categorize the fields into 4 categories:

1.  Trace configuration
    - `constant_folding`: whether to enable constant folding when
      generating code. When it is true, all non-tensor non-input values
      will be folded into the generated code.

      For example, if user's code contains following snippet, and
      `bsz=1`, `num_heads=32`, `len=1024`, `hidden_dim=128` at tracing.

      ``` python
      bsz, num_heads, len, hidden_dim = x.size()
      x = x.view(bsz * num_heads, len, hidden_dim)
      ```

      The code (graph) is folded into the following format

      ``` python
      y = x.view(32, 1024, 128)
      ```

      Constant folding is helpful to simplify the input program, and can
      make the compiling process faster and reduce the communication
      cost at runtime. However, user should make sure that inputs at
      runtime share a same schema (including shape) with tracing and
      correspond to a same computation graph. Errors may be raised at
      runtime when this assumption is broken.

    - `trace_strategy`: how to execute the functions during trace. Five
      strategies are supported:

      1.  `cpu`: Execute all functions on cpu device, model weights and
          intermediate results are on cpu device.
      2.  `cuda`: Execute all functions on cuda device, model weights
          and intermediate results are on cuda device. This strategy is
          recommended if the model can inference on single gpu.
      3.  `meta`: Execute all functions on meta device, model weights
          are on cpu and intermediate results are on meta device. For
          more information about meta device type, please view
          <https://pytorch.org/docs/stable/meta.html>.
      4.  `cuda_run_cpu_offload`: Try to execute all functions on cuda,
          and retry to execute the function on cpu as backup if OOM is
          caught, model weights and intermediate results are on cpu.
          This strategy is recommended for most cases if the model is too
          large to inference on single gpu.
      5.  `reuse_cache`: Compared to `cuda_run_cpu_offload` strategy,
          maintains a map from function signatures to output values. The
          cached output is returned when the signature of the function
          that generates it has been executed. Same signature means the
          functions are the same and have almost the same inputs (for
          tensor type input, just check if they have same tensor meta
          data[shape, dtype, requires_grad, stride, memory_format,
          ...], and don't check the value). This strategy is an
          experimental strategy to speedup the large-model-large-input
          case, and has risk of tracing an incorrect graph if the
          signature defined here cannot distinguish different
          functions used in the model, for example, torch.nonzero will
          always return the same result if the input has same meta data
          but different value.
2.  Compute environment configuration
    - `plan_ngpus`: the number of gpus to be used as a unit. The model
      is partitioned (TP or PP) within a unit, and then data parallelism
      is applied across multiple units. So every `plan_ngpus` devices
      holds the whole model. Furthermore, assume we have two workers,
      and their ranks are `rank1` and `rank2`:
      1.  if `rank1 // plan_gpus == rank2 // plan_ngpus`, then they are
          in the same unit.
      2.  If `rank1 % plan_ngpus == rank2 % plan_ngpus`, then the
          portion of model held on both gpus are exactly the same.
    - `runtime_ngpus`: the number of gpus to be used in runtime. It
      should be a multiple of `plan_ngpus`, which means we have
      `runtime_ngpus // plan_ngpus` units in runtime, and the data
      parallelism is `runtime_ngpus // plan_ngpus`. Please note all
      modules must have the same `plan_ngpus` and `runtime_ngpus`.
      If not set, it defaults to the value of the `WORLD_SIZE` environment variable.
3.  Code generation feature configuration
    - `use_zero`: the ZeRO optimization level. Supported values:
      - `0`: no ZeRO optimization.
      - `1`: ZeRO stage 1 (optimizer state sharding).
      - `3`: ZeRO stage 3 (optimizer state + gradient + parameter sharding).
      (Passing `2` is also accepted and is treated as ZeRO stage 3.)
    - `zero_ngroups`: the number of ZeRO groups. Default is `1`.
    - `zero_use_reduce_scatter`: whether to use reduce-scatter for ZeRO. Only effective when `use_zero != 0` and `zero_ngroups == 1`. May introduce parity issues in some cases; use with caution. Default is `False`.
    - `zero_param_level_sharding`: whether to shard parameters at parameter level (instead of element level) in ZeRO. Only effective when `use_zero != 0`. Must be set to `True` when using the Muon optimizer. May introduce padding. Default is `False`.
    - `inference_only`: whether to generate code for inference only. If
      it is true, the generated code cannot be used to train the model.
    - `use_end2end`: whether to use end2end training. For the
      requirement of end2end, see the description above.
    - `use_async_reducer`: whether to use async all-reduce for gradients. Only effective when `use_end2end` is `True`. Default is `False`.
    - `reducer_bucket_cap_mb`: the maximum reducer bucket size in megabytes for one all-reduce call. Also effective for sync reducers. `None`/`0` uses the default (25 MB for async, no limit for sync). Default is `None`.
    - `pas_config`: the configuration for the PAS policy
      (partition-assign-schedule policy, which describes how to place
      all computations across devices. For details, please refer to
      `PAS Policies`. It is
      a dictionary, and will be used by the PAS policy. Please note
      different PAS will have different configurations. You can also put
      any other settings that can affect code generation here, but
      please prefix the keys with `_` to avoid conflicts with PAS
      configurations.
    - `user_config`: the user configuration, which is used to decide
      whether to skip compiling and reuse the previously traced graph.

Note:

1.  You can put any custom configurations in `user_config`. The
    assumption is different `user_config` should generate different
    graph/code. So if the user config is changed, we will regenerate the
    graph/code automatically. Here are some examples:

    - Example 1: save module configuration

      ``` python
      class MyModule(torch.nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x):
              ...
              if module_config.use_3d:
                  ...
      ```

      here we can set `user_config` to
      `{'use_3d': module_config.use_3d}`, and we can be sure different
      use_3d config will never use the same graph (and eventually the
      generated code).

    - Example 2: save file stats

      If you want to track all related file stats (just like traditional
      compilers do), you can save the md5 of the files to save some
      bytes:

      ``` python
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

2.  If some settings don't affect tracing/graph generation, but do
    affect code generation, you can put them in `pas_config`. Please
    prefix the keys with `_` to avoid conflicts with predefined PAS
    configurations. One typical example is you can put the name of
    selected PAS policy in `pas_config`, so changing PAS policy will
    regenerate code but the graph will be reused.

    ``` python
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

The reuse policy for the existing generated code. It is an enum with the
following values:

``` python
class ReuseType(Enum):
    MATCH = 'match'
    OVERRIDE = 'override'
    MOO = 'moo'
    GRAPH = 'graph'
```

We call it a `match` when the `ComputeConfig` is the same with the
previous run.

1.  `MATCH`: Reuse if match, error if not match, generate if no previous
    gerenated code exists.
2.  `OVERRIDE`: Nothing will be reused. Everything will be regenerated.
3.  `MOO`: `MOO` is short for 'match or override'. It will reuse if
    match, generate if not match or no previous generated code exists.
4.  `GRAPH`: Reuse graph only if match, generate otherwise.

### PAS Policies

Writing a pas policy can be very hard and error-prone. So we provide 6
builtin PAS policies to help you. `dp`, `tp`, `pp`, `data`, `hybrid`,
and `autodist`. Please note only `autodist` policy is the recommended
policy for most cases, and all other PAS policies are mainly test
purpose only.

The configuration of the PAS policy should be passed in the `pas_config`
of `ComputeConfig` as a dictionary.

1.  `dp`: data parallelism. It will replicate the module across all
    devices, and run data parallelism across all devices. It requires
    the `plan_ngpus` must be 1 and no configurations
2.  `tp`: tensor parallelism + data parallelism. It will do tensor
    parallelism inside a scale unit, and run data parallelism across
    scale units. It has only one configuration:
    - seed: the random seed for choose the partition dimension. Default
      is `1`
3.  `pp`: pipeline parallelism + data parallelism. It will do model
    parallelism inside a scale unit, and run data parallelism across
    scale units. It requires the `use_end2end` be true. It has two
    configurations `pipeline_nmicros` and `pipeline_scheduler`. See
    `hybrid` policy for more details.
4.  `data`: tensor parallelism on batch dimension. It has no
    configurations.
5.  `hybrid`: pipeline parallelism + tensor parallelism + data
    parallelism. It will do model parallelism and tensor parallelism(on
    0 dimension) inside a scale unit, and run data parallelism across
    scale units. It requires the `use_end2end` to be true. It has the
    following configurations.
    - `pipeline_nstages`: the number of stages in the pipeline, or
      `"auto"` (let autodist to decide). Default is `"auto"`. Optional.
      - If `pipeline_nstages` is `"auto"` and `pipeline_pivots` is
        specified, it will use pipeline. (The number of stages will be
        determined automatically by autodist)
      - If `pipeline_nstages` is `"auto"` and `pipeline_pivots` is not
        specified, it will not use pipeline.
      - If `pipeline_nstages` is a 1, pipeline will not be used.
        (`pipeline_pivots` must not be set)
      - If `pipeline_nstages` is a number > 1, pipeline will be used.
        (`pipeline_pivots` must be set)
    - `pipeline_nmicros`: the number of microbatches in the pipeline.
      Required.
    - `pipeline_scheduler`: the scheduler name for the pipeline. Current
      we support four schedulers in training
      `1f1b`/`1f1b_plus`/`1f1b_interleaved`/`gpipe`/`chimera_direct` (4
      stages pipeline only), and one scheduler in inference
      `infer_pipe`. Default is `1f1b`. Optional.
    - `pp_size`: the pipeline parallelism size. Default is
      `pipeline_nstages`. Optional.
6.  `autodist`: the recommended policy for most cases. Currently it only
    support Adam-like optimizers. It will automatically choose the best
    partition for you by balancing the memory usage and speed. It has
    the following configurations.
    - `update_freq (int)`: the update frequency when training the
      module. Default is 1. Optional.
    - `mem_constraint (float)`: The memory constraint in each device in
      GB. Optional.
    - `task_name (str)`: The name of the current task to distinguish
      runs. Optional.
    - `use_fp16 (bool)`: Whether you use `fp16`. Default is `False`.
      Optional.
    - `use_memory_efficient_fp16` Whether you use memory efficient fp16
      optimizer. Default is `False`. Optional.
    - `use_bf16`: Whether you use `bf16`. Default is `False`. Optional.
    - `use_memory_efficient_bf16`: Whether you use memory efficient bf16
      optimizer. Default is `False`. Optional.
    - `re_profile (bool)`: If set to `True`, the computation profiling
      results will be overridden. Please note reprofiling will take some
      time. Optional.
    - `verbose (bool)`: Whether to print verbose information. Optional.
    - `load_plan_path (str)`: The path to the plan file to load. If
      specified, the plan will be loaded from the file instead of
      searching. Optional.
    - `save_plan_path (str)`: The path to the plan file to save.
      Optional.
    - `partition_constraints_path (str)`: The path to the partition
      constraints file. Optional.
    - `recompute_modules (str)`: The module names to recompute,
      separated by `,`. For example, `module1,module2`. Optional.
    - `pipeline_pivots (str)`: If set, autodist will try pipeline
      parallelism to find the best partition plan. It specifies the
      module names to pivot the pipeline, separated by `,`. For example,
      if `module1,module2` is specified, stages searched by pipeline
      solver only start from either `module1` or `module2`. Optional.
    - `use_apex_fused_adam_v2`: If set to `True`, the apex fused adam v2
      optimizer will be used. Default is `False`. Optional.
    - `pipeline_scheduler`: The scheduler name for the pipeline. Please
      note currently `1f1b` is the only supported scheduler in
      `autodist`. Default is `1f1b`. Optional.
    - `parallel_profile`: If set to `True`, autodist will profile
      operators in parallel by using available gpus. Default is `True`.
      Optional.
    - `max_partition_degree`: Max degree when partitioning an operator /
      node. When pipeline parallelism is enabled to explore
      (`explore_pipeline` is True), user can change the value to
      constrain the plan to be composed of stages that span on less or
      equal to `max_partition_degree` devices (recommend to set
      `max_partition_degree` to the number of devices in a node to avoid
      inter-node communication, but should be be no more than
      `plan_ngpus`). Default is `plan_ngpus`. Optional.
    - `transient_mem_coef`: In autodist, a heuristic is used to estimate
      the transient memory size:
      `transient_mem_size = opt_transient_coef * (1st_largest_infer_mem + 2nd_largest_infer_mem)`.
      This formula is useful in many cases, but it may be too strict
      when some operators consume or generate a large tensor (>= 4GB).
      In this case, you can set `transient_mem_coef` to a smaller value
      to relax the constraint. Default is `2`. Optional.

You can also put any other settings that can affect code generation
here. but please prefix the keys with `_` to avoid conflicts with
predefined keys.

Here is an example:

``` python
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
