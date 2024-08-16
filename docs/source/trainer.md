# Trainer

We provide a `Trainer` class that can be used to train and evaluate models. It will firstly parallelize the model on multiple GPUs with `parallelize` API, and then train the model with the given dataset and optimizer in a distributed way.


## Arguments

All the arguments are defined in `TrainerArgs` class. Here is the definition of `TrainerArgs`:

```python
@dataclass
class TrainerArgs:
    compute_config: ComputeConfig = None

    gen_savedir: str = './.nnscaler'
    gen_reuse: str = 'auto'
    pas_policy: str = 'autodist'
    broadcast_strategy: str = 'all'
    instance_name: str = None
    run_mode: str = 'run'
    tracing_from_weights: str = None

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    dataset_sampler: DatasetSamplerConfig = field(default_factory=DatasetSamplerConfig)
    lr_scheduler: Optional[LRSchedulerConfig] = None
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    log: List[LogConfig] = field(default_factory=list)
    hook: Union[HookConfig, HookMapConfig, None] = None

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

    seed: Optional[int] = None
    init_env_fn: str = None
```

The design philosophy of `Trainer` arguments is:
The classes(or factory functions) of components(model/optimizer/etc)
and their arguments are provided in the `TrainerArgs` class (functions/types are passed as fully qualified names),
and we are responsible for creating them.

For example, you can tell me how to create a model by providing the model type and its arguments in `ModelConfig` class.

Please note some of the arguments of components are set automatically, and you should not set them manually.
For example, arguments `dataset`, `num_replicas` and `rank` of the dataset sampler are set automatically by the `Trainer` class.
Those 3 arguments passed in the `DatasetSamplerConfig.train_args/val_args`(if any) will be ignored.

```python
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

If any argument type is a class, you can pass it as a dict, and add a special key `__type` to specify the class type.

For example, if the module `__init__` takes `ModelConfig` object
```python
class SomeModule(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        ...
```
You can pass the `model_config` as
```python
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

We also use `ast.literal_eval` to guess the type of the string arguments, You can skip it by passing a dict with `__value_type` and `value` keys. For example, you want a number to be a str, you can use
```python
{
    '__value_type': 'str',
    'value': '1'
}
```
Internally we will get the final value with `__value_type(value)`.

### Component Configs

- `model` (`ModelConfig`): The model to be trained. You need to provide the model type and its arguments in `ModelConfig` class. Here is the definition of `ModelConfig`:

    ```python
    @dataclass
    class ModelConfig:
        type: str = None
        args: Dict[str, Any] = field(default_factory=dict)
    ```
- `optimizer` (`OptimizerConfig`): The optimizer to be used.

    ```python
    @dataclass
    class OptimizerConfig:
        type: str = None
        args: Dict[str, Any] = field(default_factory=dict)
        clip_gnorm: float = 0.0

        loss_reduction: str = 'mean'
        grad_reduction: str = 'mean'
        aggregate_outputs_fn: str = None
    ```
    - `type` (`str`): The optimizer type or factory function.
    Please note the first parameter of the optimizer constructor must be the model parameters.
    - `args` (`Dict[str, Any]`): The arguments of the optimizer.
    - `clip_gnorm` (`float`): The maximum norm value for gradient clipping. 0.0/None means no clipping.
    - `loss_reduction` (`str`): The reduction method for loss.
    It can be `mean` (average the loss over all micro-batches),
    `sum` (sum the loss of all micro-batches).
    Default is `mean`.
    Please note in validation stage, this configuration is ignored the loss is always averaged over all batches
    - `grad_reduction` (`str`): The reduction method for gradients. It can be `mean` (average the gradients over all micro-batches), `sum` (sum the gradients of all micro-batches), `per-token-mean` (average the gradients over all tokens). Default is `mean`. Please note if `per-token-mean` is used, you need to specify `aggregate_outputs_fn`, which will return the number of tokens
    - `aggregate_outputs_fn` (`str`): The function to aggregate the outputs of the model. It is required when `grad_reduction` is `per-token-mean`. Its signature should be `def aggregate_outputs(self, loss_outputs, sync_group) -> AggregatedOutputs`, where `loss_outputs` is a list of outputs of the model, and `sync_group` is the `torch.distributed.ProcessGroup` to sync with. The function should return an `AggregatedOutputs` object, which defines as:
        ```python
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
        ```
- `dataset` (`DatasetConfig`): The dataset to be used.
    ```python
    @dataclass
    class DatasetConfig:
        type: str = None
        train_args: Dict[str, Any] = field(default_factory=dict)
        val_args: Dict[str, Any] = field(default_factory=dict)
    ```
    - `type` (`str`): The dataset type or factory function.
    - `train_args` (`Dict[str, Any]`): The arguments of the training dataset.
    - `val_args` (`Dict[str, Any]`): The arguments of the validation dataset.
- `dataloader` (`DataloaderConfig`): The dataloader to be used. Please note we recommend to pass `drop_last=True` in the dataloader arguments to avoid the last batch with different sizes.

    ```python
    @dataclass
    class DataloaderConfig:
        type: str = 'torch.utils.data.DataLoader'
        train_args: Dict[str, Any] = field(default_factory=dict)
        # default to train_args
        val_args: Dict[str, Any] = field(default_factory=dict)
        # default to train_args
        test_args: Dict[str, Any] = field(default_factory=dict)
    ```
    - `type` (`str`): The dataloader type or factory function.
    Please note the dataloader constructor must at least have 3 parameters `dataset`, `batch_size`, `sampler`.
    - `train_args` (`Dict[str, Any]`): The arguments (except `dataset`,`batch_size`, `sampler`) of the training dataloader. Argument `batch_size` will be set to `micro_batch_size`.
    - `val_args` (`Dict[str, Any]`): The arguments (except `dataset`,`batch_size`, `sampler`) of the validation dataloader.

- `dataset_sampler` (`DatasetSamplerConfig`): The dataset sampler to be used.

    ```python
    @dataclass
    class DatasetSamplerConfig:
        type: str = 'torch.utils.data.DistributedSampler'
        train_args: Dict[str, Any] = field(default_factory=dict)
        val_args: Dict[str, Any] = field(default_factory=dict)
        test_args: Dict[str, Any] = field(default_factory=dict)
    ```
    - `type` (`str`): The dataset sampler type or factory function.
    Please note the dataset sampler constructor must at least have 3 parameters `dataset`, `num_replicas`, `rank`.
    - `train_args` (`Dict[str, Any]`): The arguments (except `dataset`,`num_replicas`, `rank`) of the training dataset sampler.
    - `val_args` (`Dict[str, Any]`): The arguments (except `dataset`,`num_replicas`, `rank`) of the validation dataset sampler.

- `lr_scheduler` (`LRSchedulerConfig`): The learning rate scheduler to be used. This is optional.

    ```python
    @dataclass
    class LRSchedulerConfig:
        type: str = None
        args: Dict[str, Any] = field(default_factory=dict)
        interval: str = 'epoch'
    ```
    - `type` (`str`): The learning rate scheduler type or factory function.
    Please note the first parameter of the learning rate scheduler constructor must be optimizer.
    - `args` (`Dict[str, Any]`): The arguments of the learning rate scheduler.
    - `interval` (`str`): The interval to update the learning rate. It can be `epoch` or `step`. Default is `epoch`.

- `log` (`List[LogConfig]`): The loggers to be used. You can provide multiple loggers. Currently we have two builtin loggers: `TensorBoardLogger` and `WandbLogger`.

    ```python
    @dataclass
    class LogConfig:
        type: str = None
        args: Dict[str, Any] = field(default_factory=dict)
    ```
    - `type` (`str`): The logger type or factory function.
    - `args` (`Dict[str, Any]`): The arguments of the logger.

- `hook` (`Union[HookConfig, HookMapConfig, None]`): The hooks to be used.
You can provide a hook with a hook class or a map of hook functions.
Please note if your `model`/`optimizer`/`lr_scheduler` inherit from `TrainHook`,
their hook functions will be called automatically.
The order of the hook functions called is `model` -> `optimizer` -> `lr_scheduler`,
and hooks passed with this config is called in the last.

    Hook class:

        ```python
        @dataclass
        class HookConfig:
            type: str = None
            args: Dict[str, Any] = field(default_factory=dict)
        ```

    - `type` (`str`): The hook type or factory function.
    - `args` (`Dict[str, Any]`): The arguments of the hook.

    Hook map:

        ```python
        @dataclass
        class HookMapConfig:
            after_setup: str = None

            on_train_start: str = None
            on_train_end: str = None
            on_val_start: str = None
            on_val_end: str = None

            on_epoch_start: str = None
            on_epoch_end: str = None

            on_train_step_start: str = None
            on_train_step_end: str = None
            on_val_step_start: str = None
            on_val_step_end: str = None

            after_aggregate_train_step_outputs: str = None
            after_aggregate_val_step_outputs: str = None

            before_zero_grad: str = None
            after_zero_grad: str = None

            before_gnorm_clip: str = None
            after_gnorm_clip: str = None

            before_optimizer_step: str = None
            after_optimizer_step: str = None

            on_load_checkpoint: str = None
            on_save_checkpoint: str = None
        ```
    - `after_setup` (`str`): The hook function to be called after setting up the trainer.
    Only be called when `run_mode == 'run'`.
    Signature:  `def after_setup(trainer: 'Trainer') -> None:`
    - `on_train_start` (`str`): The hook function to be called at the start of the training stage. Signature:  `def on_train_start(trainer: 'Trainer') -> None:`
    - `on_train_end` (`str`): The hook function to be called at the end of the training stage. Signature:  `def on_train_end(trainer: 'Trainer') -> None:`
    - `on_val_start` (`str`): The hook function to be called at the start of the validation stage. Signature:  `def on_val_start(trainer: 'Trainer') -> None:`
    - `on_val_end` (`str`): The hook function to be called at the end of the validation stage. Signature:  `def on_val_end(trainer: 'Trainer', val_loss: float) -> None:`
    - `on_epoch_start` (`str`): The hook function to be called at the start of each epoch. Signature:  `def on_epoch_start(trainer: 'Trainer', epoch: int) -> None:`
    - `on_epoch_end` (`str`): The hook function to be called at the end of each epoch. Signature:  `def on_epoch_end(trainer: 'Trainer', epoch: int) -> None:`
    - `on_train_step_start` (`str`): The hook function to be called at the start of each training step. Signature:  `def on_train_step_start(trainer: 'Trainer', batches: List[Any], idx: int) -> None:`
    - `on_train_step_end` (`str`): The hook function to be called at the end of each training step. Signature:  `def on_train_step_end(trainer: 'Trainer', outputs: List[Any], batches: List[Any], idx: int) -> None:`
    - `on_val_step_start` (`str`): The hook function to be called at the start of each validation step. Signature:  `def on_val_step_start(trainer: 'Trainer', batches: List[Any], idx: int) -> None:`
    - `on_val_step_end` (`str`): The hook function to be called at the end of each validation step. Signature:  `def on_val_step_end(trainer: 'Trainer', outputs: List[Any], batches: List[Any], idx: int) -> None:`
    - `after_aggregate_train_step_outputs` (`str`): The hook function to be called after aggregating the outputs of the model in the training step. Signature:  `def after_aggregate_train_step_outputs(trainer: 'Trainer', aggregated_outputs: 'AggregatedOutputs', train_loss: float, idx: int) -> None:`
    - `after_aggregate_val_step_outputs` (`str`): The hook function to be called after aggregating the outputs of the model in the validation step. Signature:  `def after_aggregate_val_step_outputs(trainer: 'Trainer', aggregated_outputs: 'AggregatedOutputs', val_loss: float, idx: int) -> None:`
    - `before_zero_grad` (`str`): The hook function to be called before zeroing the gradients. Signature:  `def before_zero_grad(trainer: 'Trainer') -> None:`
    - `after_zero_grad` (`str`): The hook function to be called after zeroing the gradients. Signature:  `def after_zero_grad(trainer: 'Trainer') -> None:`
    - `before_sync_grad` (`str`): The hook function to be called before syncing the gradients between ranks.
    Please note this hook can't be triggered correctly,
    and you should not reply on this.
    Will fix it later.
    Signature:  `def before_sync_grad(trainer: 'Trainer') -> None:`
    - `after_sync_grad` (`str`): The hook function to be called after syncing the gradients between ranks. Signature:  `def after_sync_grad(trainer: 'Trainer') -> None:`
    - `before_gnorm_clip` (`str`): The hook function to be called before gradient clipping. Signature:  `def before_gnorm_clip(trainer: 'Trainer') -> None:`
    - `after_gnorm_clip` (`str`): The hook function to be called after gradient clipping. Signature:  `def after_gnorm_clip(trainer: 'Trainer', gnorm: torch.Tensor) -> None:`
    - `before_optimizer_step` (`str`): The hook function to be called before the optimizer step. Signature:  `def before_optimizer_step(trainer: 'Trainer') -> None:`
    - `after_optimizer_step` (`str`): The hook function to be called after the optimizer step. Signature:  `def after_optimizer_step(trainer: 'Trainer') -> None:`
    - `on_load_checkpoint` (`str`): The hook function to be called after loading the checkpoint. If you saved something with `on_save_checkpoint` this is
    your chance to restore this. Signature:  `def on_load_checkpoint(trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:`
    - `on_save_checkpoint` (`str`): The hook function to be called before saving the checkpoint. If you want to save something, you can add it to the checkpoint here. Signature:  `def on_save_checkpoint(trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:`

### Compute Config

All compute configs are put in `compute_config` (`ComputeConfig`).  Please refer to [link](./parallel_module.md#ComputeConfig) for more information.

Please note only end2end mode is supported in the trainer, so you must set `compute_config.use_end2end` to `True` to make it work.

### Checkpoint Config

```python
@dataclass
class CheckpointConfig:
    save_dir: str = './checkpoints'
    no_save: bool = False

    save_type: str = 'sharded'

    save_last: bool = True
    save_best: bool = True
    symlink_best_and_last: bool = True

    every_n_train_steps: Optional[int] = None
    every_n_epochs: Optional[int] = None
    keep_last_n_checkpoints: Optional[int] = None

    resume_from: str = None
```

- `save_dir` (`str`): The directory to save the checkpoints.
- `no_save` (`bool`): Whether to save the checkpoints. Default is `False`.
- `save_type` (`str`): The type of saving checkpoint. It can be `sharded` or `deduped`. Default is `sharded`.
    -  `"sharded"`: Each rank saves its shard of weights and optimizer states to a file.
    The checkpoint is a folder with as many files as the world size.
    - `"deduped"`: Each rank saves its deduped shard of weights and optimizer states to a file.
    The checkpoint is a folder with as many files as the world size.
    - `"merged"`: everything has been merged into a single file. Used internally only when you merge the checkpoint files via `Trainer.merge_checkpoints`
- `save_last` (`bool`): Whether to save the last checkpoint. Default is `True`.
- `save_best` (`bool`): Whether to save the best (lowest `val_loss`) checkpoint. Default is `True`.
- `symlink_best_and_last` (`bool`): Whether to use symlink (instead of copy) to the best and last checkpoint. Default is `True`.
- `every_n_train_steps` (`Optional[int]`): Save the checkpoint every `every_n_train_steps` training steps. Default is `None`, which means no checkpoint is saved based on training steps.
- `every_n_epochs` (`Optional[int]`): Save the checkpoint every `every_n_epochs` epochs. Default is `None`, which means no checkpoint is saved based on epochs.
- `keep_last_n_checkpoints` (`Optional[int]`): Keep the last `keep_last_n_checkpoints` checkpoints. If we have more than `keep_last_n_checkpoints` checkpoints, we will remove the oldest ones.
Default is `None`, which means all checkpoints are kept.
- `resume_from` (`str`): The path to the checkpoint to resume from. It can be `last`/`best`/a specific folder/file.
We will not resume (nor report error) if resume_from is `last` or `best` but the corresponding checkpoint does not exist.
Default is `None`.

Please note

1. When the parallel plan is changed (i.e you re-trace the model with different configurations),
the checkpoints become incompatible, and can't be loaded any more.
You must firstly merge the checkpoints to a merged checkpoint with `Trainer.merge_checkpoint` and then load the merged checkpoint just like a regular checkpoint.

    ```python
    def merge_checkpoint(cls, checkpoint_files: List[str], output_file: str):
    ```
    where `checkpoint_files` is a list of checkpoint files to merge, and `output_file` is the output file path.

2. When a checkpoint is saved,
we will run validation on the validation dataset and save the validation loss to the checkpoint file.
The validation run will ignore the `val_every_n_train_steps` and `val_every_n_epochs` configurations.
If no valid dataset is provided, validation is skipped and `valid_loss` is set to `train_loss` by default.

### Other configs
- `gen_savedir` (`str`): The directory to save the generated files. Default is `./.nnscaler`.
- `gen_reuse` (`str`):  the reuse strategy of the generated code, it can be
    - `auto`: automatically decide the reuse strategy (`moo` for `compile`, `match` for `run`)
    - one of `match`/`override`/`moo`/`graph`. See `parallelize` API for more information.
- `pas_policy` (`str`): The policy of parameter partitioning. Default is `autodist`.
You can pass builtin pas policy name or your own pas policy function.
See `parallelize` API for more information.
- `broadcast_strategy` (`str`): The strategy of broadcasting the model. Default is `all`. See `parallelize` API for more information.
- `instance_name` (`str`): The instance name of the trainer. Default is `None`. See `parallelize` API for more information.
- `run_mode` (`str`): The run mode of the trainer.
It can be `run` (compile and train the model in a single python script OR train from previous compiling results) and `compile` (only compile the model for code generation). Default is `run`.
Please note you can only use `run` mode with `torchrun`.
On the other hand, if you disable broadcasting generated files (by setting `broadcast_strategy` to `none`),
you can run `compile` mode without `torchrun`.
- `tracing_from_weights` (`str`): The path to the weights to be loaded when tracing(compiling) the model. It is only used in tracing to serve as the initial state dict of the model. Default is `None`.
- `precison`(`Union[str, Dict[_TENSOR_TYPE, _PRECISION_TYPE], None]`): The precision of the model. It can be a `str`, which means the same precision for all tensors, or a `Dict[_TENSOR_TYPE, _PRECISION_TYPE]`, which means the precision for each tensor type. Default is `None`. Currently we support 3 tensor types (`param`, `buffer`, `input`) and three precisions (`fp32`, `fp16`, `bf16`). You can set precision to `none` to avoid any precision conversion.
- `micro_batch_size` (`int`): The micro batch size. Default is `1`.
- `global_batch_size` (`Optional[int]`) and `grad_accumulation_steps` (`Optional[int]`): You can set one of `global_batch_size` and `grad_accumulation_steps` option. Please note if both are set, they must be consistent. Default is `micro_batch_size*scaling_factor` and `1` respectively.
- `max_epochs` (`Optional[int]`): The maximum number of epochs to train. Default is `None`, which means no limit.
- `max_train_steps` (`Optional[int]`): The maximum number of training steps to train. Default is `None`, which means no limit.
- `max_val_steps` (`Optional[int]`): The maximum number of validation steps to validate. Default is `None`, which means no limit.
- `val_every_n_train_steps` (`Optional[int]`): Validate every `val_every_n_train_steps` training steps. Default is `None`, which means no validation based on training steps.
- `val_every_n_epochs` (`Optional[int]`): Validate every `val_every_n_epochs` epochs. Default is `1`.
- `enable_progress_bar` (`bool`): Whether to enable the progress bar. Default is `True`.
- `seed` (`Optional[int]`): The random seed. Default is `None`.
- `init_env_fn` (`str`): The function to initialize the environment. Default is `None`.

## CLI

You can run the trainer with the following command:

```bash
torchrun [torchrun arguments] ${NNSCALER_HOME}/cli/train.py -f ${CONFIG_FILE} [other arguments]
```

CONFIG_FILE is the path to the configuration yaml file. It looks like (taken from our test case)

```yaml
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

All the arguments in the yaml file are the same as the arguments in the `TrainerArgs` class.
And they can be override with the command line arguments.
For example, you can override the `max_epochs` with `--max_epochs 2`, or override the `model` with `--model.args.dim 32 --model.args.nlayers 32`.
