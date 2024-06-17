from dataclasses import asdict, dataclass, field
import importlib
from typing import Any, Dict, List, Optional

import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import yaml
import torch

from nnscaler.parallel import ComputeConfig, build_optimizer
from nnscaler.runtime.module import ParallelModule

from .arg_parser import deserialize_dataclass, merge_args, parse_args


def load_type(type_name: str):
    parts = type_name.rsplit('.', 1)
    if len(parts) == 1:
        nm = __builtins__
        type_name = parts[0]
    else:
        namespace, type_name = parts
        nm = importlib.import_module(namespace)
    return getattr(nm, type_name)


@dataclass
class AggregatedOutputs:
    """
    Aggregated outputs from all micro-batches
    """
    loss: Optional[int] = None
    num_samples: Optional[int] = None
    num_tokens: Optional[int] = None
    # any other custom outputs
    aggregated_outputs: Any = None


@dataclass
class TrainerArgs:
    compute_config: ComputeConfig = None

    gen_savedir: str = './.nnscaler'
    pas_policy: str = 'autodist'
    broadcast_strategy: str = 'all'
    instance_name: str = None
    # compile: compile the model but not training
    # run: compile and run the model
    run_mode: str = 'run'
    # the model state dict for tracing.
    ckpt_tracing: str = None

    model_class: str = None
    model_args: Dict[str, Any] = field(default_factory=dict)
    fp16: bool = False
    bf16: bool = False

    optimizer_class: str = None
    optimizer_args: Dict[str, Any] = field(default_factory=dict)

    dataset_class: str = None
    train_dataset_args: Dict[str, Any] = field(default_factory=dict)
    val_dataset_args: Dict[str, Any] = field(default_factory=dict)
    test_dataset_args: Dict[str, Any] = field(default_factory=dict)

    dataloader_class: str = 'torch.utils.data.DataLoader'
    train_dataloader_args: Dict[str, Any] = field(default_factory=dict)
    # default to train_dataloader_args
    val_dataloader_args: Dict[str, Any] = field(default_factory=dict)
    # default to train_dataloader_args
    test_dataloader_args: Dict[str, Any] = field(default_factory=dict)

    dataset_sampler_class: str = 'torch.utils.data.DistributedSampler'
    train_dataset_sampler_args: Dict[str, Any] = field(default_factory=dict)
    val_dataset_sampler_args: Dict[str, Any] = field(default_factory=dict)
    test_dataset_sampler_args: Dict[str, Any] = field(default_factory=dict)

    lr_scheduler_class: str = None
    lr_scheduler_args: Dict[str, Any] = field(default_factory=dict)

    micro_batch_size: int = 1
    global_batch_size: int = 1

    max_epochs: int = 1000
    clip_gnorm: float = 0.0
    # TODO: support different ways of calculating grad and loss
    # sum: sum the gradients of all micro-batches
    # per-sample-mean: average the gradients over all micro-batches
    # per-token-mean: average the gradients over all tokens
    #    you must specify `aggregate_outputs_fn` and return the number of tokens
    gradient_accumulation: str = 'sum'
    # the function to aggregate the outputs from all micro-batches
    # inputs: (list of local outputs, torch group)
    # output: AggregateOutputs
    # you can use `torch.distributed.*` functions to do the work
    aggregate_outputs_fn: str = None

    ckpt_save_dir: str = None
    # `"sharded"`: Each rank saves its shard of weights and optimizer states to a file. The checkpoint is
    #   a folder with as many files as the world size.
    # `"deduped"`: Each rank saves its deduped shard of weights and optimizer states to a file. The checkpoint is
    #   a folder with as many files as the world size.
    ckpt_save_type: str = 'sharded'
    ckpt_load_file: str = None

    def __post_init__(self):
        if not self.compute_config:
            raise ValueError("compute_config is required")
        if not self.compute_config.use_end2end:
            raise ValueError("use_end2end must be True")
        if self.global_batch_size % self.micro_batch_size != 0:
            raise ValueError(f"global_batch_size {self.global_batch_size} is not divisible by micro_batch_size {self.micro_batch_size}")
        if self.run_mode not in ('compile', 'run'):
            raise ValueError(f"Invalid run_mode {self.run_mode}")
        if self.ckpt_save_type not in ('sharded', 'deduped'):
            raise ValueError(f"Invalid ckpt_save_type {self.ckpt_save_type}")
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        if not self.model_class:
            raise ValueError("model_class is required")
        if not self.optimizer_class:
            raise ValueError("optimizer_class is required")
        if not self.dataset_class:
            raise ValueError("dataset_class is required")
        if not self.dataloader_class:
            raise ValueError("dataloader_class is required")
        if not self.dataset_sampler_class:
            raise ValueError("dataset_sampler_class is required")

    @classmethod
    def from_cli(cls, argv: List[str]) -> 'TrainerArgs':
        d = {}
        if argv[0] == '-f':
            with open(argv[1], 'r') as f:
                d = yaml.safe_load(f)
            argv = argv[2:]

        merge_args(d, parse_args(argv))
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainerArgs':
        ta = deserialize_dataclass(d, TrainerArgs)
        return ta

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainerArgs':
        with open(path, 'r') as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def create_kwarg(cls, value: dict):
        for k, v in value.items():
            if isinstance(v, dict):
                value[k] = cls.create_kwarg(v)
            elif isinstance(v, list):
                value[k] = [cls.create_kwarg(i) for i in v]
            elif isinstance(v, tuple):
                value[k] = tuple(cls.create_kwarg(i) for i in v)

        if '__type' in value:
            value_type = load_type(value.pop('__type'))
            return value_type(**value)
        elif '__value_type' in value:
            if 'value' not in value:
                raise ValueError("value is required when __value_type is present")
            value_type = value.pop('__value_type')
            if value_type == 'function':  # when type is function, the value should be the full qualified name of the function
                return load_type(value['value'])
            else:
                # call its __init__ function
                value_type = load_type(value_type)
                return value_type(value['value'])
        else:
            return value

    @property
    def model_type(self):
        return load_type(self.model_class)

    @property
    def collate_fn(self):
        """
        Used to generate dummy input from dataset
        """
        args = self.train_dataloader_args
        if 'collate_fn' in args:
            return load_type(args['collate_fn'])
        # hack to get default collate_fn
        return torch.utils.data.dataloader.default_collate

    @property
    def scaling_factor(self):
        return self.compute_config.runtime_ngpus // self.compute_config.plan_ngpus

    @property
    def update_freq(self):
        return self.global_batch_size // self.micro_batch_size // self.scaling_factor

    def create_model(self) -> torch.nn.Module:
        kwargs = self.create_kwarg(self.model_args)
        return self.model_type(**kwargs)

    def create_parallel_optimizer(self, parallel_model: ParallelModule):
        kwargs = self.create_kwarg(self.optimizer_args)
        optimizer_class = load_type(self.optimizer_class)
        return build_optimizer(parallel_model, optimizer_class, **kwargs)

    def create_dataset(self, stage='train'):
        dataset_args = getattr(self, f'{stage}_dataset_args')
        if not dataset_args:
            return None
        kwargs = self.create_kwarg(dataset_args)
        dataset_class = load_type(self.dataset_class)
        if issubclass(dataset_class, torch.utils.data.IterableDataset):
            raise ValueError("IterableDataset is not supported")
        return dataset_class(**kwargs)

    def create_sampler(self, dataset, stage='train'):
        sampler_args = getattr(self, f'{stage}_dataset_sampler_args')
        sampler_args = sampler_args or self.train_dataset_sampler_args
        kwargs = self.create_kwarg(sampler_args)
        kwargs['dataset'] = dataset
        kwargs['num_replicas'] = self.compute_config.runtime_ngpus // self.compute_config.plan_ngpus
        kwargs['rank'] = torch.distributed.get_rank() // self.compute_config.plan_ngpus
        sampler_class = load_type(self.dataset_sampler_class)
        return sampler_class(**kwargs)

    def create_dataloader(self, stage='train', dataset=None):
        dataloader_args = getattr(self, f'{stage}_dataloader_args')
        dataloader_args = dataloader_args or self.train_dataloader_args
        kwargs = self.create_kwarg(dataloader_args)
        kwargs['dataset'] = dataset or self.create_dataset(stage)
        if kwargs['dataset'] is None:
            return None
        if 'collate_fn' in kwargs:
            # special handling for collate_fn as a function
            # here we don't use self.collate_fn to avoid its implementation hacking
            kwargs['collate_fn'] = load_type(kwargs['collate_fn'])
        kwargs['batch_size'] = self.micro_batch_size
        kwargs['sampler'] = self.create_sampler(kwargs['dataset'], stage)
        dataloader_class = load_type(self.dataloader_class)
        return dataloader_class(**kwargs)

    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        if not self.lr_scheduler_class:
            return None
        kwargs = self.create_kwarg(self.lr_scheduler_args)
        lr_scheduler_class = load_type(self.lr_scheduler_class)
        return lr_scheduler_class(optimizer, **kwargs)
