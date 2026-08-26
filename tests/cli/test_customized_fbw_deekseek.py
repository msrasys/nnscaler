#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
#
# The deferred weight-gradient store and callback scheduling pattern are
# adapted from DeepSeek DualPipe for nnScaler.
# Copyright (c) 2025 DeepSeek
# DualPipe is licensed under the MIT License.
# https://github.com/deepseek-ai/DualPipe

"""DeepSeek-style FBW: B saves explicit dW callbacks and W runs them later.

The generated scheduler is unchanged. It still calls the module-level
``nnscaler.runtime.executor.backward_input`` and ``backward_weight`` symbols;
the test temporarily replaces those two symbols at runtime.
"""

from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Iterable, Optional

import pytest
import torch

import nnscaler
import nnscaler.runtime.executor as executor
from nnscaler.cli import (
    CheckpointConfig,
    DataloaderConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    Trainer,
    TrainerArgs,
)
from nnscaler.parallel import ComputeConfig
from nnscaler.runtime.executor import Executor
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_equal


class _WeightGradStore:
    """Per-process FIFO of complete dW callback batches, keyed by segment."""

    # `cache` belongs to the B that is currently traversing custom operators.
    # `queues[name]` contains only completed B batches ready for matching W calls.
    enabled = False
    active_name: Optional[str] = None
    cache: list[Callable[[], None]] = []
    queues: dict[str, deque[list[Callable[[], None]]]] = defaultdict(deque)
    deferred = 0
    executed = 0

    @classmethod
    def begin(cls, name: str) -> None:
        # A rank executes one Python autograd traversal at a time. Supporting
        # concurrent/reentrant B calls would require task-local state instead.
        assert not cls.enabled
        cls.enabled = True
        cls.active_name = name
        cls.cache = []

    @classmethod
    def put(cls, callback: Callable[[], None]) -> None:
        assert cls.enabled and cls.active_name is not None
        cls.cache.append(callback)
        cls.deferred += 1

    @classmethod
    def flush(cls) -> None:
        assert cls.enabled and cls.active_name is not None
        # `cache` is the unpublished batch of dW callbacks produced by one B.
        # Publish it atomically so one W pops exactly one complete microbatch;
        # `abort()` can discard a partial batch if B fails before this point.
        cls.queues[cls.active_name].append(cls.cache)
        cls.enabled = False
        cls.active_name = None
        cls.cache = []

    @classmethod
    def abort(cls) -> None:
        cls.enabled = False
        cls.active_name = None
        cls.cache = []

    @classmethod
    def pop(cls, name: str) -> None:
        assert cls.queues[name], f'No deferred weight backward for segment {name}'
        # An empty list is a valid batch: the corresponding B simply found no
        # custom dW work. Only a missing batch indicates broken B/W pairing.
        for callback in cls.queues[name].popleft():
            callback()
            cls.executed += 1

    @classmethod
    def clear(cls) -> None:
        cls.enabled = False
        cls.active_name = None
        cls.cache = []
        cls.queues = defaultdict(deque)
        cls.deferred = 0
        cls.executed = 0

    @classmethod
    def check_clear(cls) -> None:
        assert not cls.enabled
        assert not cls.cache
        assert all(not queue for queue in cls.queues.values())
        assert cls.deferred == cls.executed


@nnscaler.register_op('a b, b c -> a c')
class _DeepSeekLinearFunction(torch.autograd.Function):
    """Opaque Linear whose expensive dW GEMM can be moved from B to W."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor, weight)
        return input_tensor @ weight

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        input_tensor, weight = ctx.saved_tensors
        # dX must be produced during B so the preceding pipeline stage can run.
        input_grad = output_grad @ weight.T if ctx.needs_input_grad[0] else None

        if not ctx.needs_input_grad[1]:
            return input_grad, None

        if not _WeightGradStore.enabled:
            # Baseline/full backward path: return dW to autograd normally.
            return input_grad, input_tensor.T @ output_grad

        # Keep only the tensors needed by the explicit dW formula. They are
        # detached because W does not need to traverse their autograd history.
        # The incoming output_grad was synchronized at the B entry before this
        # callback was created, so another synchronization in W is redundant.
        saved_tensors = [input_tensor.detach(), output_grad.detach()]

        def accumulate_weight_grad() -> None:
            saved_input, saved_output_grad = saved_tensors
            weight_grad = saved_input.T @ saved_output_grad
            # Do not assign `weight.grad = weight_grad` directly. Reducers rely
            # on the weight's AccumulateGrad hook; leaf backward triggers it.
            torch.autograd.backward((weight,), grad_tensors=(weight_grad,))

        _WeightGradStore.put(accumulate_weight_grad)
        return input_grad, None


class _DeepSeekLinear(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(dim, dim))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return _DeepSeekLinearFunction.apply(input_tensor, self.weight)


class _DeepSeekModel(torch.nn.Module):
    def __init__(self, dim: int = 8, nlayers: int = 4):
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList(
            [_DeepSeekLinear(dim) for _ in range(nlayers)]
        )

    def forward(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        output = sample['input']
        for layer in self.layers:
            output = torch.tanh(layer(output))
        return torch.nn.functional.mse_loss(output, sample['target'])


class _DeepSeekDataset:
    def __init__(self, dim: int = 8, size: int = 16):
        generator = torch.Generator().manual_seed(1)
        self.input = torch.randn(size, dim, generator=generator)
        self.target = torch.randn(size, dim, generator=generator)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {'input': self.input[index], 'target': self.target[index]}

    def __len__(self) -> int:
        return len(self.input)


def _deepseek_backward_input(
    name: str,
    input_tensors: list[torch.Tensor],
    output_tensors: list[torch.Tensor],
    output_tensor_grads: list[Optional[torch.Tensor]],
    weights: Iterable[torch.nn.Parameter],
):
    # Generated async adapters may hand B a tensor with pending work. Resolve it
    # once here, before autograd and before custom operators retain output_grad.
    del weights
    output_tensor_grads = executor.sync_tensors(output_tensor_grads)

    # Unlike nnScaler's graph-based FBW helper, DeepSeek runs an ordinary full
    # traversal. Custom operators still emit dX but enqueue dW instead of
    # returning it, so the traversal itself performs only the B work.
    _WeightGradStore.begin(name)
    try:
        result = Executor.backward(
            name, input_tensors, output_tensors, output_tensor_grads
        )
    except Exception:
        _WeightGradStore.abort()
        raise
    _WeightGradStore.flush()
    return result


def _deepseek_backward_weight(
    name: str,
    weights: Iterable[torch.nn.Parameter],
) -> None:
    # The callbacks already capture their weights. `name` selects the FIFO for
    # this segment, and one W consumes exactly one completed B batch.
    del weights
    _WeightGradStore.pop(name)


def _trainer_args(work_dir: Path, mode: str) -> TrainerArgs:
    # baseline uses the same opaque model with ordinary FB scheduling;
    # deepseek_async additionally proves the B-entry sync is sufficient.
    use_fbw = mode != 'baseline'
    return TrainerArgs(
        instance_name=f'customized_fbw_{mode}',
        compute_config=ComputeConfig(
            plan_ngpus=2,
            runtime_ngpus=2,
            use_end2end=True,
            use_fbw=use_fbw,
            use_async_comm=mode == 'deepseek_async',
            pas_config={
                'pipeline_nmicros': 2,
                'pipeline_nstages': 2,
                'pipeline_scheduler': '1f1b',
            },
        ),
        gen_reuse='override',
        gen_savedir=work_dir / 'gen',
        pas_policy='hybrid',
        model=ModelConfig(type=_DeepSeekModel),
        optimizer=OptimizerConfig(type=torch.optim.Adam, args={'lr': 0.01}),
        dataset=DatasetConfig(
            type=_DeepSeekDataset,
            train_args={'size': 16},
        ),
        dataloader=DataloaderConfig(train_args={'drop_last': True}),
        checkpoint=CheckpointConfig(
            save_dir=work_dir / 'checkpoints',
            save_type='deduped',
            save_best=False,
        ),
        micro_batch_size=2,
        global_batch_size=4,
        max_train_steps=2,
        val_every_n_epochs=None,
        enable_progress_bar=False,
        log_progress_every_n_train_steps=None,
        seed=0,
    )


def _deepseek_worker(root_dir: str, mode: str) -> None:
    work_dir = Path(root_dir) / mode
    _WeightGradStore.clear()
    patch = executor.custom_fbw(
        _deepseek_backward_input,
        _deepseek_backward_weight,
    ) if mode != 'baseline' else nullcontext()

    with patch:
        trainer = Trainer(train_args=_trainer_args(work_dir, mode))
        trainer.run()

    assert trainer.model.use_scheduler
    assert trainer.model.nmicros_per_scheduler_step == 2

    if mode != 'baseline':
        # Summing over ranks also proves that deferred work was exercised on
        # every stage collectively, not merely that checkpoints happened to load.
        _WeightGradStore.check_clear()
        counts = torch.tensor(
            [_WeightGradStore.deferred, _WeightGradStore.executed],
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(counts)
        assert counts[0].item() > 0
        assert counts[0].item() == counts[1].item()

    if trainer.rank == 0:
        checkpoint_files = list((work_dir / 'checkpoints' / 'last').glob('*.ckpt'))
        Trainer.merge_checkpoint(
            checkpoint_files,
            Path(root_dir) / f'{mode}.pt',
        )
    torch.distributed.barrier()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason='lack of gpu devices',
)
def test_deepseek_customized_fbw(tmp_path):
    launch_torchrun(2, _deepseek_worker, str(tmp_path), 'baseline')
    launch_torchrun(2, _deepseek_worker, str(tmp_path), 'deepseek')
    launch_torchrun(2, _deepseek_worker, str(tmp_path), 'deepseek_async')

    baseline = torch.load(tmp_path / 'baseline.pt', weights_only=False)
    deepseek = torch.load(tmp_path / 'deepseek.pt', weights_only=False)
    deepseek_async = torch.load(tmp_path / 'deepseek_async.pt', weights_only=False)
    # Compare optimizer state as well as parameters: matching final weights alone
    # can hide missing/duplicated gradients over a very short run.
    assert_equal(baseline['model'], deepseek['model'])
    assert_equal(baseline['optimizer'], deepseek['optimizer'])
    assert_equal(baseline['model'], deepseek_async['model'])
    assert_equal(baseline['optimizer'], deepseek_async['optimizer'])
