from typing import Any, Dict, List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from nnscaler.cli.trainer import Trainer
    from nnscaler.cli.trainer_args import AggregatedOutputs


class TrainHook:
    """
    Note: All hooks are called in all ranks, and the inputs of hooks are only the local data.
    """
    def on_train_start(self, trainer: 'Trainer') -> None:
        """Called at the beginning of training"""

    def on_train_end(self, trainer: 'Trainer') -> None:
        """Called at the end of training"""

    def on_val_start(self, trainer: 'Trainer') -> None:
        """Called at the beginning of validation"""

    def on_val_end(self, trainer: 'Trainer', val_loss: float) -> None:
        """Called at the end of validation"""

    def on_epoch_start(self, trainer: 'Trainer', epoch: int) -> None:
        """
        Called at the beginning of each epoch
        Args:
            epoch: the current epoch index
        """

    def on_epoch_end(self, trainer: 'Trainer', epoch: int) -> None:
        """
        Called at the end of each epoch
        Args:
            epoch: the current epoch index
        """

    def on_train_step_start(self, trainer: 'Trainer', batches: List[Any], idx: int) -> None:
        """
        Called at the beginning of each training step
        Please note one train step may contain multiple batches
        Args:
            batches: the current batches
            idx: the index of current step
        """

    def on_train_step_end(self, trainer: 'Trainer', outputs: List[Any], batches: List[Any], idx: int) -> None:
        """
        Called at the end of each training step
        Args:
            outputs: the outputs of the train_step
            batches: the current batches
            idx: the index of current step
        """

    def on_val_step_start(self, trainer: 'Trainer', batches: List[Any], idx: int) -> None:
        """
        Called at the beginning of each validating step
        Please note one val step may contain multiple batches
        Args:
            batches: the current batches
            idx: the index of current step
        """

    def on_val_step_end(self, trainer: 'Trainer', outputs: List[Any], batches: List[Any], idx: int) -> None:
        """
        Called at the end of each validating step
        Args:
            outputs: the outputs of the val_step
            batches: the current batches
            idx: the index of current step
        """

    def after_aggregate_train_step_outputs(self, trainer: 'Trainer', aggregated_outputs: 'AggregatedOutputs', train_loss: float, idx: int) -> None:
        """
        Called after aggregating outputs in train step
        Args:
            aggregated_outputs: the aggregated outputs
            train_loss: the loss of the current step
            idx: the index of current step
        """

    def after_aggregate_val_step_outputs(self, trainer: 'Trainer', aggregated_outputs: 'AggregatedOutputs', val_loss: float, idx: int) -> None:
        """
        Called after aggregating outputs in val step
        Args:
            aggregated_outputs: the aggregated outputs
            val_loss: the loss of the current step
            idx: the index of current step
        """

    def before_zero_grad(self, trainer: 'Trainer') -> None:
        """
        Called before zero_grad
        """

    def after_zero_grad(self, trainer: 'Trainer') -> None:
        """
        Called after zero_grad
        """

    def before_gnorm_clip(self, trainer: 'Trainer') -> None:
        """
        Called before gradient clipping
        """

    def after_gnorm_clip(self, trainer: 'Trainer', gnorm: torch.Tensor) -> None:
        """
        Called after gradient clipping
        """

    def before_optimizer_step(self, trainer: 'Trainer') -> None:
        """
        Called before optimizer.step()
        """

    def after_optimizer_step(self, trainer: 'Trainer') -> None:
        """
        Called after optimizer.step()
        """

    def on_load_checkpoint(self, trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:
        """
        Called after loading checkpoint.
        If you saved something with `on_save_checkpoint` this is
        your chance to restore this.

        Args:
            checkpoint: the checkpoint loaded
        """

    def on_save_checkpoint(self, trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:
        """
        Called before saving checkpoint.
        If you want to save something, you can add it to the checkpoint here.

        Args:
            checkpoint: the checkpoint to be saved
        """
