import atexit
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import yaml
import torch
try:
    _tensorboard_writers = []
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from nnscaler.utils import rank_zero_only
from .logger_base import LoggerBase


class TensorBoardLogger(LoggerBase):
    def __init__(
        self,
        name: str,
        root_dir: str,
        **kwargs,
    ):
        if SummaryWriter is None:
            raise RuntimeError(
                "tensorboard not found, please install with: pip install tensorboard"
            )

        super().__init__()
        self._name = name
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._kwargs = kwargs

        self._summary_writer = None

    @property
    def log_dir(self) -> str:
        """
        Root directory to save logging output, which is `_log_dir/_name`.
        """
        sub_path = [s for s in [self._name] if s]
        ld = self._root_dir.joinpath(*sub_path)
        ld.mkdir(parents=True, exist_ok=True)
        return str(ld)

    @rank_zero_only
    def setup(self, config: Dict) -> None:
        self._ensure_writer()
        self._summary_writer.add_text("config", yaml.dump(config))

    def _ensure_writer(self):
        if not self._summary_writer:
            self._summary_writer = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
            _tensorboard_writers.append(self._summary_writer)
        return self._summary_writer

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self._ensure_writer()

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                self._summary_writer.add_scalars(k, v, step)
            else:
                self._summary_writer.add_scalar(k, v, step)

    @rank_zero_only
    def finalize(self) -> None:
        if self._summary_writer:
            self._summary_writer.close()
            _tensorboard_writers.remove(self._summary_writer)


def _close_writers():
    for w in _tensorboard_writers:
        w.close()

# Close all writers on exit
atexit.register(_close_writers)
