#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Optional, Dict


class LoggerBase(ABC):
    """
    Base class for experiment loggers.
    """

    @abstractmethod
    def setup(self, config: Dict) -> None:
        """
        Setup logger with trainer args. This is useful for saving hyperparameters.
        Will be called once before `log_metrics`
        """
        ...

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int, *, tag: Optional[str] = None) -> None:
        ...

    @abstractmethod
    def finalize(self) -> None:
        ...

    def is_async(self) -> bool:
        """
        Whether this logger is asynchronous.
        If True, the trainer will call `log_metrics` in a separate thread and will not wait for it to finish.
        """
        return False
