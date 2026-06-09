#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .logger_base import LoggerBase

logger = logging.getLogger(__name__)


class AsyncLogger(LoggerBase):
    """
    A composite logger that wraps multiple sub-loggers and ensures all
    log_metrics calls are non-blocking.

    - Sub-loggers that are already async (is_async() returns True) are called directly.
    - Sub-loggers that are synchronous are dispatched to a thread pool.
    """

    def __init__(self, loggers: List[LoggerBase], *, max_workers: int = 1):
        super().__init__()
        self._async_loggers: List[LoggerBase] = []
        self._sync_loggers: List[LoggerBase] = []
        for lg in loggers:
            if lg.is_async():
                self._async_loggers.append(lg)
            else:
                self._sync_loggers.append(lg)

        self._max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None

    def is_async(self) -> bool:
        return True

    def setup(self, config: Dict) -> None:
        for lg in self._async_loggers + self._sync_loggers:
            lg.setup(config)

        if self._sync_loggers:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

    def log_metrics(self, metrics: Dict[str, float], step: int, *, tag: Optional[str] = None) -> None:
        for lg in self._async_loggers:
            try:
                lg.log_metrics(metrics, step, tag=tag)
            except Exception:
                logger.exception("Error in async logger %s", type(lg).__name__)

        if self._executor is not None:
            self._executor.submit(self._log_sync, metrics, step, tag)

    def finalize(self) -> None:
        if self._executor is not None:
            # wait for all pending log tasks to complete
            self._executor.shutdown(wait=True)
            self._executor = None

        for lg in self._async_loggers + self._sync_loggers:
            try:
                lg.finalize()
            except Exception:
                logger.exception("Error in finalize of logger %s", type(lg).__name__)

    def _log_sync(self, metrics: Dict[str, float], step: int, tag: Optional[str]) -> None:
        for lg in self._sync_loggers:
            try:
                lg.log_metrics(metrics, step, tag=tag)
            except Exception:
                logger.exception("Error in sync logger %s", type(lg).__name__)
