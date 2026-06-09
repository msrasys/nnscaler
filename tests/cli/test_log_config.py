#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
from typing import Dict, List, Optional

import pytest

from nnscaler.cli.loggers import LoggerBase, AsyncLogger
from nnscaler.cli.trainer_args import LogConfig, LogsConfig, TrainerArgs


class FakeLogger(LoggerBase):
    """A synchronous logger that records all calls."""

    def __init__(self, name: str = 'fake'):
        self.name = name
        self.setup_calls: List[Dict] = []
        self.log_calls: List[tuple] = []
        self.finalized = False

    def setup(self, config: Dict) -> None:
        self.setup_calls.append(config)

    def log_metrics(self, metrics: Dict[str, float], step: int, *, tag: Optional[str] = None) -> None:
        self.log_calls.append((metrics, step, tag))

    def finalize(self) -> None:
        self.finalized = True


class FakeAsyncLogger(FakeLogger):
    """A logger that claims to be async."""

    def is_async(self) -> bool:
        return True


class FailingLogger(FakeLogger):
    """A logger whose log_metrics always raises."""

    def log_metrics(self, metrics, step, *, tag=None):
        raise RuntimeError("boom")


# ─── LogsConfig.deserialize: old format (list) ────────────────


def test_old_format_list_of_dicts():
    data = [
        {'type': 'TensorBoardLogger', 'args': {'name': 'tb', 'root_dir': '/tmp/tb'}},
    ]
    config = LogsConfig.deserialize(data)
    assert isinstance(config, LogsConfig)
    assert len(config.logs) == 1
    assert config.logs[0].type == 'nnscaler.cli.loggers.TensorBoardLogger'
    # defaults should apply
    assert config.async_logging is True
    assert config.max_workers == 1


def test_old_format_list_of_multiple_dicts():
    data = [
        {'type': 'nnscaler.cli.loggers.TensorBoardLogger', 'args': {'name': 'tb', 'root_dir': '/tmp/tb'}},
        {'type': 'nnscaler.cli.loggers.WandbLogger', 'args': {'project': 'test'}},
    ]
    config = LogsConfig.deserialize(data)
    assert len(config.logs) == 2
    assert config.logs[0].type == 'nnscaler.cli.loggers.TensorBoardLogger'
    assert config.logs[1].type == 'nnscaler.cli.loggers.WandbLogger'


def test_old_format_tuple():
    data = (
        {'type': 'TensorBoardLogger', 'args': {'name': 'tb', 'root_dir': '/tmp/tb'}},
    )
    config = LogsConfig.deserialize(data)
    assert isinstance(config, LogsConfig)
    assert len(config.logs) == 1


def test_old_format_dict_with_numeric_keys():
    """Command-line parsed format: {'0': {...}, '1': {...}}"""
    data = {
        '0': {'type': 'TensorBoardLogger', 'args': {'name': 'tb', 'root_dir': '/tmp/tb'}},
    }
    config = LogsConfig.deserialize(data)
    assert isinstance(config, LogsConfig)
    assert len(config.logs) == 1


def test_old_format_empty_list():
    config = LogsConfig.deserialize([])
    assert isinstance(config, LogsConfig)
    assert len(config.logs) == 0


# ─── LogsConfig.deserialize: new format (dict with fields) ────


def test_new_format_full():
    data = {
        'async_logging': False,
        'max_workers': 4,
        'logs': [
            {'type': 'TensorBoardLogger', 'args': {'name': 'tb', 'root_dir': '/tmp/tb'}},
        ],
    }
    config = LogsConfig.deserialize(data)
    assert isinstance(config, LogsConfig)
    assert config.async_logging is False
    assert config.max_workers == 4
    assert len(config.logs) == 1


def test_new_format_defaults():
    data = {
        'logs': [
            {'type': 'TensorBoardLogger', 'args': {'name': 'tb', 'root_dir': '/tmp/tb'}},
        ],
    }
    config = LogsConfig.deserialize(data)
    assert config.async_logging is True
    assert config.max_workers == 1
    assert len(config.logs) == 1


def test_new_format_async_disabled():
    data = {
        'async_logging': False,
        'logs': [
            {'type': 'TensorBoardLogger', 'args': {'name': 'tb', 'root_dir': '/tmp/tb'}},
        ],
    }
    config = LogsConfig.deserialize(data)
    assert config.async_logging is False


def test_new_format_empty_logs():
    data = {
        'async_logging': True,
        'max_workers': 2,
        'logs': [],
    }
    config = LogsConfig.deserialize(data)
    assert len(config.logs) == 0
    assert config.max_workers == 2


# ─── LogConfig type resolution ────────────────────────────────


def test_log_config_short_type_expanded():
    lc = LogConfig(type='TensorBoardLogger')
    assert lc.type == 'nnscaler.cli.loggers.TensorBoardLogger'


def test_log_config_fqn_type_unchanged():
    lc = LogConfig(type='nnscaler.cli.loggers.TensorBoardLogger')
    assert lc.type == 'nnscaler.cli.loggers.TensorBoardLogger'


def test_log_config_type_required():
    with pytest.raises(ValueError, match="type is required"):
        LogConfig()


# ─── AsyncLogger behavior ─────────────────────────────────────


def test_async_logger_is_async():
    al = AsyncLogger([FakeLogger()])
    assert al.is_async() is True


def test_async_logger_sync_dispatched_to_pool():
    lg = FakeLogger()
    al = AsyncLogger([lg])
    al.setup({'key': 'value'})
    al.log_metrics({'loss': 0.5}, step=1)
    al.finalize()

    assert lg.setup_calls == [{'key': 'value'}]
    assert lg.log_calls == [({'loss': 0.5}, 1, None)]
    assert lg.finalized


def test_async_logger_async_called_directly():
    lg = FakeAsyncLogger()
    al = AsyncLogger([lg])
    al.setup({'key': 'value'})
    al.log_metrics({'loss': 0.5}, step=1, tag='train')
    al.finalize()

    assert lg.log_calls == [({'loss': 0.5}, 1, 'train')]
    assert lg.finalized


def test_async_logger_mixed_sync_and_async():
    sync_lg = FakeLogger('sync')
    async_lg = FakeAsyncLogger('async')
    al = AsyncLogger([sync_lg, async_lg])
    al.setup({})
    al.log_metrics({'loss': 1.0}, step=0)
    al.log_metrics({'loss': 0.5}, step=1, tag='val')
    al.finalize()

    assert len(sync_lg.log_calls) == 2
    assert len(async_lg.log_calls) == 2
    assert sync_lg.finalized
    assert async_lg.finalized


def test_async_logger_max_workers():
    lg = FakeLogger()
    al = AsyncLogger([lg], max_workers=4)
    al.setup({})
    assert al._executor._max_workers == 4
    al.finalize()


def test_async_logger_no_executor_without_sync_loggers():
    lg = FakeAsyncLogger()
    al = AsyncLogger([lg])
    al.setup({})
    assert al._executor is None
    al.finalize()


def test_async_logger_finalize_waits_for_pending():
    lg = FakeLogger()
    al = AsyncLogger([lg])
    al.setup({})
    for i in range(10):
        al.log_metrics({'step': float(i)}, step=i)
    al.finalize()
    assert len(lg.log_calls) == 10


def test_async_logger_error_does_not_crash():
    lg = FailingLogger()
    al = AsyncLogger([lg])
    al.setup({})
    al.log_metrics({'x': 1.0}, step=0)
    # finalize should not raise
    al.finalize()


def test_async_logger_multiple_finalize_safe():
    lg = FakeLogger()
    al = AsyncLogger([lg])
    al.setup({})
    al.log_metrics({'x': 1.0}, step=0)
    al.finalize()
    al.finalize()  # second call should be a no-op
    assert lg.log_calls == [({'x': 1.0}, 0, None)]


# ─── LogsConfig passthrough ──────────────────────────────────


def test_logs_config_passthrough():
    config = LogsConfig(
        async_logging=False,
        max_workers=2,
        logs=[LogConfig(type='TensorBoardLogger')],
    )
    assert isinstance(config, LogsConfig)
    assert config.async_logging is False
    assert config.max_workers == 2
    assert len(config.logs) == 1


# ─── CLI argument parsing ────────────────────────────────────


_CONFIG_PATH = str(Path(__file__).with_name('trainer_args.yaml').resolve())

# common args to make TrainerArgs.__post_init__ happy with the yaml defaults
_COMMON_CLI_ARGS = [
    '-f', _CONFIG_PATH,
    '--compute_config.plan_ngpus', '2',
    '--compute_config.runtime_ngpus', '4',
]


def test_cli_old_format_log():
    """Old format: --log.0.type, --log.0.args.* (flat list via numeric keys)"""
    args = TrainerArgs.from_cli([
        *_COMMON_CLI_ARGS,
        '--log.0.type', 'nnscaler.cli.loggers.TensorBoardLogger',
        '--log.0.args.name', 'test-tb',
        '--log.0.args.root_dir', '/tmp/tb',
        '--log.1.type', 'nnscaler.cli.loggers.WandbLogger',
        '--log.1.args.name', 'test-wandb',
        '--log.1.args.project', 'nnscaler',
    ])
    assert isinstance(args.log, LogsConfig)
    assert len(args.log.logs) == 2
    assert args.log.logs[0].type == 'nnscaler.cli.loggers.TensorBoardLogger'
    assert args.log.logs[0].args == {'name': 'test-tb', 'root_dir': '/tmp/tb'}
    assert args.log.logs[1].type == 'nnscaler.cli.loggers.WandbLogger'
    assert args.log.logs[1].args == {'name': 'test-wandb', 'project': 'nnscaler'}
    # defaults for async_logging / max_workers
    assert args.log.async_logging is True
    assert args.log.max_workers == 1


def test_cli_new_format_log():
    """New format: --log.async_logging, --log.max_workers, --log.logs.0.*"""
    args = TrainerArgs.from_cli([
        *_COMMON_CLI_ARGS,
        '--log.async_logging', 'false',
        '--log.max_workers', '4',
        '--log.logs.0.type', 'nnscaler.cli.loggers.TensorBoardLogger',
        '--log.logs.0.args.name', 'test-tb',
        '--log.logs.0.args.root_dir', '/tmp/tb',
    ])
    assert isinstance(args.log, LogsConfig)
    assert args.log.async_logging is False
    assert args.log.max_workers == 4
    assert len(args.log.logs) == 1
    assert args.log.logs[0].type == 'nnscaler.cli.loggers.TensorBoardLogger'
    assert args.log.logs[0].args == {'name': 'test-tb', 'root_dir': '/tmp/tb'}


def test_cli_new_format_multiple_loggers():
    """New format with multiple loggers via CLI."""
    args = TrainerArgs.from_cli([
        *_COMMON_CLI_ARGS,
        '--log.async_logging', 'true',
        '--log.max_workers', '2',
        '--log.logs.0.type', 'nnscaler.cli.loggers.TensorBoardLogger',
        '--log.logs.0.args.name', 'test-tb',
        '--log.logs.0.args.root_dir', '/tmp/tb',
        '--log.logs.1.type', 'nnscaler.cli.loggers.WandbLogger',
        '--log.logs.1.args.name', 'test-wandb',
        '--log.logs.1.args.project', 'nnscaler',
    ])
    assert isinstance(args.log, LogsConfig)
    assert args.log.async_logging is True
    assert args.log.max_workers == 2
    assert len(args.log.logs) == 2
    assert args.log.logs[0].type == 'nnscaler.cli.loggers.TensorBoardLogger'
    assert args.log.logs[1].type == 'nnscaler.cli.loggers.WandbLogger'


def test_cli_no_log():
    """No log args: should produce empty LogsConfig."""
    args = TrainerArgs.from_cli([
        *_COMMON_CLI_ARGS,
    ])
    assert isinstance(args.log, LogsConfig)
    assert len(args.log.logs) == 0


# ─── TrainerArgs.__post_init__ log field handling ─────────────


def test_post_init_logs_config_kept():
    """LogsConfig instance is kept as-is through __post_init__."""
    args = TrainerArgs.from_cli([
        *_COMMON_CLI_ARGS,
        '--log.async_logging', 'false',
        '--log.max_workers', '3',
        '--log.logs.0.type', 'nnscaler.cli.loggers.TensorBoardLogger',
        '--log.logs.0.args.name', 'tb',
        '--log.logs.0.args.root_dir', '/tmp/tb',
    ])
    assert isinstance(args.log, LogsConfig)
    assert args.log.async_logging is False
    assert args.log.max_workers == 3
    assert len(args.log.logs) == 1


def test_post_init_list_converted_to_logs_config():
    """A list of LogConfig is converted to LogsConfig in __post_init__."""
    args = TrainerArgs.from_cli([*_COMMON_CLI_ARGS])
    # Manually set log to a list to trigger __post_init__ compat path
    log_list = [LogConfig(type='TensorBoardLogger')]
    args.log = log_list
    args.__post_init__()
    assert isinstance(args.log, LogsConfig)
    assert len(args.log.logs) == 1
    assert args.log.logs[0].type == 'nnscaler.cli.loggers.TensorBoardLogger'
    # defaults
    assert args.log.async_logging is True
    assert args.log.max_workers == 1


def test_post_init_invalid_log_raises():
    """An invalid log value raises ValueError in __post_init__."""
    args = TrainerArgs.from_cli([*_COMMON_CLI_ARGS])
    args.log = "invalid"
    with pytest.raises(ValueError, match="Invalid log config"):
        args.__post_init__()


# ─── create_loggers ──────────────────────────────────────────


def test_create_loggers_async_wraps(tmp_path):
    """When async_logging is True, create_loggers wraps loggers in AsyncLogger."""
    args = TrainerArgs.from_cli([
        *_COMMON_CLI_ARGS,
        '--log.async_logging', 'true',
        '--log.max_workers', '2',
        '--log.logs.0.type', 'nnscaler.cli.loggers.TensorBoardLogger',
        '--log.logs.0.args.name', 'test',
        '--log.logs.0.args.root_dir', str(tmp_path),
    ])
    loggers = args.create_loggers()
    assert len(loggers) == 1
    assert isinstance(loggers[0], AsyncLogger)
    assert loggers[0]._max_workers == 2


def test_create_loggers_no_async(tmp_path):
    """When async_logging is False, create_loggers returns raw loggers."""
    args = TrainerArgs.from_cli([
        *_COMMON_CLI_ARGS,
        '--log.async_logging', 'false',
        '--log.logs.0.type', 'nnscaler.cli.loggers.TensorBoardLogger',
        '--log.logs.0.args.name', 'test',
        '--log.logs.0.args.root_dir', str(tmp_path),
    ])
    loggers = args.create_loggers()
    assert len(loggers) == 1
    assert not isinstance(loggers[0], AsyncLogger)


def test_create_loggers_empty():
    """No log configs: create_loggers returns empty list regardless of async_logging."""
    args = TrainerArgs.from_cli([*_COMMON_CLI_ARGS])
    loggers = args.create_loggers()
    assert loggers == []
