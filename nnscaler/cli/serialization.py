#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Any, Callable, Protocol, Type
from pathlib import Path
import shutil
import logging

import torch

from nnscaler.runtime.serialization import load, save


logger = logging.getLogger(__name__)


class _LoadProc(Protocol):
    def __call__(self, f: str | Path, *, device='cpu') -> Any: ...


class _SaveProc(Protocol):
    def __call__(self, obj: Any, f: str | Path) -> None: ...


class CheckpointFormat(Protocol):
    """
    A placeholder class for new serialization formats.
    """
    name: str
    suffix: str

    @classmethod
    def load(cls, f: str | Path, *, device='cpu') -> Any:
        ...

    @classmethod
    def save(cls, obj: Any, f: str | Path) -> None:
        ...


class SerializationRunner(Protocol):
    name: str

    def run_load(
        self,
        load_func: _LoadProc,
        f: str | Path,
        *,
        device='cpu'
    ) -> Any:
        ...

    def run_save(
        self,
        save_func: _SaveProc,
        obj: Any,
        f: str | Path
    ) -> None:
        ...

    def flush(self) -> None:
        """
        Flushes any pending operations for saving.
        Loading operations are assumed to be synchronous.
        """
        ...


class _DefaultSerializationRunner:
    name: str = ''

    def run_load(
        self,
        load_func: _LoadProc,
        f: str | Path,
        *,
        device='cpu'
    ) -> Any:
        return load_func(f, device=device)

    def run_save(
        self,
        save_func: _SaveProc,
        obj: Any,
        f: str | Path
    ) -> None:
        save_func(obj, f)

    def flush(self) -> None:
        pass


def make_hybrid_serialization_runner(
    load_serializer: Type[SerializationRunner],
    save_serializer: Type[SerializationRunner]
) -> Type[SerializationRunner]:
    """
    Creates a hybrid serialization runner that uses different runners for loading and saving.
    """
    class HybridSerializationRunner(SerializationRunner):
        name = f"{load_serializer.name}:{save_serializer.name}"

        def __init__(self, load_args=None, save_args=None):
            self._load_runner = load_serializer(**(load_args or {}))
            self._save_runner = save_serializer(**(save_args or {}))

        def run_load(
            self,
            load_func: _LoadProc,
            f: str | Path,
            *,
            device='cpu'
        ) -> Any:
            return self._load_runner.run_load(load_func, f, device=device)

        def run_save(
            self,
            save_func: _SaveProc,
            obj: Any,
            f: str | Path
        ) -> None:
            self._save_runner.run_save(save_func, obj, f)

        def flush(self) -> None:
            self._save_runner.flush()

    return HybridSerializationRunner


def _torch_load(f: str | Path, *, device='cpu') -> Any:
    return torch.load(f, map_location=device, weights_only=False)


def _torch_save(obj: Any, f: str | Path) -> None:
    torch.save(obj, f)


class Checkpointer:
    # the format of the checkpoint file
    # keys: epoch, step, rank
    # currently it is not configurable
    # TODO: make it configurable
    CHECKPOINT_FILE_NAME_FORMAT: str = '{rank}{suffix}'
    CHECKPOINT_FILE_FORMAT: str = '{epoch:04d}-{step:04d}/' + CHECKPOINT_FILE_NAME_FORMAT
    CHECKPOINT_LAST_DIR_NAME: str = 'last'
    CHECKPOINT_BEST_DIR_NAME: str = 'best'
    CHECKPOINT_MERGED_FILE_NAME: str = 'merged{suffix}'
    CHECKPOINT_LAST_FILE_FORMAT: str = 'last/{rank}{suffix}'
    CHECKPOINT_BEST_FILE_FORMAT: str = 'best/{rank}{suffix}'
    NAME_MAP: dict[str, str] = {
        'pt': '.ckpt',
        'safetensors': '.safetensors'
    }
    SUFFIX_MAP: dict[str, str] = {v: k for k, v in NAME_MAP.items()}
    # will use torch.load and torch.save for other suffixes
    SUFFIX_HANDLERS: dict[str, tuple[_LoadProc, _SaveProc]] = {
        '.safetensors': (load, save),
    }
    REGISTERED_RUNNERS: dict[str, Type[SerializationRunner]] = {
        '': _DefaultSerializationRunner,
    }

    def __init__(self, format: str = 'pt', serializer: str = None, serializer_args: dict[str, Any] = None):
        """
        Args:
            format (`str`, *optional*, defaults to `"pt"`):
                The checkpoint format to use. Builtin formats are:
                - `"pt"`: PyTorch checkpoint format.
                - `"safetensors"`: Safetensors format.
            serializer (`str`, *optional*):
                The serialization runner to use. Builtin runners are:
                - `""` (empty string): Default runner that directly uses the load and save functions.
                You can also specify a hybrid runner by using the format `load_serializer:save_serializer`,
                e.g., `"split:async"`.
            serializer_args (`dict`, *optional*):
                args for the serialization runner.
        """
        if format not in self.NAME_MAP:
            raise ValueError(f"Unsupported checkpoint format: {format}")
        self.format = format
        self.suffix = self.NAME_MAP[format]

        self.runner: SerializationRunner
        serializer = serializer or ''

        if ':' in serializer:
            parts = serializer.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid hybrid serialization runner: {serializer}")
            load_serializer_name = parts[0]
            save_serializer_name = parts[1]
            if load_serializer_name not in self.REGISTERED_RUNNERS:
                raise ValueError(f"Unsupported serialization runner: {load_serializer_name}")
            if save_serializer_name not in self.REGISTERED_RUNNERS:
                raise ValueError(f"Unsupported serialization runner: {save_serializer_name}")
            load_serializer_type = self.REGISTERED_RUNNERS[load_serializer_name]
            save_serializer_type = self.REGISTERED_RUNNERS[save_serializer_name]
            runner_cls = make_hybrid_serialization_runner(
                load_serializer_type,
                save_serializer_type
            )
        else:
            if serializer not in self.REGISTERED_RUNNERS:
                raise ValueError(f"Unsupported serialization runner: {serializer}")
            runner_cls = self.REGISTERED_RUNNERS[serializer]

        self.runner = runner_cls(**(serializer_args or {}))

    def get_checkpoint_file_path(self, epoch: int, step: int, rank: int) -> str:
        return self.CHECKPOINT_FILE_FORMAT.format(epoch=epoch, step=step, rank=rank, suffix=self.suffix)

    def get_last_checkpoint_file_path(self, rank: int) -> str:
        return self.CHECKPOINT_LAST_FILE_FORMAT.format(rank=rank, suffix=self.suffix)

    def get_best_checkpoint_file_path(self, rank: int) -> str:
        return self.CHECKPOINT_BEST_FILE_FORMAT.format(rank=rank, suffix=self.suffix)

    def get_merged_checkpoint_file_name(self) -> str:
        return self.CHECKPOINT_MERGED_FILE_NAME.format(suffix=self.suffix)

    def get_last_dir_name(self) -> str:
        return self.CHECKPOINT_LAST_DIR_NAME

    def get_best_dir_name(self) -> str:
        return self.CHECKPOINT_BEST_DIR_NAME

    def load(self, f: str | Path, *, device='cpu') -> Any:
        """
        Loads a checkpoint file

        Args:
            f: filename of the checkpoint file.
               if the suffix is .safetensors, it will be loaded as safetensors file.
               otherwise, it will be loaded as a PyTorch checkpoint file.
            device (`str`, *optional*, defaults to `"cpu"`):
                The device on which you want the tensors.
        """
        suffix = Path(f).suffix
        if suffix in self.SUFFIX_HANDLERS:
            load_func, _ = self.SUFFIX_HANDLERS[suffix]
        else:
            load_func = _torch_load

        return self.runner.run_load(load_func, f, device=device)

    def save(self, obj: Any, f: str | Path) -> None:
        """
        Saves a checkpoint file

        Args:
            obj (`Any`):
                The object to save.
            f: filename of the checkpoint file.
               if the suffix is .safetensors, it will be saved as safetensors file.
               otherwise, it will be saved as a PyTorch checkpoint file.
        """
        suffix = Path(f).suffix
        if suffix in self.SUFFIX_HANDLERS:
            _, save_func = self.SUFFIX_HANDLERS[suffix]
        else:
            save_func = _torch_save

        self.runner.run_save(save_func, obj, f)

    def load_for_rank(self, dir: str | Path, rank: int, device='cpu') -> Any:
        """
        Loads a checkpoint file for a specific rank

        Args:
            dir (`str`):
                The directory where the checkpoint files are stored.
            rank (`int`):
                The rank of the checkpoint file to load.
            device (`str`, `int`, *optional*):
                The device on which you want the tensors.
        """
        for suffix in self.NAME_MAP.values():
            f = Path(dir) / f"{rank}{suffix}"
            if f.exists():
                return self.load(f, device=device)
        raise FileNotFoundError(f"No checkpoint file found for rank {rank} in directory {dir}")

    def save_for_rank(self, obj: Any, dir: str | Path, rank: int) -> None:
        """
        Saves a checkpoint file for a specific rank

        Args:
            obj (`Any`):
                The object to save.
            dir (`str`):
                The directory where the checkpoint files are stored.
            rank (`int`):
                The rank of the checkpoint file to save.
        """
        f = Path(dir) / self.CHECKPOINT_FILE_NAME_FORMAT.format(rank=rank, suffix=self.suffix)
        self.save(obj, f)

    def remove_for_rank(self, dir: str | Path, rank: int) -> None:
        """
        Removes a checkpoint file for a specific rank.
        Args:
            dir (`str`):
                The directory where the checkpoint files are stored.
            rank (`int`):
                The rank of the checkpoint file to remove.
        """
        self.flush()

        suffixes = set(list(self.NAME_MAP.values()) + [self.suffix])
        for suffix in suffixes:
            f = Path(dir) / f"{rank}{suffix}"
            if f.is_symlink() or f.exists():
                logger.warning(f"Removing checkpoint file: {f}")
                f.unlink()
            for extra_file in Path(dir).glob(f"{rank}{suffix}.*"):
                logger.warning(f"Removing extra checkpoint file: {extra_file}")
                extra_file.unlink()

    def copy_for_rank(self, src: str | Path, dst: str | Path, rank: int, symlink: bool = False) -> None:
        """
        Copies a checkpoint file for a specific rank from one directory to another.
        Args:
            src (`str`):
                The source directory where the checkpoint files are stored.
            dst (`str`):
                The destination directory where the checkpoint files will be copied.
            rank (`int`):
                The rank of the checkpoint file to copy.
            symlink (`bool`, *optional*, defaults to `False`):
                Whether to create a symbolic link instead of copying the file.
        """

        self.flush()
        src = Path(src).resolve()
        dst = Path(dst).resolve()
        dst.mkdir(parents=True, exist_ok=True)

        src_f = Path(src) / f"{rank}{self.suffix}"
        dst_f = Path(dst) / f"{rank}{self.suffix}"

        if not src_f.exists():
            raise FileNotFoundError(f"No checkpoint file found for rank {rank} in directory {src}")

        if symlink:
            # this restricts symlink creation within the same directory
            # so we can create relative symlink safely
            if src.parent != dst.parent:
                raise ValueError("Cannot create symlink when source and destination are not in the same directory.")

        if symlink:
            if dst_f.exists() or dst_f.is_symlink():
                logger.warning(f"Removing existing checkpoint file: {dst_f}")
                dst_f.unlink()
            dst_f.symlink_to(Path('..') / src.name / src_f.name)
            for extra_file in src.glob(f"{rank}{self.suffix}.*"):
                dst_extra_file = Path(dst) / extra_file.name
                if dst_extra_file.exists() or dst_extra_file.is_symlink():
                    logger.warning(f"Removing existing extra checkpoint file: {dst_extra_file}")
                    dst_extra_file.unlink()
                dst_extra_file.symlink_to(Path('..') / src.name / extra_file.name)
        else:
            shutil.copy2(src_f, dst_f)
            for extra_file in src.glob(f"{rank}{self.suffix}.*"):
                dst_extra_file = Path(dst) / extra_file.name
                shutil.copy2(extra_file, dst_extra_file)

    def list_checkpoints(self, dir: str | Path) -> list[Path]:
        """
        List the main checkpoint files in a directory
        Args:
            dir (`str`):
                The directory where the checkpoint files are stored.
        Returns:
            (`list[Path]`):
                The list of checkpoint files in the directory.
        """
        self.flush()

        p = Path(dir)
        files = []
        for suffix in self.NAME_MAP.values():
            fs = list(p.glob(f"*{suffix}"))
            if fs:
                if files:
                    raise ValueError(f"Mixed checkpoint file formats in directory {dir}")
                else:
                    files.extend(fs)
        return files

    def flush(self) -> None:
        """
        Flushes any pending operations.
        """
        self.runner.flush()

    @classmethod
    def get_format(cls, suffix: str) -> str:
        """
        Gets the format name from the suffix.
        """
        suffix = '.' + suffix.lstrip('.')
        if suffix not in Checkpointer.SUFFIX_MAP:
            raise ValueError(f"Unsupported checkpoint suffix: {suffix}")
        return Checkpointer.SUFFIX_MAP[suffix]


def register_format(format: Type[CheckpointFormat]) -> None:
    """
    Registers a new serialization format.
    """
    suffix = '.' + format.suffix.lstrip('.')
    Checkpointer.NAME_MAP[format.name] = suffix
    Checkpointer.SUFFIX_MAP[suffix] = format.name
    Checkpointer.SUFFIX_HANDLERS[suffix] = (format.load, format.save)


def register_serialization_runner(runner: Type[SerializationRunner]) -> None:
    """
    Register a new serialization runner, which can intercept the load and save process.
    For example, file redirection, chunking, asynchronous IO or other logic.

    Please note if you create extra files during saving,
    you must make sure
    1. the suffix of the main checkpoint file must match registered formats.
    2. the name of extra files should start with the main checkpoint file name + '.',
        but the suffix should not conflict with registered formats.

    For example, if the input checkpoint file is `model.ckpt`,
    you must create a file called 'model.ckpt',
    and you can use extra file names like 'model.ckpt.1', 'model.ckpt.meta', 'model.ckpt.opt' etc.
    """
    if ':' in runner.name:
        raise ValueError("Serialization runner name cannot contain ':'")
    Checkpointer.REGISTERED_RUNNERS[runner.name] = runner


def convert_format(
    src: str | Path,
    dst: str | Path,
    *,
    src_serializer: str = None,
    src_serializer_args: dict = None,
    dst_serializer: str = None,
    dst_serializer_args: dict = None,
    device: str = 'cpu'
) -> None:
    """
    Converts a checkpoint file from one format to another.

    Args:
        src (`str` or `Path`):
            The input checkpoint file.
        dst (`str` or `Path`):
            The output checkpoint file.
        src_serializer (`str`, *optional*):
            The serialization runner of the input checkpoint file.
        src_serializer_args (`dict`, *optional*):
            The arguments for the serialization runner of the input checkpoint file.
        dst_serializer (`str`, *optional*):
            The serialization runner of the output checkpoint file.
        dst_serializer_args (`dict`, *optional*):
            The arguments for the serialization runner of the output checkpoint file.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device on which you want the tensors.
    """
    src_format = Checkpointer.get_format(Path(src).suffix)
    dst_format = Checkpointer.get_format(Path(dst).suffix)

    if src_format == dst_format and src_serializer == dst_serializer:
        raise ValueError("Input and output formats and serializers are the same, no conversion needed.")

    src_checkpointer = Checkpointer(format=src_format, serializer=src_serializer, serializer_args=src_serializer_args)
    dst_checkpointer = Checkpointer(format=dst_format, serializer=dst_serializer, serializer_args=dst_serializer_args)

    obj = src_checkpointer.load(src, device=device)
    dst_checkpointer.save(obj, dst)
    dst_checkpointer.flush()
