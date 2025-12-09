#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Any, Callable, Protocol
from pathlib import Path

import torch

from nnscaler.runtime.serialization import load, save


class _LoadProc(Protocol):
    def __call__(self, f: str | Path, *, device='cpu') -> Any: ...


class _SaveProc(Protocol):
    def __call__(self, obj: Any, f: str | Path) -> None: ...


class Checkpointer:
    # the format of the checkpoint file
    # keys: epoch, step, rank
    # currently it is not configurable
    # TODO: make it configurable
    CHECKPOINT_FILE_FORMAT: str = '{epoch:04d}-{step:04d}/{rank}{suffix}'
    CHECKPOINT_LAST_DIR_NAME: str = 'last'
    CHECKPOINT_BEST_DIR_NAME: str = 'best'
    CHECKPOINT_MERGED_FILE_NAME: str = 'merged{suffix}'
    CHECKPOINT_LAST_FILE_FORMAT: str = 'last/{rank}{suffix}'
    CHECKPOINT_BEST_FILE_FORMAT: str = 'best/{rank}{suffix}'
    SUFFIX_MAP: dict[str, str] = {
        'pt': '.ckpt',
        'safetensors': '.safetensors'
    }
    # will use torch.load and torch.save for other suffixes
    SUFFIX_HANDLERS: dict[str, tuple[_LoadProc, _SaveProc]] = {
        '.safetensors': (load, save),
    }

    def __init__(self, format: str = 'pt'):
        if format not in self.SUFFIX_MAP:
            raise ValueError(f"Unsupported checkpoint format: {format}")
        self.format = format
        self.suffix = self.SUFFIX_MAP[format]

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

    @classmethod
    def load(cls, f: str | Path, *, device='cpu') -> Any:
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
        if suffix in cls.SUFFIX_HANDLERS:
            load_func, _ = cls.SUFFIX_HANDLERS[suffix]
            return load_func(f, device=device)
        else:
            return torch.load(f, map_location=device, weights_only=False)

    @classmethod
    def save(cls, obj: Any, f: str | Path) -> None:
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
        if suffix in cls.SUFFIX_HANDLERS:
            _, save_func = cls.SUFFIX_HANDLERS[suffix]
            save_func(obj, f)
        else:
            torch.save(obj, f)

    @classmethod
    def load_for_rank(cls, dir: str | Path, rank: int, device='cpu') -> Any:
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
        for suffix in cls.SUFFIX_MAP.values():
            f = Path(dir) / f"{rank}{suffix}"
            if f.exists():
                return cls.load(f, device=device)
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
        f = Path(dir) / f"{rank}{self.suffix}"
        self.save(obj, f)

    @classmethod
    def remove_for_rank(cls, dir: str | Path, rank: int) -> None:
        """
        Removes a checkpoint file for a specific rank

        Args:
            dir (`str`):
                The directory where the checkpoint files are stored.
            rank (`int`):
                The rank of the checkpoint file to remove.
        """
        for suffix in cls.SUFFIX_MAP.values():
            f = Path(dir) / f"{rank}{suffix}"
            if f.exists():
                f.unlink()

    @classmethod
    def list_checkpoints(cls, dir: str | Path) -> list[Path]:
        """
        List the checkpoint files in a directory
        Args:
            dir (`str`):
                The directory where the checkpoint files are stored.
        Returns:
            (`list[Path]`):
                The list of checkpoint files in the directory.
        """
        p = Path(dir)
        files = []
        for suffix in cls.SUFFIX_MAP.values():
            fs = list(p.glob(f"*{suffix}"))
            if fs:
                if files:
                    raise ValueError(f"Mixed checkpoint file formats in directory {dir}")
                else:
                    files.extend(fs)
        return files


def register_format(
        name: str,
        suffix: str,
        save_func: _SaveProc,
        load_func: _LoadProc,
    ) -> None:
    """
    Registers a new serialization format.
    Args:
        name (`str`):
            The name of the format.
        suffix (`str`):
            The file suffix of the format.
        load_func:
            The function to load the format.
        save_func:
            The function to save the format.
    """
    suffix = '.' + suffix.lstrip('.')
    Checkpointer.SUFFIX_MAP[name] = suffix
    Checkpointer.SUFFIX_HANDLERS[suffix] = (load_func, save_func)
