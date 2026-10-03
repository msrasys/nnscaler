#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import dataclass
import dis
import importlib
import linecache
from pathlib import Path
import sys
from functools import lru_cache

from typing import Tuple, Optional


@lru_cache(maxsize=512)
def _instructions(code):
    # Code objects are immutable, including when a function is replaced or
    # patched. Bound the cache because tracing can generate many functions.
    return tuple(dis.get_instructions(code))


def get_instructions(back_times=1) -> Tuple[Tuple[dis.Instruction, ...], int]:
    """
    Get the instructions of the (back_times)-th frame from the bottom.

    Args:
        back_times: The number of frames to go back.
            By default (back_times=1), the instruction of the frame who call this function will be returned.

    Returns:
        A tuple of two elements:
            - A tuple of dis.Instruction objects in frame.
            - The index of the current instruction in the list.
    """
    # Avoid retaining this function's own frame in its locals, which creates
    # a reference cycle holding the caller's tensors until GC runs.
    calling_frame = sys._getframe(back_times + 1)
    insts = _instructions(calling_frame.f_code)

    if sys.version_info >= (3, 11):
        from bisect import bisect_left
        # bisect_left find the position where an element should be inserted in a sorted list to maintain the list’s order.
        # If the element already exists in the list,
        # bisect_left returns the position to the left of the first occurrence of that element.
        # here use bisect_left to find the position of calling_frame.f_lasti in the insts.
        cur = bisect_left(insts, calling_frame.f_lasti, key=lambda x: x.offset)
    else:
        # based on the assumption that most bytecodes in Python are two bytes,
        # dividing by 2 results in the sequence number of the instructions.
        cur = calling_frame.f_lasti // 2

    # From python doc:
    # EXTENDED_ARG(ext): Prefixes any opcode which has an argument too big to fit into the default one byte.
    # ext holds an additional byte which act as higher bits in the argument.
    # For each opcode, at most three prefixal EXTENDED_ARG are allowed, forming an argument from two-byte to four-byte.
    while insts[cur].opname == 'EXTENDED_ARG':
        cur += 1
    return insts, cur


def get_last_instruction(back_times=1) -> dis.Instruction:
    """
    Get the current instruction of the (back_times)-th frame from the bottom.

    Args:
        back_times: The number of frames to go back.
            By default (back_times=1), the instruction of the frame who call this function will be returned.

    Returns:
        The current instruction in that frame.
    """
    # +1 because the first frame is the frame of get_last_instruction
    insts, cur = get_instructions(back_times + 1)
    return insts[cur]


@dataclass
class FrameRecord:
    filename: str
    lineno: str
    line: str
    # the name of the frame is the function name
    name: str

    def __repr__(self) -> str:
        if self.filename:
            return f'File "{self.filename}", line {self.lineno}, in {self.name},  {self.line}'
        else:
            return ''


@lru_cache(maxsize=1)
def _ignored_frame_directories():
    cube_path = str(Path(importlib.util.find_spec('nnscaler').origin).parent) + '/'  # the cube path
    torch_path = str(Path(importlib.util.find_spec('torch').origin).parent) + '/'  # the torch path
    return cube_path, torch_path


def get_frame_record() -> Optional[FrameRecord]:
    # Only the nearest user frame is needed. Extracting the entire stack also
    # looks up source/bytecode positions for every internal tracer frame.
    frame = sys._getframe(1)
    try:
        ignore_dirs = _ignored_frame_directories()
        while frame is not None:
            filename = frame.f_code.co_filename
            if not any(p in filename for p in ignore_dirs):
                lineno = frame.f_lineno
                linecache.checkcache(filename)
                line = linecache.getline(filename, lineno, frame.f_globals).strip()
                return FrameRecord(filename, lineno, line, frame.f_code.co_name)
            frame = frame.f_back
        return None
    finally:
        del frame
