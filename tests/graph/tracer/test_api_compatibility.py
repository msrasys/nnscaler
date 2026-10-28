# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import subprocess
import sys
import warnings

import pytest
import torch

from nnscaler.graph.tracer.metadata import AutocastInfo
from nnscaler.runtime.module import CubeModule


def test_unicode_escape_is_ready_before_tuple_is_patched():
    # A fresh interpreter reproduces lazy CodecInfo initialization during tracing.
    code = '''
import codecs
import sys
import torch
def empty_search(name):
    return None
codecs.register(empty_search)
codecs.unregister(empty_search)
sys.modules.pop('encodings.unicode_escape', None)
import nnscaler.graph.parser.external
import builtins
original = builtins.tuple
builtins.tuple = lambda *args: original(*args)
try:
    codec = codecs.lookup('unicode_escape')
finally:
    builtins.tuple = original
assert codec.decode(b'\\\\u03bb')[0] == '\u03bb'
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_autocast_metadata_uses_device_aware_queries():
    if torch.__version__ < (2, 4, 0):
        pytest.skip('device-aware autocast API requires PyTorch 2.4+')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            info = AutocastInfo.from_context()
        assert info.cpu_enabled and info.cpu_dtype == torch.bfloat16
        assert not info.cuda_enabled
        assert not [w for w in emitted if 'deprecated' in str(w.message)]


def test_autocast_metadata_keeps_legacy_query_compatibility(monkeypatch):
    monkeypatch.setattr(torch, '__version__', torch.torch_version.TorchVersion('2.3.0'))
    monkeypatch.setattr(torch, 'is_autocast_cpu_enabled', lambda: True)
    monkeypatch.setattr(torch, 'get_autocast_cpu_dtype', lambda: torch.bfloat16)
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda: False)
    monkeypatch.setattr(torch, 'get_autocast_gpu_dtype', lambda: torch.float16)
    info = AutocastInfo.from_context()
    assert info.cpu_enabled and info.cpu_dtype == torch.bfloat16
    assert not info.cuda_enabled and info.cuda_dtype == torch.float16


@pytest.mark.parametrize('non_blocking', [False, True])
def test_module_to_warns_only_for_requested_non_blocking(monkeypatch, non_blocking):
    module = CubeModule()
    monkeypatch.setattr(module, 'cpu', lambda: module)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        assert module.to('cpu', non_blocking=non_blocking) is module
    messages = [w for w in emitted if 'blocking approach' in str(w.message)]
    assert len(messages) == int(non_blocking)
