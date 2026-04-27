#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import pytest
import torch

import nnscaler
from nnscaler.cli.trainer_args import (
    load_type, ComputeConfig, OptionalComputeConfig, TrainerArgs,
    _resolve_precision, _to_precision, _get_tensor_dtype,
)
from nnscaler.runtime.utils import set_grad_dtype, get_grad_dtype


def test_load_type():
    assert load_type(int) == int
    assert load_type('int') == int
    assert load_type(int.to_bytes) == int.to_bytes
    assert load_type('int.to_bytes') == int.to_bytes
    assert load_type('nnscaler.cli.trainer_args.TrainerArgs') == nnscaler.cli.trainer_args.TrainerArgs
    assert load_type('nnscaler.cli.trainer_args.TrainerArgs.from_cli') == nnscaler.cli.trainer_args.TrainerArgs.from_cli

    with pytest.raises(RuntimeError):
        load_type('not_exist_name')

    with pytest.raises(RuntimeError):
        load_type('not_exist_namespace.not_exist_name')

    with pytest.raises(RuntimeError):
        load_type('nnscaler.not_exist_name')

    with pytest.raises(RuntimeError):
        load_type('nnscaler.cli.trainer_args.TrainerArgs.not_exist_name')


def test_compute_config_merge():
    cc = ComputeConfig(1, 2, constant_folding=True, use_end2end=True, use_zero=True)
    occ = OptionalComputeConfig(constant_folding=False, use_zero=False)
    rcc = occ.resolve(cc)
    assert rcc == ComputeConfig(1, 2, constant_folding=False, use_end2end=False)

    occ2 = OptionalComputeConfig(zero_ngroups=-1)
    with pytest.raises(ValueError):
        occ2.resolve(cc)


def test_arg_merge_resolve():
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    args = TrainerArgs.from_cli(['-f', config_path,
        '--vars.dim', '22',
        '--vars.hello', '$(compute_config.plan_ngpus)',
        '--global_batch_size!',
        '--vars.vr_0', '$$(vars.dim)',
        '--vars.vr_1', '$$x',
        '--vars.vr_3', '$$!(vars.dim)',
        '--vars.vr_4', '$$!y',
    ])
    assert args.vars['vr_0'] == '$(vars.dim)'
    assert args.vars['vr_1'] == '$$x'
    assert args.vars['vr_3'] == '$$!(vars.dim)'
    assert args.vars['vr_4'] == '$$!y'
    assert args.vars['dim'] == 22
    assert args.dataset.train_args['dim'] == 22
    assert args.dataset.val_args['dim'] == 22
    assert args.vars['hello'] == args.compute_config.plan_ngpus



def gen_instance_name(stem):
    return f'instance_{stem}'


def test_dyn_str_config():
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    args = TrainerArgs.from_cli(['-f', config_path,
        '--instance_name.__type', 'tests.cli.test_train_args.gen_instance_name',
        '--instance_name.stem', 'p$(compute_config.plan_ngpus)',
        '--compute_config.plan_ngpus', '1',
        '--global_batch_size!',
    ])
    assert args.instance_name == 'instance_p1'


# --- grad dtype tests ---

def test_resolve_precision_default_grad_none():
    """Default precision 'none' sets grad to None dtype."""
    p = _resolve_precision('none')
    assert p['grad'] == 'none'
    assert _get_tensor_dtype(p, 'grad') is None


def test_resolve_precision_string_sets_grad():
    """String precision 'bf16' applies to all tensor types except grad."""
    p = _resolve_precision('bf16')
    assert p['grad'] == 'none'
    assert all(p[t] == 'bf16' for t in p if t != 'grad')
    assert _get_tensor_dtype(p, 'grad') is None


@pytest.mark.skipif(torch.__version__ < (2, 10), reason='requires torch >= 2.10 for grad_dtype support')
def test_resolve_precision_dict_with_grad():
    """Dict precision with explicit grad key."""
    p = _resolve_precision({'param': 'bf16', 'buffer': 'bf16', 'input': 'bf16', 'grad': 'fp32'})
    assert _get_tensor_dtype(p, 'grad') == torch.float32
    assert _get_tensor_dtype(p, 'param') == torch.bfloat16


def test_resolve_precision_dict_missing_grad():
    """Dict precision without grad key defaults to 'none'."""
    p = _resolve_precision({'param': 'bf16', 'buffer': 'bf16'})
    assert p['grad'] == 'none'
    assert _get_tensor_dtype(p, 'grad') is None

    assert p['input'] == 'none'
    assert _get_tensor_dtype(p, 'input') is None


def test_set_get_grad_dtype_on_param():
    """set_grad_dtype/get_grad_dtype on a single parameter."""
    p = torch.nn.Parameter(torch.randn(4, dtype=torch.bfloat16))
    assert get_grad_dtype(p) == torch.bfloat16  # default: same as param dtype
    if torch.__version__ < (2, 10):
        with pytest.raises(RuntimeError):
            set_grad_dtype(p, torch.float32)  # grad dtype must be same as param dtype prior to 2.10
    else:
        set_grad_dtype(p, torch.float32)
        assert get_grad_dtype(p) == torch.float32


def test_set_grad_dtype_none_is_noop():
    """set_grad_dtype with None should be a no-op."""
    p = torch.nn.Parameter(torch.randn(4, dtype=torch.bfloat16))
    set_grad_dtype(p, None)
    assert get_grad_dtype(p) == torch.bfloat16


def test_set_grad_dtype_on_module():
    """set_grad_dtype on a module sets grad_dtype for all trainable parameters."""
    m = torch.nn.Linear(4, 4).to(torch.bfloat16)
    if torch.__version__ < (2, 10):
        with pytest.raises(RuntimeError):
            set_grad_dtype(m, torch.float32)  # grad dtype must be same as param dtype prior to 2.10
    else:
        set_grad_dtype(m, torch.float32)
        for p in m.parameters():
            if p.requires_grad:
                assert get_grad_dtype(p) == torch.float32


@pytest.mark.skipif(torch.__version__ < (2, 10), reason="grad dtype must be same as param dtype prior to PyTorch 2.10")
def test_to_precision_sets_grad_dtype():
    """_to_precision with grad precision sets grad_dtype on module parameters."""
    m = torch.nn.Linear(4, 4)
    precision = _resolve_precision({'param': 'bf16', 'buffer': 'bf16', 'input': 'bf16', 'grad': 'fp32'})
    m = _to_precision(m, precision)
    for p in m.parameters():
        if p.requires_grad:
            assert p.dtype == torch.bfloat16
            assert get_grad_dtype(p) == torch.float32


def test_to_precision_no_grad_precision():
    """_to_precision without grad precision leaves grad_dtype as param dtype."""
    m = torch.nn.Linear(4, 4)
    precision = _resolve_precision({'param': 'bf16', 'buffer': 'bf16', 'input': 'bf16'})
    m = _to_precision(m, precision)
    for p in m.parameters():
        if p.requires_grad:
            assert p.dtype == torch.bfloat16
            assert get_grad_dtype(p) == torch.bfloat16


@pytest.mark.skipif(torch.__version__ < (2, 10), reason="grad dtype must be same as param dtype prior to PyTorch 2.10")
def test_trainer_args_precision_with_grad():
    """TrainerArgs with precision dict containing grad key."""
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    args = TrainerArgs.from_cli([
        '-f', config_path,
        '--precision.param', 'bf16',
        '--precision.buffer', 'bf16',
        '--precision.input', 'bf16',
        '--precision.grad', 'fp32',
        '--global_batch_size!',
    ])
    assert args.grad_dtype == torch.float32
    assert args.param_dtype == torch.bfloat16
    assert args.buffer_dtype == torch.bfloat16
