import pytest
import torch

from nnscaler.parallel import load_merged_state_dict
from nnscaler.runtime.module import ParallelModule


class MinimalParallel(ParallelModule, skip_init=True):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.zeros(4))
        self.loads = []

    def to(self, *args, **kwargs):
        return torch.nn.Module.to(self, *args, **kwargs)

    def load_merged_state_dict(self, state_dict, prefix='', strict=True):
        self.loads.append((strict, torch.get_num_threads()))
        key = prefix + 'original_weight'
        if key not in state_dict:
            if strict:
                raise RuntimeError('Missing key(s) in state_dict')
            return [key]
        self.weight.data.copy_(state_dict[key])
        return []


@pytest.mark.parametrize('once', [False, True])
def test_nested_merged_load_preserves_values_hooks_and_strictness(monkeypatch, once):
    monkeypatch.setenv('NNSCALER_MERGED_LOAD_ONCE', '1' if once else '0')
    root = torch.nn.Module()
    root.child = MinimalParallel()
    root.other = torch.nn.Parameter(torch.zeros(1))
    hooks = []
    root.child.register_load_state_dict_post_hook(
        lambda module, keys: hooks.append(module.weight.detach().clone()))
    expected = torch.arange(4, dtype=torch.float32)
    load_merged_state_dict(root, {'child.original_weight': expected, 'other': torch.ones(1)}, device='cpu')
    assert len(root.child.loads) == (1 if once else 2)
    assert torch.equal(root.child.weight, expected)
    assert torch.equal(root.other, torch.ones(1))
    assert len(hooks) == 1 and torch.equal(hooks[0], expected)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_merged_state_dict(root, {}, device='cpu')
    # Failed strict loading must not leak its context into ordinary permissive loads.
    result = root.load_state_dict({}, strict=False)
    assert 'child.original_weight' in result.missing_keys


def test_thread_setting_restored_on_success_and_failure(monkeypatch):
    before = torch.get_num_threads()
    requested = 2 if before != 2 else 1
    monkeypatch.setenv('NNSCALER_MERGED_LOAD_ONCE', '1')
    monkeypatch.setenv('NNSCALER_MERGED_LOAD_THREADS', str(requested))
    module = MinimalParallel()
    load_merged_state_dict(module, {'original_weight': torch.ones(4)}, device='cpu')
    assert module.loads == [(True, requested)]
    assert torch.get_num_threads() == before
    with pytest.raises(RuntimeError):
        load_merged_state_dict(module, {}, device='cpu')
    assert torch.get_num_threads() == before
