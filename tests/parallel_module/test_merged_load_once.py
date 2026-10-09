import pytest
import torch

from nnscaler.parallel import load_merged_state_dict
from nnscaler.runtime.module import ParallelModule, load_merged_module, merged_load_session


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
    assert module.loads == [(False, requested)]
    assert torch.get_num_threads() == before
    with pytest.raises(RuntimeError):
        load_merged_state_dict(module, {}, device='cpu')
    assert torch.get_num_threads() == before


def test_custom_parallel_load_handler_keeps_explicit_fallback(monkeypatch):
    class CustomParallel(MinimalParallel, skip_init=True):
        def _load_from_state_dict(self, *args, **kwargs):
            # A user-defined module need not run ParallelModule's handler.
            pass

    monkeypatch.setenv('NNSCALER_MERGED_LOAD_ONCE', '1')
    root = torch.nn.Module()
    root.child = CustomParallel()
    expected = torch.arange(4, dtype=torch.float32)
    load_merged_state_dict(root, {'child.original_weight': expected}, device='cpu')
    assert len(root.child.loads) == 1
    torch.testing.assert_close(root.child.weight, expected, rtol=0, atol=0)


def test_shared_parallel_module_keeps_canonical_prefix_fallback(monkeypatch):
    monkeypatch.setenv('NNSCALER_MERGED_LOAD_ONCE', '1')
    root = torch.nn.Module()
    root.first = MinimalParallel()
    root.alias = root.first
    expected = torch.arange(4, dtype=torch.float32)
    load_merged_state_dict(root, {'first.original_weight': expected}, device='cpu')
    torch.testing.assert_close(root.first.weight, expected, rtol=0, atol=0)


def test_repeated_recursive_calls_share_only_current_restore(monkeypatch):
    class RepeatedRoot(torch.nn.Module):
        def load_state_dict(self, state_dict, **kwargs):
            super().load_state_dict(state_dict, **kwargs)
            return super().load_state_dict(state_dict, **kwargs)

    monkeypatch.setenv('NNSCALER_MERGED_LOAD_ONCE', '1')
    root = RepeatedRoot()
    root.child = MinimalParallel()
    state = {'child.original_weight': torch.arange(4, dtype=torch.float32)}
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 1
    root.child.weight.data.fill_(100)
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 2
    torch.testing.assert_close(root.child.weight, state['child.original_weight'], rtol=0, atol=0)
    state['child.original_weight'].add_(3)
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 3
    torch.testing.assert_close(root.child.weight, state['child.original_weight'], rtol=0, atol=0)


@pytest.mark.parametrize('once', [False, True])
def test_shared_alias_values_and_hooks_keep_original_order(monkeypatch, once):
    monkeypatch.setenv('NNSCALER_MERGED_LOAD_ONCE', str(int(once)))
    root = torch.nn.Module()
    root.first = MinimalParallel()
    root.alias = root.first
    observed = []
    root.first.register_load_state_dict_post_hook(
        lambda m, keys: observed.append(m.weight.detach().clone()))
    load_merged_state_dict(root, {'first.original_weight': torch.ones(4),
                                  'alias.original_weight': torch.full((4,), 2.)}, device='cpu')
    assert len(root.first.loads) == 3
    assert all(torch.equal(a, torch.full((4,), b)) for a, b in zip(observed, [1., 2.]))
    assert torch.equal(root.first.weight, torch.ones(4))


def test_failed_load_does_not_mark_module_complete():
    class FailsOnce(MinimalParallel, skip_init=True):
        def load_merged_state_dict(self, *args, **kwargs):
            result = super().load_merged_state_dict(*args, **kwargs)
            if len(self.loads) == 1:
                raise RuntimeError('partial restore failed')
            return result

    module = FailsOnce()
    state = {'original_weight': torch.ones(4)}
    with merged_load_session():
        with pytest.raises(RuntimeError, match='partial restore failed'):
            load_merged_module(module, state)
        load_merged_module(module, state)
        load_merged_module(module, state)
    assert len(module.loads) == 2


def test_cached_permissive_result_still_checks_missing_keys():
    module = MinimalParallel()
    with merged_load_session():
        result = load_merged_module(module, {}, strict=False)
        result.clear()  # callers cannot mutate the cached strict-check result
        with pytest.raises(RuntimeError, match='Missing key'):
            load_merged_module(module, {}, strict=True)
    assert len(module.loads) == 1


def test_nested_restore_invalidates_parent_completion_records():
    module = MinimalParallel()
    first = {'original_weight': torch.ones(4)}
    second = {'original_weight': torch.full((4,), 2.)}
    with merged_load_session():
        load_merged_module(module, first)
        with merged_load_session():
            load_merged_module(module, second)
        load_merged_module(module, first)
    assert len(module.loads) == 3
    assert torch.equal(module.weight, torch.ones(4))


def test_hook_failure_clears_restore_session(monkeypatch):
    monkeypatch.setenv('NNSCALER_MERGED_LOAD_ONCE', '1')
    module = MinimalParallel()
    def fail(*args):
        raise RuntimeError('hook failed')
    handle = module.register_load_state_dict_post_hook(fail)
    state = {'original_weight': torch.ones(4)}
    with pytest.raises(RuntimeError, match='hook failed'):
        load_merged_state_dict(module, state, device='cpu')
    handle.remove()
    module.weight.data.zero_()
    load_merged_state_dict(module, state, device='cpu')
    assert len(module.loads) == 2
    assert torch.equal(module.weight, torch.ones(4))


def test_custom_transformed_input_does_not_hide_canonical_load(monkeypatch):
    class TransformingRoot(torch.nn.Module):
        def load_state_dict(self, state_dict, **kwargs):
            transformed = {k: v + 5 for k, v in state_dict.items()}
            return super().load_state_dict(transformed, **kwargs)

    monkeypatch.setenv('NNSCALER_MERGED_LOAD_ONCE', '1')
    root = TransformingRoot()
    root.child = MinimalParallel()
    state = {'child.original_weight': torch.ones(4)}
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 2
    assert torch.equal(root.child.weight, torch.ones(4))


def test_input_mutation_invalidates_same_session_record():
    module = MinimalParallel()
    state = {'original_weight': torch.ones(4)}
    with merged_load_session():
        load_merged_module(module, state)
        state['original_weight'].add_(3)
        load_merged_module(module, state)
    assert len(module.loads) == 2
    assert torch.equal(module.weight, torch.full((4,), 4.))


def test_unversioned_inference_input_does_not_break_loading():
    module = MinimalParallel()
    with torch.inference_mode():
        state = {'original_weight': torch.ones(4)}
    with merged_load_session():
        load_merged_module(module, state)
        load_merged_module(module, state)
    assert len(module.loads) == 2
    assert torch.equal(module.weight, torch.ones(4))
