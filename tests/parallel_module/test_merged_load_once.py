import pytest
import torch
from pathlib import Path
from functools import wraps
from types import SimpleNamespace

from nnscaler.parallel import load_merged_state_dict, _merged_load_cpu_threads, _checkpoint_load_num_threads
from nnscaler.runtime.module import AttrMeta, ParallelModule, load_merged_module, merged_load_session


class MinimalParallel(ParallelModule, skip_init=True):
    rank = 0
    world_size = 1
    compute_config = SimpleNamespace(use_zero=0)
    dist_param_map = {'weight': 'original_weight'}
    attr_meta_maps = [{'weight': AttrMeta(
        tid=0, is_param=True, orig_name='original_weight', shape=(4,),
        slicers=(slice(None),), val_chunks=1, dtype=torch.float32, sub_shape=(4,))}]

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.zeros(4))
        self._fullmap = self.attr_meta_maps[0]
        self._non_presistent_buffers_inited = True
        self.loads = []

    def to(self, *args, **kwargs):
        return torch.nn.Module.to(self, *args, **kwargs)


@pytest.fixture(autouse=True)
def record_default_loader_calls(monkeypatch):
    # Exercise the real default loader. Overriding it in MinimalParallel would
    # test a custom loader, which must retain the original two-pass behavior.
    original = ParallelModule.load_merged_state_dict

    @wraps(original)
    def record(self, state_dict, prefix='', strict=True):
        self.loads.append((strict, torch.get_num_threads()))
        return original(self, state_dict, prefix, strict)

    monkeypatch.setattr(ParallelModule, 'load_merged_state_dict', record)


def test_standard_load_deduplicates_only_within_one_restore():
    root = torch.nn.Module()
    root.child = MinimalParallel()
    state = {'child.original_weight': torch.arange(4, dtype=torch.float32)}
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 1
    root.child.weight.data.fill_(20)
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 2
    torch.testing.assert_close(root.child.weight, state['child.original_weight'], rtol=0, atol=0)


@pytest.mark.parametrize('override', ['subclass', 'instance'])
def test_custom_merged_loader_keeps_strict_validation(override):
    from types import MethodType

    calls = []

    def custom_load(self, state_dict, prefix='', strict=True):
        calls.append(strict)
        if strict and not torch.isfinite(state_dict[prefix + 'original_weight']).all():
            raise ValueError('custom strict validation')
        return MinimalParallel.load_merged_state_dict(self, state_dict, prefix, strict)

    class CustomParallel(MinimalParallel, skip_init=True):
        load_merged_state_dict = custom_load

    root = torch.nn.Module()
    root.child = CustomParallel() if override == 'subclass' else MinimalParallel()
    if override == 'instance':
        root.child.load_merged_state_dict = MethodType(custom_load, root.child)

    with pytest.raises(ValueError, match='custom strict validation'):
        load_merged_state_dict(root, {'child.original_weight': torch.full((4,), float('nan'))}, device='cpu')
    assert calls == [False, True]


@pytest.mark.parametrize('bound_hook', [False, True])
def test_builtin_pre_load_hook_keeps_deduplication(monkeypatch, bound_hook):
    root = torch.nn.Module()
    root.child = MinimalParallel()
    monkeypatch.setattr(root.child, '_warn_uninitialized_non_persistent_buffers', lambda: None)
    if bound_hook:
        root.child._register_load_state_dict_pre_hook(root.child._pre_load_state_dict_hook)
    else:
        # Match ParallelModule._post_init's actual registration form.
        root.child._register_load_state_dict_pre_hook(
            ParallelModule._pre_load_state_dict_hook, with_module=True)
    state = {'child.original_weight': torch.arange(4, dtype=torch.float32)}

    load_merged_state_dict(root, state, device='cpu')

    assert len(root.child.loads) == 1
    torch.testing.assert_close(root.child.weight, state['child.original_weight'], rtol=0, atol=0)


def test_custom_post_load_hook_preserves_values_hooks_and_strictness():
    root = torch.nn.Module()
    root.child = MinimalParallel()
    root.other = torch.nn.Parameter(torch.zeros(1))
    hooks = []
    root.child.register_load_state_dict_post_hook(
        lambda module, keys: hooks.append(module.weight.detach().clone()))
    expected = torch.arange(4, dtype=torch.float32)
    load_merged_state_dict(root, {'child.original_weight': expected, 'other': torch.ones(1)}, device='cpu')
    assert len(root.child.loads) == 2
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
    requested = 2
    monkeypatch.setattr('nnscaler.parallel._checkpoint_load_num_threads', lambda: requested)
    module = MinimalParallel()
    load_merged_state_dict(module, {'original_weight': torch.ones(4)}, device='cpu')
    assert module.loads == [(False, requested)]
    assert torch.get_num_threads() == before
    with pytest.raises(RuntimeError):
        load_merged_state_dict(module, {}, device='cpu')
    assert torch.get_num_threads() == before


@pytest.mark.parametrize('cpus,ranks,expected', [(96, 6, 8), (12, 6, 2), (2, 8, 1)])
def test_restore_threads_respect_per_rank_budget_and_nested_calls(monkeypatch, cpus, ranks, expected):
    def no_quota(*args, **kwargs):
        raise FileNotFoundError
    monkeypatch.setattr(Path, 'read_text', no_quota)
    monkeypatch.setattr('os.sched_getaffinity', lambda _: set(range(cpus)))
    monkeypatch.setenv('LOCAL_WORLD_SIZE', str(ranks))
    previous = torch.get_num_threads()
    with _merged_load_cpu_threads():
        assert torch.get_num_threads() == expected
        with _merged_load_cpu_threads():
            assert torch.get_num_threads() == expected
        assert torch.get_num_threads() == expected
    assert torch.get_num_threads() == previous


@pytest.mark.parametrize('quotas,local_ranks,expected', [
    ({}, 6, 8),
    ({'/sys/fs/cgroup/cpu.max': '400000 100000'}, 2, 2),
    ({'/sys/fs/cgroup/pod/cpu.max': '1200000 100000',
      '/sys/fs/cgroup/pod/worker/cpu.max': 'max 100000'}, 6, 2),
    ({'/sys/fs/cgroup/pod/cpu.max': '1200000 100000',
      '/sys/fs/cgroup/pod/worker/cpu.max': '350000 100000'}, 2, 1),
    ({'/sys/fs/cgroup/cpu.max': '50000 100000'}, 8, 1),
])
def test_restore_threads_respect_container_and_parent_quotas(monkeypatch, quotas, local_ranks, expected):
    files = {'/proc/self/cgroup': '0::/pod/worker\n', **quotas}
    def read(path, *args, **kwargs):
        try:
            return files[str(path)]
        except KeyError:
            raise FileNotFoundError(path)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr('os.sched_getaffinity', lambda _: set(range(96)))
    monkeypatch.setenv('LOCAL_WORLD_SIZE', str(local_ranks))
    assert _checkpoint_load_num_threads() == expected


def test_custom_parallel_load_handler_keeps_explicit_fallback(monkeypatch):
    class CustomParallel(MinimalParallel, skip_init=True):
        def _load_from_state_dict(self, *args, **kwargs):
            # A user-defined module need not run ParallelModule's handler.
            pass

    root = torch.nn.Module()
    root.child = CustomParallel()
    expected = torch.arange(4, dtype=torch.float32)
    load_merged_state_dict(root, {'child.original_weight': expected}, device='cpu')
    assert len(root.child.loads) == 1
    torch.testing.assert_close(root.child.weight, expected, rtol=0, atol=0)


def test_shared_parallel_module_keeps_canonical_prefix_fallback(monkeypatch):
    root = torch.nn.Module()
    root.first = MinimalParallel()
    root.alias = root.first
    expected = torch.arange(4, dtype=torch.float32)
    load_merged_state_dict(root, {'first.original_weight': expected}, device='cpu')
    torch.testing.assert_close(root.first.weight, expected, rtol=0, atol=0)


def test_custom_repeated_loader_preserves_original_passes():
    class RepeatedRoot(torch.nn.Module):
        def load_state_dict(self, state_dict, **kwargs):
            super().load_state_dict(state_dict, **kwargs)
            return super().load_state_dict(state_dict, **kwargs)

    root = RepeatedRoot()
    root.child = MinimalParallel()
    state = {'child.original_weight': torch.arange(4, dtype=torch.float32)}
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 3
    root.child.weight.data.fill_(100)
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 6
    torch.testing.assert_close(root.child.weight, state['child.original_weight'], rtol=0, atol=0)
    state['child.original_weight'].add_(3)
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 9
    torch.testing.assert_close(root.child.weight, state['child.original_weight'], rtol=0, atol=0)


def test_shared_alias_values_and_hooks_keep_original_order(monkeypatch):
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

    root = TransformingRoot()
    root.child = MinimalParallel()
    state = {'child.original_weight': torch.ones(4)}
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 2
    assert torch.equal(root.child.weight, torch.ones(4))


@pytest.mark.parametrize('mutation', [
    'root_loader', 'instance_loader', 'child_loader', 'root_post_hook',
    'child_post_hook', 'sibling_pre_hook',
])
def test_custom_destination_mutation_keeps_original_final_weights(mutation):
    class MutatingRoot(torch.nn.Module):
        def load_state_dict(self, state_dict, **kwargs):
            result = torch.nn.Module.load_state_dict(self, state_dict, **kwargs)
            self.child.weight.data.fill_(20)
            return result

    root = MutatingRoot() if mutation == 'root_loader' else torch.nn.Module()
    root.child = MinimalParallel()
    root.sibling = MinimalParallel()
    def mutate(*args):
        root.child.weight.data.fill_(20)
    if mutation == 'instance_loader':
        from types import MethodType
        root.load_state_dict = MethodType(MutatingRoot.load_state_dict, root)
    elif mutation == 'child_loader':
        from types import MethodType
        original = root.child._load_from_state_dict
        def load_then_mutate(self, *args, **kwargs):
            original(*args, **kwargs)
            self.weight.data.fill_(20)
        root.child._load_from_state_dict = MethodType(load_then_mutate, root.child)
    elif mutation == 'root_post_hook':
        root.register_load_state_dict_post_hook(mutate)
    elif mutation == 'child_post_hook':
        root.child.register_load_state_dict_post_hook(mutate)
    elif mutation == 'sibling_pre_hook':
        root.sibling.register_load_state_dict_pre_hook(mutate)

    state = {'child.original_weight': torch.ones(4), 'sibling.original_weight': torch.full((4,), 2.)}
    load_merged_state_dict(root, state, device='cpu')
    assert len(root.child.loads) == 2
    torch.testing.assert_close(root.child.weight, torch.ones(4), rtol=0, atol=0)
    torch.testing.assert_close(root.sibling.weight, torch.full((4,), 2.), rtol=0, atol=0)


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
