"""Group initialization chooses its implementation from the active world."""
import pytest
import torch

from nnscaler.runtime import device, group_init


@pytest.mark.parametrize('already_initialized', [False, True])
@pytest.mark.parametrize('supported', [False, True])
def test_group_implementation_is_selected_automatically(monkeypatch, already_initialized, supported):
    calls = []
    for key, value in {'LOCAL_RANK': '1', 'LOCAL_WORLD_SIZE': '2', 'GROUP_RANK': '0'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(device, 'is_running_distributed', lambda: True)
    monkeypatch.setattr(device.CompileFlag, 'dev_mode', False)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: already_initialized)
    monkeypatch.setattr(torch.distributed, 'init_process_group', lambda **kw: calls.append(kw))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'default_stream', lambda: object())
    monkeypatch.setattr(group_init, 'supports_local_groups', lambda: supported)
    created = []
    monkeypatch.setattr(group_init, 'new_local_group', lambda *a, **k: created.append('local'))
    monkeypatch.setattr(torch.distributed, 'new_group', lambda *a, **k: created.append('public'))

    result = device._DeviceGroup()
    assert result.rank == 1 and result.local_rank == 1
    assert len(calls) == (0 if already_initialized else 1)
    if calls:
        assert calls[0]['backend'] == 'nccl' and 'device_id' not in calls[0]
    result.get_group([1])
    result.get_group([1])
    assert created == ['local' if supported else 'public']
