import pytest
import torch

from nnscaler.runtime import device, group_init


@pytest.mark.parametrize('local_init', [False, True])
@pytest.mark.parametrize('retired_lazy_switch', [False, True])
def test_default_group_binding_depends_only_on_group_mode(
    monkeypatch, local_init, retired_lazy_switch,
):
    calls, support_checks = [], []
    monkeypatch.setenv('NNSCALER_LOCAL_GROUP_INIT', '1' if local_init else '0')
    if retired_lazy_switch:
        monkeypatch.setenv('NNSCALER_EAGER_GROUP_INIT', '0')
    else:
        monkeypatch.delenv('NNSCALER_EAGER_GROUP_INIT', raising=False)
    for key, value in {'LOCAL_RANK': '1', 'LOCAL_WORLD_SIZE': '2', 'GROUP_RANK': '0'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(device, 'is_running_distributed', lambda: True)
    monkeypatch.setattr(device.CompileFlag, 'dev_mode', False)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    monkeypatch.setattr(torch.distributed, 'init_process_group', lambda **kw: calls.append(kw))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'default_stream', lambda: object())
    monkeypatch.setattr(group_init, 'check_local_group_support', lambda: support_checks.append(True))

    result = device._DeviceGroup()
    assert result.rank == 1 and result.local_rank == 1
    assert len(calls) == 1 and calls[0]['backend'] == 'nccl'
    assert ('device_id' in calls[0]) == (not local_init)
    assert support_checks == ([True] if local_init else [])
    if not local_init:
        assert calls[0]['device_id'] == torch.device('cuda', 1)
