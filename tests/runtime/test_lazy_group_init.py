import torch
import pytest
from nnscaler.runtime import device

@pytest.mark.parametrize('eager',[True,False])
def test_default_group_binding_is_explicitly_configurable(monkeypatch,eager):
    calls=[]
    monkeypatch.setenv('NNSCALER_EAGER_GROUP_INIT','1' if eager else '0')
    for k,v in {'LOCAL_RANK':'1','LOCAL_WORLD_SIZE':'2','GROUP_RANK':'0'}.items():monkeypatch.setenv(k,v)
    monkeypatch.setattr(device,'is_running_distributed',lambda:True)
    monkeypatch.setattr(device.CompileFlag,'dev_mode',False)
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.distributed,'init_process_group',lambda **kw:calls.append(kw))
    monkeypatch.setattr(torch.distributed,'get_rank',lambda:1)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:2)
    monkeypatch.setattr(torch.cuda,'set_device',lambda _:None)
    monkeypatch.setattr(torch.cuda,'default_stream',lambda:object())
    result=device._DeviceGroup()
    assert result.rank==1 and result.local_rank==1
    assert len(calls)==1 and calls[0]['backend']=='nccl'
    assert ('device_id' in calls[0])==eager
    if eager:assert calls[0]['device_id']==torch.device('cuda',1)
