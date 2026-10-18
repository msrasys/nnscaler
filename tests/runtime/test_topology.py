# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
import os
from datetime import timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import torch
import torch.distributed as dist

from nnscaler.runtime import topology
from ..launch_torchrun import launch_torchrun

RAW = b'''<system version="1"><cpu host_hash="42"><pci busid="0000:1a:00.0">
<gpu rank="7" keep="1" dev="0" sm="100"><nvlink target="0000:1b:00.0" count="18"/></gpu>
</pci></cpu></system>'''


def test_clean_removes_membership_but_preserves_hardware():
    root = ET.fromstring(topology._clean_xml(RAW))
    assert all('rank' not in node.attrib and 'keep' not in node.attrib for node in root.iter())
    gpu = root.find('.//gpu')
    assert gpu.attrib == {'dev': '0', 'sm': '100'}
    assert gpu.find('nvlink').attrib == {'target': '0000:1b:00.0', 'count': '18'}


def test_network_plugin_properties_are_rediscovered():
    data = RAW.replace(b'dev="0"', b'gdr="1" dev="0"').replace(
        b'</pci>', b'<nic><net name="mlx5_0" gdr="1" coll="1" speed="400000"/></nic></pci>')
    root = ET.fromstring(topology._clean_xml(data))
    assert root.find('.//gpu').get('gdr') is None
    assert root.find('.//net') is None
    assert root.find('.//pci').get('busid') == '0000:1a:00.0'


@pytest.mark.parametrize('name', ['NCCL_TOPO_FILE', 'NCCL_TOPO_DUMP_FILE'])
def test_configuration_topology_overrides_are_respected(monkeypatch, name):
    monkeypatch.delenv('NCCL_TOPO_FILE', raising=False)
    monkeypatch.delenv('NCCL_TOPO_DUMP_FILE', raising=False)
    monkeypatch.setattr(topology, '_configuration', lambda: {'/etc/nccl.conf': f'{name}=/user/topology.xml\n'})
    assert topology._has_topology_override()
    monkeypatch.setattr(topology, '_configuration', lambda: {'/etc/nccl.conf': f'# {name}=disabled\nNCCL_DEBUG=WARN'})
    assert not topology._has_topology_override()


@pytest.mark.parametrize('data', [b'broken', b'<system version="1"/>',
    RAW.replace(b'version="1"', b'version="2"'),
    RAW.replace(b'</system>', b'<cpu host_hash="99"/></system>')])
def test_unusable_topology_is_rejected(data):
    with pytest.raises((ET.ParseError, ValueError)):
        topology._clean_xml(data)


def test_cache_requires_complete_matching_files(tmp_path):
    path = tmp_path / 'topology.xml'
    data = topology._clean_xml(RAW)
    assert not topology._valid_cache(path)
    topology._atomic_write(path, data)
    assert not topology._valid_cache(path)
    topology._atomic_write(path.with_suffix('.sha256'), hashlib.sha256(data).hexdigest().encode())
    assert topology._valid_cache(path)
    path.write_bytes(data[:-8])
    assert not topology._valid_cache(path)
    path.write_bytes(RAW)
    path.with_suffix('.sha256').write_text(hashlib.sha256(RAW).hexdigest())
    assert not topology._valid_cache(path)


def test_node_plan_reuses_cache_and_allocates_unique_dump_paths(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    monkeypatch.setattr(topology, '_fingerprint', lambda size: f'node-{size}')
    first = topology._node_plan(8)
    assert first['ready'] and not first['cached']
    path = Path(first['path'])
    data = topology._clean_xml(RAW)
    topology._atomic_write(path, data)
    topology._atomic_write(path.with_suffix('.sha256'), hashlib.sha256(data).hexdigest().encode())
    second = topology._node_plan(8)
    assert second['cached'] and second['path'] == first['path']
    assert second['raw'] != first['raw']
    assert not topology._node_plan(4)['cached']


def test_inventory_failure_disables_cache(monkeypatch):
    def unavailable(size):
        raise OSError('missing GPU inventory')
    monkeypatch.setattr(topology, '_fingerprint', unavailable)
    assert topology._node_plan(8) == {'ready': False, 'cached': False}


@pytest.mark.parametrize('override', ['NCCL_TOPO_FILE', 'NCCL_TOPO_DUMP_FILE'])
def test_user_override_bypasses_cache(monkeypatch, override):
    monkeypatch.setattr(topology.dist, 'is_initialized', lambda: False)
    monkeypatch.setattr(topology.dist, 'is_nccl_available', lambda: True)
    monkeypatch.setattr(topology, '_managed_file', None)
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '1')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv(override, '/user/topology.xml')
    calls = []
    monkeypatch.setattr(topology.dist, 'init_process_group', lambda **kw: calls.append(kw))
    topology.init_process_group('nccl', device_id=torch.device('cuda', 0),
                                store=dist.HashStore(), rank=0, world_size=1)
    assert len(calls) == 1 and calls[0]['backend'] == 'nccl'


def test_unbound_initialization_is_unchanged(monkeypatch):
    monkeypatch.setattr(topology.dist, 'is_initialized', lambda: False)
    calls = []
    monkeypatch.setattr(topology.dist, 'init_process_group', lambda **kw: calls.append(kw))
    topology.init_process_group('nccl')
    assert calls == [{'backend': 'nccl'}]


def _topology_cache_worker(directory):
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    root = Path(directory)
    # Two logical nodes exercise uniform dump decisions when only one misses.
    os.environ['LOCAL_WORLD_SIZE'] = '2'
    os.environ['LOCAL_RANK'] = str(rank % 2)
    os.environ['XDG_CACHE_HOME'] = str(root / f'node-{rank // 2}')
    os.environ.pop('NCCL_TOPO_FILE', None)
    os.environ.pop('NCCL_TOPO_DUMP_FILE', None)
    seen = []
    native_init = dist.init_process_group

    def capture(**kwargs):
        seen.append((os.environ.get('NCCL_TOPO_DUMP_FILE'), os.environ.get('NCCL_TOPO_FILE')))
        return native_init(**kwargs)

    dist.init_process_group = capture
    try:
        for generation in range(4):
            if generation == 1 and rank == 2:
                for path in (root / 'node-1').rglob('*'):
                    if path.is_file():
                        path.unlink()
            if generation == 3 and rank == 0:
                manual = root / 'manual.xml'
                manual.write_bytes(Path(os.environ['NCCL_TOPO_FILE']).read_bytes())
                os.environ['NCCL_TOPO_FILE'] = str(manual)
            topology.init_process_group('nccl', rank=rank, world_size=4,
                init_method=f'file://{root / str(generation)}', device_id=device,
                timeout=timedelta(seconds=60))
            assert bool(seen[-1][0]) == (generation < 2), seen
            assert bool(seen[-1][1]) == (generation == 2 or (generation == 3 and rank == 0)), seen
            if generation < 3:
                path = Path(os.environ['NCCL_TOPO_FILE'])
                assert topology._valid_cache(path)
            elif rank == 0:
                assert os.environ['NCCL_TOPO_FILE'] == str(root / 'manual.xml')
            assert 'NCCL_TOPO_DUMP_FILE' not in os.environ
            value = torch.tensor([rank + 1.0], device=device)
            dist.all_reduce(value)
            assert value.item() == 10.0
            dist.barrier()
            dist.destroy_process_group()
    finally:
        dist.init_process_group = native_init


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_distributed_cache_miss_mixed_hit_and_hit(tmp_path):
    launch_torchrun(4, _topology_cache_worker, str(tmp_path))
