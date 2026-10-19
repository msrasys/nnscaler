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
    assert first['available'] and not first['cache_hit']
    path = Path(first['cache_path'])
    data = topology._clean_xml(RAW)
    topology._atomic_write(path, data)
    topology._atomic_write(path.with_suffix('.sha256'), hashlib.sha256(data).hexdigest().encode())
    second = topology._node_plan(8)
    assert second['cache_hit'] and second['cache_path'] == first['cache_path']
    assert second['dump_path'] != first['dump_path']
    assert not topology._node_plan(4)['cache_hit']


def test_inventory_failure_disables_cache(monkeypatch):
    def unavailable(size):
        raise OSError('missing GPU inventory')
    monkeypatch.setattr(topology, '_fingerprint', unavailable)
    assert topology._node_plan(8) == {'available': False, 'cache_hit': False}


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


@pytest.fixture
def cache_init(monkeypatch, tmp_path):
    monkeypatch.setattr(dist, 'is_initialized', lambda: False)
    monkeypatch.setattr(dist, 'is_nccl_available', lambda: True)
    monkeypatch.setattr(topology, '_managed_file', None)
    monkeypatch.setattr(topology, '_configuration', lambda: {})
    monkeypatch.setattr(topology, '_fingerprint', lambda size: f'node-{size}')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '1')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    for name in (topology._INPUT, topology._DUMP, topology._DUMP_RANK):
        # Restore the original environment even if initialization adds a key.
        monkeypatch.setenv(name, '')
        monkeypatch.delenv(name)
    return dict(backend='nccl', device_id=0, store=dist.HashStore(), rank=0, world_size=1)


@pytest.mark.parametrize('cached', [False, True])
def test_cache_surrounds_initialization(monkeypatch, cache_init, cached):
    path = Path(topology._node_plan(1)['cache_path'])
    if cached:
        data = topology._clean_xml(RAW)
        topology._atomic_write(path, data)
        topology._atomic_write(path.with_suffix('.sha256'), hashlib.sha256(data).hexdigest().encode())
    calls, dumps = [], []
    result = object()

    def initialize(**kwargs):
        calls.append(kwargs)
        assert kwargs == {**cache_init, 'device_id': torch.device('cuda', 0)}
        if cached:
            assert os.environ[topology._INPUT] == str(path)
            assert topology._DUMP not in os.environ
        else:
            assert topology._INPUT not in os.environ
            assert os.environ[topology._DUMP_RANK] == '0'
            assert not path.exists()
            raw = Path(os.environ[topology._DUMP])
            raw.write_bytes(RAW)
            dumps.append(raw)
        return result

    monkeypatch.setattr(dist, 'init_process_group', initialize)
    assert topology.init_process_group(**cache_init) is result
    assert len(calls) == 1
    assert topology._valid_cache(path)
    assert os.environ[topology._INPUT] == str(path)
    assert topology._DUMP not in os.environ and topology._DUMP_RANK not in os.environ
    assert all(not raw.exists() for raw in dumps)


def test_initialization_failure_restores_dump_settings_without_publishing(monkeypatch, cache_init):
    path = Path(topology._node_plan(1)['cache_path'])
    monkeypatch.setenv(topology._DUMP_RANK, '17')
    failure = RuntimeError('WORLD initialization failed')
    calls = []

    def initialize(**kwargs):
        calls.append(kwargs)
        assert os.environ[topology._DUMP_RANK] == '0'
        Path(os.environ[topology._DUMP]).write_bytes(RAW)
        raise failure

    monkeypatch.setattr(dist, 'init_process_group', initialize)
    with pytest.raises(RuntimeError) as caught:
        topology.init_process_group(**cache_init)
    assert caught.value is failure
    assert len(calls) == 1
    assert topology._DUMP not in os.environ
    assert os.environ[topology._DUMP_RANK] == '17'
    assert topology._INPUT not in os.environ
    assert not path.exists() and not path.with_suffix('.sha256').exists()


def test_unavailable_cache_still_initializes_once(monkeypatch, cache_init):
    monkeypatch.setattr(topology, '_node_plan', lambda size: dict(available=False, cache_hit=False))
    calls = []
    result = object()

    def initialize(**kwargs):
        calls.append(kwargs)
        assert topology._DUMP not in os.environ and topology._INPUT not in os.environ
        return result

    monkeypatch.setattr(dist, 'init_process_group', initialize)
    assert topology.init_process_group(**cache_init) is result
    assert len(calls) == 1


def test_rendezvous_returns_prepared_args_without_mutating_input(monkeypatch):
    store = dist.HashStore()
    supplied = dict(backend='nccl', init_method='file:///example', rank=-1, world_size=-1)
    calls = []

    def rendezvous(method, rank, world_size, *, timeout):
        calls.append((method, rank, world_size))
        return iter([(store, 0, 2)])

    monkeypatch.setattr(dist, 'rendezvous', rendezvous)
    resolved = topology._rendezvous_kwargs(supplied)
    assert calls == [('file:///example', -1, -1)]
    assert supplied == dict(backend='nccl', init_method='file:///example', rank=-1, world_size=-1)
    assert resolved['rank'] == 0 and resolved['world_size'] == 2
    assert 'init_method' not in resolved
    resolved['store'].set('probe', 'value')
    assert store.check(['default_pg/probe'])


def test_topology_context_yields_args_without_mutating_input(cache_init):
    original = cache_init.copy()
    with topology._topology_cache(cache_init) as prepared:
        assert prepared['device_id'] == torch.device('cuda', 0)
        Path(os.environ[topology._DUMP]).write_bytes(RAW)
    assert cache_init == original


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
