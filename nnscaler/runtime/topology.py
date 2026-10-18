# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reuse node-local NCCL hardware topology without changing user overrides."""
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import tempfile
import uuid
import xml.etree.ElementTree as ET

import torch
import torch.distributed as dist
from torch.distributed import constants

_logger = logging.getLogger(__name__)
_generation = itertools.count()
_managed_file = None
_FORMAT = 2
_DUMP = 'NCCL_TOPO_DUMP_FILE'
_DUMP_RANK = 'NCCL_TOPO_DUMP_FILE_RANK'
_INPUT = 'NCCL_TOPO_FILE'


def _read(path):
    return Path(path).read_text().strip()


def _configuration():
    paths = [Path('/etc/nccl.conf'), Path(os.environ.get('NCCL_CONF_FILE', Path.home() / '.nccl.conf'))]
    contents = {}
    for path in paths:
        try:
            contents[str(path)] = path.read_text()
        except FileNotFoundError:
            pass
    return contents


def _has_topology_override():
    if os.environ.get(_INPUT) or os.environ.get(_DUMP):
        return True
    try:
        return any(line.partition('=')[0].strip() in (_INPUT, _DUMP)
                   for content in _configuration().values() for line in content.splitlines())
    except OSError:
        # An unreadable config may contain an override; leave initialization alone.
        return True


def _fingerprint(local_size):
    # These are lightweight proc/sysfs reads, not NVML NVLink discovery.
    gpu_files = sorted(Path('/proc/driver/nvidia/gpus').glob('*/information'))
    if not gpu_files:
        raise OSError('NVIDIA GPU inventory is unavailable')
    visible = [getattr(torch.cuda.get_device_properties(i), 'uuid', None)
               for i in range(torch.cuda.device_count())]
    if not visible or any(value is None for value in visible):
        raise OSError('Visible CUDA device identities are unavailable')
    network = {}
    for adapter in sorted(Path('/sys/class/infiniband').glob('*')):
        network[adapter.name] = {
            'device': str((adapter / 'device').resolve()),
            'guid': _read(adapter / 'node_guid'),
            'ports': {str(p.relative_to(adapter)): _read(p)
                      for p in sorted((adapter / 'ports').glob('*/*'))
                      if p.name in ('state', 'phys_state', 'rate', 'link_layer')},
        }
    ignored = {_INPUT, _DUMP, _DUMP_RANK, 'NCCL_DEBUG', 'NCCL_DEBUG_FILE', 'NCCL_DEBUG_SUBSYS'}
    environment = {key: value for key, value in os.environ.items()
                   if (key.startswith('NCCL_') and key not in ignored)
                   or key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH')}
    configuration = _configuration()
    virtual_topology = Path('/var/run/nvidia-topologyd/virtualTopology.xml')
    try:
        configuration[str(virtual_topology)] = virtual_topology.read_text()
    except FileNotFoundError:
        pass
    value = dict(format=_FORMAT, boot=_read('/proc/sys/kernel/random/boot_id'),
                 driver=_read('/proc/driver/nvidia/version'),
                 gpus={p.parent.name: _read(p) for p in gpu_files},
                 visible_gpus=list(map(str, visible)), network=network,
                 torch=str(torch.__version__), cuda=torch.version.cuda,
                 nccl=torch.cuda.nccl.version(), local_size=local_size,
                 environment=environment, configuration=configuration)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _clean_xml(data):
    root = ET.fromstring(data)
    if root.tag != 'system' or root.get('version') != '1' or not list(root.iter('gpu')):
        raise ValueError('Not an NCCL GPU topology')
    hosts = {node.get('host_hash') for node in root.iter('cpu') if node.get('host_hash')}
    if len(hosts) > 1:
        raise ValueError('Multi-host topology cannot be cached as node-local hardware')
    for node in root.iter():
        node.attrib.pop('rank', None)
        node.attrib.pop('keep', None)
        node.attrib.pop('gdr', None)
        # Network-plugin properties and GDR support depend on the current job.
        # NCCL rediscovers these; retain only the PCI/GPU/NVLink hardware tree.
        for child in list(node):
            if child.tag == 'net':
                node.remove(child)
    return ET.tostring(root, encoding='utf-8')


def _valid_cache(path):
    try:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != _read(path.with_suffix('.sha256')):
            return False
        root = ET.fromstring(data)
        return (root.tag == 'system' and root.get('version') == '1' and bool(list(root.iter('gpu')))
                and all(node.tag != 'net' and not {'rank', 'keep', 'gdr'}.intersection(node.attrib)
                        for node in root.iter()))
    except (OSError, ET.ParseError):
        return False


def _atomic_write(path, data):
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _node_plan(local_size):
    try:
        key = _fingerprint(local_size)
        directory = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'nnscaler' / 'nccl-topology'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (key + '.xml')
        return dict(ready=True, cached=_valid_cache(path), path=str(path),
                    raw=str(directory / f'.{key}.{uuid.uuid4().hex}.raw.xml'))
    except (OSError, ValueError) as error:
        _logger.debug('NCCL topology caching unavailable: %s', error)
        return dict(ready=False, cached=False)


def _restore(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def init_process_group(backend=None, **kwargs):
    """Initialize a PyTorch process group, reusing local NCCL topology when possible.

    Arguments are forwarded to torch.distributed.init_process_group. With a
    device-bound NCCL WORLD, torchrun nodes cache hardware topology under
    $XDG_CACHE_HOME/nnscaler (default ~/.cache/nnscaler). Explicit NCCL_TOPO_FILE
    and NCCL_TOPO_DUMP_FILE settings take precedence. No GPU ranks or live
    communicators are cached.
    """
    global _managed_file
    kwargs['backend'] = backend
    device = kwargs.get('device_id')
    if isinstance(device, int):
        device = kwargs['device_id'] = torch.device('cuda', device)
    if (dist.is_initialized() or backend != 'nccl' or device is None
            or device.type != 'cuda' or not dist.is_nccl_available()
            or 'LOCAL_WORLD_SIZE' not in os.environ):
        return dist.init_process_group(**kwargs)
    if os.environ.get(_INPUT) == _managed_file and _managed_file is not None:
        os.environ.pop(_INPUT)
        _managed_file = None
    user_override = _has_topology_override()

    timeout = kwargs.get('timeout') or getattr(constants, 'default_pg_nccl_timeout', constants.default_pg_timeout)
    rank, world = kwargs.get('rank', -1), kwargs.get('world_size', -1)
    store = kwargs.get('store')
    if store is None:
        method = kwargs.pop('init_method', None) or 'env://'
        store, rank, world = next(dist.rendezvous(method, rank, world, timeout=timeout))
        store.set_timeout(timeout)
        store = dist.PrefixStore('default_pg', store)
        kwargs.update(store=store, rank=rank, world_size=world)
    elif kwargs.get('init_method') or rank < 0 or world <= 0:
        return dist.init_process_group(**kwargs)

    local_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_size <= 0 or world % local_size or local_rank != rank % local_size:
        raise ValueError('NCCL topology caching requires homogeneous torchrun nodes')
    leader = rank - local_rank
    token = f'{os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")}-{next(_generation)}'
    coordination = dist.PrefixStore(f'nnscaler-topology/{token}', store)
    # An override on even one rank disables automatic export everywhere; taking
    # different dump branches would introduce mismatched bootstrap collectives.
    coordination.set(f'override/{rank}', str(int(user_override)))
    if rank == 0:
        overrides = coordination.multi_get([f'override/{i}' for i in range(world)])
        coordination.set('external', str(int(any(value == b'1' for value in overrides))))
    if coordination.get('external') == b'1':
        return dist.init_process_group(**kwargs)
    if local_rank == 0:
        coordination.set(str(leader), json.dumps(_node_plan(local_size)))
    if rank == 0:
        plans = [json.loads(coordination.get(str(i))) for i in range(0, world, local_size)]
        # Dumping introduces extra NCCL bootstrap collectives. All ranks must
        # make the same choice, including when only some nodes have a cache hit.
        mode = ('disabled' if not all(p['ready'] for p in plans) else
                'cached' if all(p['cached'] for p in plans) else 'dump')
        coordination.set('mode', mode)
    mode = coordination.get('mode').decode()
    plan = json.loads(coordination.get(str(leader)))
    if mode == 'disabled':
        return dist.init_process_group(**kwargs)

    path = Path(plan['path'])
    old_dump, old_rank = os.environ.get(_DUMP), os.environ.get(_DUMP_RANK)
    if mode == 'cached':
        if _valid_cache(path):
            os.environ[_INPUT] = _managed_file = str(path)
    else:
        os.environ[_DUMP] = plan['raw']
        os.environ[_DUMP_RANK] = str(leader)
    try:
        result = dist.init_process_group(**kwargs)
    finally:
        _restore(_DUMP, old_dump)
        _restore(_DUMP_RANK, old_rank)

    if mode == 'dump':
        if local_rank == 0:
            raw = Path(plan['raw'])
            try:
                data = _clean_xml(raw.read_bytes())
                _atomic_write(path, data)
                _atomic_write(path.with_suffix('.sha256'), hashlib.sha256(data).hexdigest().encode())
                ready = str(path)
            except (OSError, ValueError, ET.ParseError) as error:
                _logger.warning('Continuing without NCCL topology cache: %s', error)
                ready = ''
            finally:
                try:
                    raw.unlink(missing_ok=True)
                except OSError:
                    pass
            coordination.set(f'ready/{leader}', ready)
        ready = coordination.get(f'ready/{leader}').decode()
        if ready and _valid_cache(Path(ready)):
            os.environ[_INPUT] = _managed_file = ready
    return result
