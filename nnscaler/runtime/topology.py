# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reuse node-local NCCL hardware topology without changing user overrides."""
from contextlib import contextmanager
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
        return dict(available=True, cache_hit=_valid_cache(path), cache_path=str(path),
                    dump_path=str(directory / f'.{key}.{uuid.uuid4().hex}.raw.xml'))
    except (OSError, ValueError) as error:
        _logger.debug('NCCL topology caching unavailable: %s', error)
        return dict(available=False, cache_hit=False)


def _rendezvous_kwargs(kwargs):
    """Return initialization arguments with a resolved Store, or None if invalid."""
    rank, world_size = kwargs.get('rank', -1), kwargs.get('world_size', -1)
    if kwargs.get('store') is not None:
        if kwargs.get('init_method') or rank < 0 or world_size <= 0:
            return None  # Let PyTorch report invalid arguments in the ordinary path.
        return kwargs
    timeout = kwargs.get('timeout') or getattr(constants, 'default_pg_nccl_timeout', constants.default_pg_timeout)
    method = kwargs.get('init_method') or 'env://'
    store, rank, world_size = next(dist.rendezvous(method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)
    resolved = dict(kwargs, store=dist.PrefixStore('default_pg', store), rank=rank, world_size=world_size)
    resolved.pop('init_method', None)
    return resolved


def _coordinate_cache(store, *, rank, world_size, local_size, leader, user_override):
    """Choose one cache mode for all ranks and return this node's file plan."""
    store.set(f'override/{rank}', str(int(user_override)))
    if rank == 0:
        overrides = store.multi_get([f'override/{i}' for i in range(world_size)])
        store.set('external', str(int(any(value == b'1' for value in overrides))))
    if store.get('external') == b'1':
        return 'disabled', None
    if rank == leader:
        store.set(str(leader), json.dumps(_node_plan(local_size)))
    if rank == 0:
        plans = [json.loads(store.get(str(i))) for i in range(0, world_size, local_size)]
        # Export adds NCCL bootstrap collectives. Even a partial cache miss must
        # select the same export path on every rank; user overrides bypass it.
        if not all(plan['available'] for plan in plans):
            mode = 'disabled'
        elif all(plan['cache_hit'] for plan in plans):
            mode = 'cached'
        else:
            mode = 'dump'
        store.set('mode', mode)
    return store.get('mode').decode(), json.loads(store.get(str(leader)))


def _publish_cache(plan):
    """Save a completed topology dump and return its cache path, or an empty string."""
    cache_path, dump_path = Path(plan['cache_path']), Path(plan['dump_path'])
    try:
        data = _clean_xml(dump_path.read_bytes())
        _atomic_write(cache_path, data)
        _atomic_write(cache_path.with_suffix('.sha256'), hashlib.sha256(data).hexdigest().encode())
        return str(cache_path)
    except (OSError, ValueError, ET.ParseError) as error:
        _logger.warning('Continuing without NCCL topology cache: %s', error)
        return ''
    finally:
        try:
            dump_path.unlink(missing_ok=True)
        except OSError:
            pass


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
    kwargs['backend'] = backend
    with _topology_cache(kwargs) as init_kwargs:
        return dist.init_process_group(**init_kwargs)


@contextmanager
def _topology_cache(kwargs):
    """Prepare topology before WORLD initialization and publish its cache after.

    Yield prepared initialization arguments without changing the caller's dict.
    Restore temporary dump settings on failure; publish a cache only on success.
    """
    global _managed_file
    kwargs = kwargs.copy()
    backend = kwargs['backend']
    device = kwargs.get('device_id')
    if isinstance(device, int):
        device = kwargs['device_id'] = torch.device('cuda', device)
    if (dist.is_initialized() or backend != 'nccl' or device is None
            or device.type != 'cuda' or not dist.is_nccl_available()
            or 'LOCAL_WORLD_SIZE' not in os.environ):
        yield kwargs
        return
    if os.environ.get(_INPUT) == _managed_file and _managed_file is not None:
        os.environ.pop(_INPUT)
        _managed_file = None
    user_override = _has_topology_override()

    resolved = _rendezvous_kwargs(kwargs)
    if resolved is None:
        yield kwargs
        return
    kwargs = resolved
    rank, world_size = kwargs['rank'], kwargs['world_size']

    local_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_size <= 0 or world_size % local_size or local_rank != rank % local_size:
        raise ValueError('NCCL topology caching requires homogeneous torchrun nodes')
    leader = rank - local_rank
    token = f'{os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")}-{next(_generation)}'
    coordination = dist.PrefixStore(f'nnscaler-topology/{token}', kwargs['store'])
    mode, plan = _coordinate_cache(
        coordination, rank=rank, world_size=world_size,
        local_size=local_size, leader=leader, user_override=user_override)
    if mode == 'disabled':
        yield kwargs
        return

    cache_path = Path(plan['cache_path'])
    old_dump, old_rank = os.environ.get(_DUMP), os.environ.get(_DUMP_RANK)
    if mode == 'cached':
        if _valid_cache(cache_path):
            os.environ[_INPUT] = _managed_file = str(cache_path)
    else:
        os.environ[_DUMP] = plan['dump_path']
        os.environ[_DUMP_RANK] = str(leader)
    try:
        yield kwargs
    finally:
        _restore(_DUMP, old_dump)
        _restore(_DUMP_RANK, old_rank)

    if mode == 'dump':
        if rank == leader:
            coordination.set(f'ready/{leader}', _publish_cache(plan))
        published_path = coordination.get(f'ready/{leader}').decode()
        if published_path and _valid_cache(Path(published_path)):
            os.environ[_INPUT] = _managed_file = published_path
