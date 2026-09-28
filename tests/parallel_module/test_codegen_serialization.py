#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import io
import subprocess
import sys
from types import SimpleNamespace

import cloudpickle
import dill
import pickle
import pytest
import torch

from nnscaler import register_op
from nnscaler.codegen.serialization import dump_codegen_payload, load_codegen_payload
from nnscaler.graph.parser.register import CustomizedOps
from nnscaler.graph.function.function import _reshape_anno


@register_op('a b -> a b')
def _neg(x):
    return -x


def _registered_during_setup(x):
    return -x


@register_op('a b -> a b')
class _AutogradNeg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return -x

    @staticmethod
    def backward(ctx, grad):
        return -grad


class _FactoryGraph:
    def __init__(self, nodes):
        self._nodes = nodes

    def nodes(self, *, flatten):
        assert flatten
        return self._nodes


def test_cached_registered_factories_load_in_fresh_process(tmp_path):
    signatures = [f'{__name__}._neg', f'{__name__}._AutogradNeg.apply']
    originals = [CustomizedOps.kOpMap[s] for s in signatures]
    restored = [dill.loads(dill.dumps(f)) for f in originals]
    assert all(a is not b for a, b in zip(originals, restored))
    graph = _FactoryGraph([
        SimpleNamespace(signature=s, _create_fn=(f,))
        for s, f in zip(signatures, restored)
    ])
    payload = {
        'factories': restored,
        'signatures': signatures,
        'module_codegen': SimpleNamespace(execplan=SimpleNamespace(graph=graph)),
    }
    path = tmp_path / 'payload.pkl'
    with path.open('wb') as stream:
        assert dump_codegen_payload(payload, stream) == 'cloudpickle'
    # No pre-import of this test module: the factory resolver must import it,
    # including the module of an autograd subclass whose apply is inherited.
    subprocess.run([sys.executable, '-c', '''
import sys
from nnscaler.codegen.serialization import load_codegen_payload
from nnscaler.graph.parser.register import CustomizedOps
from nnscaler.ir.tensor import IRFullTensor
with open(sys.argv[1], 'rb') as stream:
    payload = load_codegen_payload(stream)
for factory, signature in zip(payload['factories'], payload['signatures']):
    assert callable(factory)
    node = factory(IRFullTensor((2, 3)).tosub(), signature=signature)
    assert node.signature == signature
for node, factory in zip(payload['module_codegen'].execplan.graph.nodes(flatten=True), payload['factories']):
    assert node._create_fn[0] is factory
''', str(path)], check=True, timeout=60)


def test_fast_payload_preserves_links_and_local_emitter():
    prefix = 'torch.neg'

    def emit(argument):
        return f'{prefix}({argument})'

    tensor = torch.arange(6)
    node = SimpleNamespace(tensor=tensor)
    node.parent = node
    payload = {'node': node, 'alias': node, 'tensor': tensor, 'emitter': emit}
    stream = io.BytesIO()
    assert dump_codegen_payload(payload, stream) == 'cloudpickle'
    stream.seek(0)
    restored = load_codegen_payload(stream)
    assert restored['node'] is restored['alias']
    assert restored['node'].parent is restored['node']
    assert restored['node'].tensor is restored['tensor']
    assert torch.equal(restored['tensor'], tensor)
    assert restored['emitter']('x') == 'torch.neg(x)'


def test_factory_registered_only_in_parent_loads_in_worker(tmp_path, monkeypatch):
    signature = f'{__name__}._registered_during_setup'
    assert signature not in CustomizedOps.kOpMap
    # Preserve all registry dictionaries after this dynamically registered op.
    for name, value in vars(CustomizedOps).items():
        if name.startswith('kOp') and isinstance(value, dict):
            monkeypatch.setattr(CustomizedOps, name, dict(value))
    register_op('a b -> a b')(_registered_during_setup)
    path = tmp_path / 'dynamic.pkl'
    with path.open('wb') as stream:
        assert dump_codegen_payload({'factory': CustomizedOps.kOpMap[signature]}, stream) == 'cloudpickle'
    subprocess.run([sys.executable, '-c', '''
import sys
from nnscaler.codegen.serialization import load_codegen_payload
from nnscaler.graph.parser.register import CustomizedOps
from nnscaler.ir.tensor import IRFullTensor
with open(sys.argv[1], 'rb') as stream:
    payload = load_codegen_payload(stream)
signature = sys.argv[2]
assert signature not in CustomizedOps.kOpMap
node = payload['factory'](IRFullTensor((2, 3)).tosub(), signature=signature)
assert node.signature == signature
''', str(path), signature], check=True, timeout=60)


def test_shared_closure_state_keeps_aliases():
    state = []

    def append(value):
        state.append(value)

    def read():
        return state

    stream = io.BytesIO()
    assert dump_codegen_payload({'append': append, 'read': read}, stream) == 'cloudpickle'
    stream.seek(0)
    restored = load_codegen_payload(stream)
    restored['append'](42)
    assert restored['read']() == [42]


@pytest.mark.parametrize('capture', ['closure', 'default'])
def test_mutable_callable_state_keeps_alias_to_payload(capture):
    state = []
    if capture == 'closure':
        def append(value):
            state.append(value)
    else:
        def append(value, target=state):
            target.append(value)
    stream = io.BytesIO()
    assert dump_codegen_payload({'append': append, 'state': state}, stream) == 'cloudpickle'
    stream.seek(0)
    restored = load_codegen_payload(stream)
    restored['append'](42)
    assert restored['state'] == [42]


def test_reshape_partition_rules_keep_fast_path():
    _, rules = _reshape_anno([4, 6], [2, 2, 6], 'size')
    kwargs = {'size': (2, 2, 6)}
    expected = rules[0].modifier()(kwargs, 0, 0, 2, 0)
    stream = io.BytesIO()
    assert dump_codegen_payload({'rules': rules}, stream) == 'cloudpickle'
    stream.seek(0)
    restored = load_codegen_payload(stream)
    assert restored['rules'][0].modifier()(kwargs, 0, 0, 2, 0) == expected
    assert kwargs == {'size': (2, 2, 6)}


def test_local_class_preserves_fast_path():
    class LocalConfig:
        value = 42

    stream = io.BytesIO(b'prefix')
    stream.seek(6)
    assert dump_codegen_payload({'data': b'x' * 100_000, 'config': LocalConfig()}, stream) == 'cloudpickle'
    assert stream.getvalue().startswith(b'prefix')
    stream.seek(6)
    restored = load_codegen_payload(stream)
    assert restored['data'] == b'x' * 100_000
    assert restored['config'].value == 42
    assert stream.read() == b''


def test_legacy_dill_payload_still_loads():
    offset = 3
    payload = {'fn': lambda x: x + offset}
    restored = load_codegen_payload(io.BytesIO(dill.dumps(payload)))
    assert restored['fn'](2) == 5


def test_io_errors_do_not_retry_with_dill(monkeypatch):
    class BrokenStream(io.BytesIO):
        def write(self, data):
            raise OSError('disk full')

    def unexpected_fallback(*args, **kwargs):
        pytest.fail('I/O errors must not start a second serialization')

    monkeypatch.setattr(dill, 'dump', unexpected_fallback)
    with pytest.raises(OSError, match='disk full'):
        dump_codegen_payload({'data': b'x' * 100_000}, BrokenStream())


@pytest.mark.parametrize('error', [pickle.PicklingError, TypeError, AttributeError, RecursionError])
def test_fallback_replaces_partial_payload(monkeypatch, error):
    def fail_dump(payload, stream, **kwargs):
        stream.write(b'partial cloudpickle payload' * 100)
        raise error('unsupported object')

    monkeypatch.setattr(cloudpickle, 'dump', fail_dump)
    state = []
    def append(value):
        state.append(value)
    stream = io.BytesIO(b'prefix')
    stream.seek(6)
    assert dump_codegen_payload({'append': append, 'state': state}, stream) == 'dill'
    assert stream.getvalue().startswith(b'prefix')
    stream.seek(6)
    restored = load_codegen_payload(stream)
    restored['append'](42)
    assert restored['state'] == [42]
    assert stream.read() == b''
