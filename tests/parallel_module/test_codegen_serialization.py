#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import io
import subprocess
import sys
from types import SimpleNamespace

import dill
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
        assert dump_codegen_payload(payload, stream) == 'pickle'
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
    assert factory is CustomizedOps.kOpMap[signature]
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
    assert dump_codegen_payload(payload, stream) == 'pickle'
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
        assert dump_codegen_payload({'factory': CustomizedOps.kOpMap[signature]}, stream) == 'pickle'
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


def test_shared_closure_state_uses_whole_payload_fallback():
    state = []

    def append(value):
        state.append(value)

    def read():
        return state

    stream = io.BytesIO()
    assert dump_codegen_payload({'append': append, 'read': read}, stream) == 'dill'
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
    assert dump_codegen_payload({'append': append, 'state': state}, stream) == 'dill'
    stream.seek(0)
    restored = load_codegen_payload(stream)
    restored['append'](42)
    assert restored['state'] == [42]


def test_reshape_partition_rules_keep_fast_path():
    _, rules = _reshape_anno([4, 6], [2, 2, 6], 'size')
    kwargs = {'size': (2, 2, 6)}
    expected = rules[0].modifier()(kwargs, 0, 0, 2, 0)
    stream = io.BytesIO()
    assert dump_codegen_payload({'rules': rules}, stream) == 'pickle'
    stream.seek(0)
    restored = load_codegen_payload(stream)
    assert restored['rules'][0].modifier()(kwargs, 0, 0, 2, 0) == expected
    assert kwargs == {'size': (2, 2, 6)}


def test_local_class_fallback_replaces_partial_pickle():
    class LocalConfig:
        value = 42

    stream = io.BytesIO(b'prefix')
    stream.seek(6)
    # Force the fast pickler to write data before encountering the local class.
    assert dump_codegen_payload({'data': b'x' * 100_000, 'config': LocalConfig()}, stream) == 'dill'
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


@pytest.mark.parametrize('serializer', ['pickle', 'dill', 'nested_callable'])
def test_cuda_storage_in_payload_loads_on_cpu(serializer, monkeypatch):
    # Produce a real CUDA-tagged torch storage without allocating a GPU.
    # The worker must remap it before the default CUDA deserializer runs.
    registry = list(torch.serialization._package_registry)
    monkeypatch.setattr(torch.serialization, '_package_registry', registry)
    torch.serialization.register_package(
        0, lambda storage: 'cuda:0' if storage.device.type == 'cpu' else None,
        lambda storage, location: None,
    )
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    view = tensor[:, 1:]
    payload = {'tensor': tensor, 'alias': tensor, 'view': view,
               'device_metadata': torch.device('cuda:0')}
    stream = io.BytesIO()
    if serializer == 'pickle':
        assert dump_codegen_payload(payload, stream) == 'pickle'
    elif serializer == 'dill':
        dill.dump(payload, stream)
    else:
        def read_tensor(value=tensor):
            return value
        # Exercise the nested dill callable resolver too.
        from nnscaler.codegen.serialization import _load_dill_callable
        restored = _load_dill_callable(dill.dumps(read_tensor))()
        assert restored.device.type == 'cpu'
        assert torch.equal(restored, tensor)
        return
    assert b'cuda:0' in stream.getvalue()
    stream.seek(0)
    restored = load_codegen_payload(stream)
    assert restored['tensor'].device.type == 'cpu'
    assert restored['tensor'] is restored['alias']
    assert torch.equal(restored['tensor'], tensor)
    assert torch.equal(restored['view'], view)
    assert restored['view'].stride() == view.stride()
    assert restored['view'].storage_offset() == view.storage_offset()
    assert restored['device_metadata'] == torch.device('cuda:0')


def test_io_errors_do_not_retry_with_dill(monkeypatch):
    class BrokenStream(io.BytesIO):
        def write(self, data):
            raise OSError('disk full')

    def unexpected_fallback(*args, **kwargs):
        pytest.fail('I/O errors must not start a second serialization')

    monkeypatch.setattr(dill, 'dump', unexpected_fallback)
    with pytest.raises(OSError, match='disk full'):
        dump_codegen_payload({'data': b'x' * 100_000}, BrokenStream())


def _make_closures():
    # One local function, many closures: what `_reshape_anno` produces per node.
    def factory(first, second, name):
        def modifier(kwargs):
            return first, second, name
        return modifier
    states = [(('a', 'b'), ('b', 'a'), 'shape'), (('a', 'b'), ('b', 'a'), 'shape'),
              (('a', None), ('a',), 'size'), (1, 2, 'x'), (True, 2, 'x'), (1.0, 2, 'x'),
              (0.0, 2, 'x'), (-0.0, 2, 'x'), ((1, 2), frozenset({3}), 'x'), ((1, True), 2, 'x')]
    return [factory(*state) for state in states], states


def test_callable_dump_cache_is_exact():
    from nnscaler.codegen import serialization as S
    import dill, io, pickle

    closures, states = _make_closures()
    pickler = S._CodegenPickler(io.BytesIO(), {})
    reduced = [pickler.reducer_override(fn) for fn in closures]
    for fn, (loader, (data,)) in zip(closures, reduced):
        assert loader is S._load_dill_callable
        # Cached bytes equal a fresh dump of this very function.
        assert data == dill.dumps(fn, protocol=pickle.HIGHEST_PROTOCOL)
    # Equal state shares one dump; 1/True/1.0 and 0.0/-0.0 stay distinct.
    assert reduced[0][1][0] is reduced[1][1][0]
    assert len({id(r[1][0]) for r in reduced}) == len(states) - 1
    for fn, (loader, (data,)) in zip(closures, reduced):
        loaded = loader(data)
        assert loaded({}) == fn({}) and [type(v) for v in loaded({})] == [type(v) for v in fn({})]


def test_callable_dump_cache_skips_unkeyable_state():
    from nnscaler.codegen import serialization as S

    def factory(value):
        def modifier():
            return value
        return modifier
    fn = factory(object())  # not copyable: the pickler refuses it as before
    import io, pickle
    with pytest.raises(pickle.PicklingError):
        S._CodegenPickler(io.BytesIO(), {}).reducer_override(fn)
    unpicklable = factory(object.__new__(type('Local', (), {})))
    assert S._callable_key(unpicklable, S._code_object_ids(unpicklable.__code__)) is None


def test_graph_dumps_loads_with_dill_and_falls_back(monkeypatch):
    import dill, io
    from nnscaler.codegen import serialization as S
    from nnscaler.graph.graph import IRGraph

    graph = IRGraph([], [], [], 'empty')
    data = graph.dumps()
    ids, loaded = dill.load(io.BytesIO(data))
    assert isinstance(loaded, IRGraph) and loaded.name == 'empty'
    calls = []
    monkeypatch.setattr(S, 'try_dumps_graph', lambda obj, g: calls.append(g) or None)
    fallback = graph.dumps()
    assert calls == [graph]
    assert isinstance(dill.load(io.BytesIO(fallback))[1], IRGraph)


def test_callable_dump_cache_on_reshape_modifiers():
    from nnscaler.codegen import serialization as S
    from nnscaler.graph.function.function import _reshape_anno
    import dill, io, pickle

    shapes = [([4, 6], [24], 'shape'), ([4, 6], [24], 'shape'), ([2, 3, 4], [6, 4], 'shape'), ([4, 6], [24], 'size')]
    modifiers = []
    for in_shape, ou_shape, name in shapes:
        _, rules = _reshape_anno(in_shape, ou_shape, name)
        modifiers += [rule.modifier() for rule in rules if rule.modifier() is not None]
    assert modifiers and all(fn.__qualname__ == '_reshape_anno.<locals>.modifier' for fn in modifiers)
    pickler = S._CodegenPickler(io.BytesIO(), {})
    for fn in modifiers:
        loader, (data,) = pickler.reducer_override(fn)
        assert data == dill.dumps(fn, protocol=pickle.HIGHEST_PROTOCOL)
    assert len(pickler._callable_dumps) < len(modifiers)


def _emitted(pickler, fn):
    from nnscaler.codegen import serialization as S
    loader, (data,) = pickler.reducer_override(fn)
    assert loader is S._load_dill_callable
    return data


def _assert_cache_matches_fresh_dumps(closures):
    """Every emitted dump equals a fresh dill dump, and loading preserves behavior."""
    from nnscaler.codegen import serialization as S
    import dill, io, pickle

    pickler = S._CodegenPickler(io.BytesIO(), {})
    emitted = [_emitted(pickler, fn) for fn in closures]
    for fn, data in zip(closures, emitted):
        assert data == dill.dumps(fn, protocol=pickle.HIGHEST_PROTOCOL)
        assert S._load_dill_callable(data)() == fn()
    return pickler, emitted


def test_callable_dump_cache_keeps_identity_of_captured_objects():
    # Review case: same tuple captured twice vs two equal tuples.
    def factory(a, b):
        def same():
            return a is b
        return same
    shared = (1, 2)
    text = ''.join(['y'] * 40)
    closures = [factory(shared, shared), factory((1, 2), tuple([1, 2])), factory(shared, shared),
                factory(''.join(['x'] * 40), ''.join(['x'] * 40)), factory(text, text)]
    assert [fn() for fn in closures] == [True, False, True, False, True]
    pickler, emitted = _assert_cache_matches_fresh_dumps(closures)
    assert emitted[0] is emitted[2]  # equal state still shares one dump
    assert emitted[0] != emitted[1] and emitted[3] != emitted[4]


def test_callable_dump_cache_keeps_frozenset_iteration_order():
    # Review case: equal frozensets whose iteration orders differ.
    first, second = frozenset([0, 8]), frozenset([8, 0])
    assert first == second
    if list(first) == list(second):
        pytest.skip('this Python build iterates both frozensets in the same order')

    def factory(values):
        def order():
            return list(values)
        return order
    closures = [factory(first), factory(second)]
    _, emitted = _assert_cache_matches_fresh_dumps(closures)
    assert emitted[0] != emitted[1]


def test_callable_dump_cache_keeps_sharing_with_code_constants():
    # A captured value may be the code object's own constant (an alias dill
    # writes as a memo reference) or an equal copy.
    from nnscaler.codegen import serialization as S

    def factory(name):
        def check():
            return name, 'shared-constant-abc'
        return check
    literal = factory(None)()[1]
    copy = ''.join(['shared-constant-', 'abc'])
    assert copy == literal and copy is not literal
    closures = [factory(literal), factory(copy), factory(literal)]
    _, emitted = _assert_cache_matches_fresh_dumps(closures)
    assert emitted[0] is emitted[2] and emitted[0] != emitted[1]
    loaded = [S._load_dill_callable(data)() for data in emitted]
    assert [value[0] is value[1] for value in loaded] == [True, False, True]
