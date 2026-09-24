import dis
import importlib.util
import inspect
import gc
import weakref

from nnscaler.graph.tracer.frame_utils import _instructions, get_frame_record, get_last_instruction


def test_nearest_user_frame_and_source_line():
    line = inspect.currentframe().f_lineno + 1
    record = get_frame_record()
    assert record.filename == __file__
    assert record.lineno == line
    assert record.name == 'test_nearest_user_frame_and_source_line'
    assert record.line == 'record = get_frame_record()'


def test_dynamic_source_line_refresh(tmp_path):
    path = tmp_path/'caller.py'
    source = ('from nnscaler.graph.tracer.frame_utils import get_frame_record\n'
              'def caller():\n    return get_frame_record()\n')
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('frame_caller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.caller().line == 'return get_frame_record()'
    path.write_text(source.replace('return get_frame_record()', 'return get_frame_record()  # edited'))
    assert module.caller().line == 'return get_frame_record()  # edited'


def test_disassembly_cache_uses_code_identity_and_is_bounded():
    def original(x):
        return x + 1
    def replacement(x):
        return x * 2
    _instructions.cache_clear()
    expected = tuple(dis.get_instructions(original))
    assert _instructions(original.__code__) == expected
    assert _instructions(original.__code__) is _instructions(original.__code__)
    original.__code__ = replacement.__code__
    assert _instructions(original.__code__) == tuple(dis.get_instructions(replacement))
    assert _instructions.cache_info().maxsize == 512


def test_instruction_lookup_does_not_retain_caller_locals():
    class Sentinel:
        pass
    def caller():
        sentinel = Sentinel()
        get_last_instruction()
        return weakref.ref(sentinel)
    enabled = gc.isenabled()
    gc.disable()
    try:
        assert caller()() is None
    finally:
        if enabled:
            gc.enable()
