"""Time the real compile phases without changing model or graph semantics."""
import cProfile
import gc
import signal
import psutil
import weakref
import functools
import importlib
import json
import os
from pathlib import Path
import pstats
import runpy
import sys
import time


def main():
    root = Path(os.environ['REF_KL_PROFILE_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    llm = Path(os.environ['REF_KL_LLM_ROOT'])
    sys.path.insert(0, str(llm / 'llm'))
    import torch
    from nnscaler.graph.parser.frame import Frame
    from nnscaler.graph import parser as graph_parser
    from nnscaler.graph.segment import IRSegment
    from nnscaler.graph import IRGraph
    parallel = importlib.import_module('nnscaler.parallel')
    events = (root / 'phases.jsonl').open('w', buffering=1)
    profile = None
    exist_calls = 0
    exist_seconds = 0.0
    exist_depth = 0
    original_exist = IRSegment.exist
    def measured_exist(self, *args, **kwargs):
        nonlocal exist_calls, exist_seconds, exist_depth
        exist_calls += 1
        outer = exist_depth == 0
        if outer:
            started = time.perf_counter()
        exist_depth += 1
        try:
            return original_exist(self, *args, **kwargs)
        finally:
            exist_depth -= 1
            if outer:
                exist_seconds += time.perf_counter() - started
    IRSegment.exist = measured_exist

    def record(event, **details):
        value = dict(event=event, wall=time.time(), monotonic=time.monotonic(), **details)
        events.write(json.dumps(value) + '\n')
        print('REF_KL_PROFILE ' + json.dumps(value), flush=True)

    if os.environ.get('REF_KL_TRACE_DIAGNOSTIC'):
        from nnscaler.graph.tracer.concrete_tracer import ConcreteTracer
        from nnscaler.graph.tracer import wrap_utils, pytree_utils
        original_node = ConcreteTracer.create_node
        seen = []
        calls = 0
        def observed_node(self, *args, **kwargs):
            nonlocal calls
            node = original_node(self, *args, **kwargs)
            with wrap_utils.do_temp_call_origin():
                result = kwargs.get('node_result', args[6] if len(args) > 6 else None)
                if args and args[0] not in ('get_attr', 'placeholder'):
                    for value in pytree_utils.tree_leaves(result):
                        if torch.is_tensor(value) and value.device.type == 'cpu' and value.numel() * value.element_size() >= 2**20:
                            seen.append((weakref.ref(value), tuple(value.shape), value.requires_grad, value.is_leaf))
                calls += 1
                if calls % 500 == 0:
                    before = psutil.Process().memory_info().rss
                    live_before = sum(ref() is not None for ref, *_ in seen)
                    collected = gc.collect()
                    after = psutil.Process().memory_info().rss
                    live = [(ref(), shape, req, leaf) for ref, shape, req, leaf in seen if ref() is not None]
                    constants = sum(t.numel() * t.element_size() for t in self.tensor_attrs if torch.is_tensor(t))
                    record('trace_gc', nodes=calls, rss_before=before, rss_after=after, collected=collected, live_before=live_before, live_after=len(live), constant_bytes=constants, examples=[{'shape':shape,'requires_grad':req,'leaf':leaf,'referrers':[type(x).__name__ for x in gc.get_referrers(t)][:8]} for t,shape,req,leaf in live[:3]])
            return node
        ConcreteTracer.create_node = observed_node

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))

    def timed(owner, name):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            details = {}
            if name in ('save_attr_content', 'save_np_buffer_content'):
                attrs = args[0]._attr_map
                details['logical_bytes'] = sum(v.numel() * v.element_size() for t, (_, v) in attrs.items()
                    if name == 'save_attr_content' or (t.is_buffer() and not t.is_persistent()))
            record(name + ':start', **details)
            succeeded = False
            try:
                result = original(*args, **kwargs)
                succeeded = True
                if name == 'to_fx_graph' and os.environ.get('REF_KL_TRACE_ONLY'):
                    record('trace_only_complete')
                    raise SystemExit(0)
                return result
            finally:
                record(name + ':end', elapsed_s=time.monotonic() - start, succeeded=succeeded, **details)
        setattr(owner, name, wrapper)

    for owner, names in [(Frame, ['save_attr_content', 'save_np_buffer_content']),
                         (graph_parser, ['to_fx_graph', 'to_ir_graph']),
                         (parallel, ['_gen_graph', '_gencode', '_gencode_in_subprocesses'])]:
        for name in names:
            timed(owner, name)
    # cProfile retains the original qualified exist/index function names.
    record('entry', arguments=sys.argv[1:])
    # Keep whole-program cProfile off for representative graph replay timings.
    entry = llm / 'llm/nnscaler_train_vl.py'
    sys.argv[0] = str(entry)
    try:
        runpy.run_path(str(entry), run_name='__main__')
        record('complete')
    finally:
        record('exist_measurement', calls=exist_calls, inclusive_outer_seconds=exist_seconds, note='Outermost calls only; child codegen processes are not instrumented')
        events.close()


if __name__ == '__main__':
    main()
