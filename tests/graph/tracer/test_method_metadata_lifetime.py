import gc
import weakref

import torch
from torch.fx import GraphModule

from nnscaler.graph.tracer.concrete_tracer import ConcreteTracer


class RepeatedMethods(torch.nn.Module):
    def forward(self, x):
        for _ in range(8):
            x = (x + 1).sin()
        return x


def test_unused_method_metadata_does_not_keep_intermediates():
    tracer = ConcreteTracer('cpu')
    original = tracer.create_node
    observed = []

    def capture(*args, **kwargs):
        node = original(*args, **kwargs)
        result = kwargs.get('node_result', args[6] if len(args) > 6 else None)
        if getattr(node.target, '__name__', None) == 'add' and torch.is_tensor(result):
            observed.append(weakref.ref(result))
        return node

    tracer.create_node = capture
    model = RepeatedMethods()
    x = torch.randn(32)
    graph = tracer.trace(model, concrete_args={'x': x})
    gc.collect()
    assert len(observed) == 8
    assert all(ref() is None for ref in observed)
    # Even without DCE the unused getattr operations remain executable.
    traced = GraphModule(tracer.root, graph)
    torch.testing.assert_close(traced(x), model(x), rtol=0, atol=0)


class MethodAsValue(torch.nn.Module):
    def forward(self, x):
        method = x.sin
        with torch.no_grad():
            y = method()
        # The same method is used as a value after a direct call; its getattr
        # metadata/context must be restored, without altering the call context.
        return y, method


def test_method_used_as_a_value_restores_metadata_and_original_context():
    tracer = ConcreteTracer('cpu')
    x = torch.randn(32, requires_grad=True)
    graph = tracer.trace(MethodAsValue(), concrete_args={'x': x})
    attrs = [node for node in graph.nodes if node.target is getattr and node.users]
    assert len(attrs) == 1
    assert callable(attrs[0].meta['tensor_meta'])
    assert attrs[0].meta['op_context']['grad_mode']['grad_mode']
    calls = [node for node in graph.nodes if node.op == 'call_method']
    assert len(calls) == 1
    assert calls[0].meta['op_context']['grad_mode']['no_grad_mode']
    y, method = GraphModule(tracer.root, graph)(x)
    torch.testing.assert_close(y, x.sin())
    torch.testing.assert_close(method(), x.sin())
