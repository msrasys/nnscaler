import pytest
import torch
from torch.fx import GraphModule

from nnscaler.graph.tracer.concrete_tracer import ConcreteTracer


class BranchModel(torch.nn.Module):
    def forward(self, x):
        if x.sum().item() > 0:
            return x + 1
        return x - 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_reusing_tracer_does_not_reuse_previous_input_values():
    model = BranchModel()
    tracer = ConcreteTracer('reuse_cache')
    for sign in [1, -1]:
        x = torch.full((4,), float(sign))
        graph = tracer.trace(model, concrete_args={'x': x})
        traced = GraphModule(tracer.root, graph)
        assert torch.equal(traced(x), model(x))
        assert not tracer.strategy.cache
        assert tracer.strategy.cache_size == 0


class FailingModel(torch.nn.Module):
    def forward(self, x):
        y = x.sin()
        raise ValueError('intentional trace failure')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_failed_trace_also_releases_intermediate_values():
    tracer = ConcreteTracer('reuse_cache')
    with pytest.raises(ValueError, match='intentional trace failure'):
        tracer.trace(FailingModel(), concrete_args={'x': torch.ones(4)})
    assert not tracer.strategy.cache
    assert tracer.strategy.cache_size == 0
