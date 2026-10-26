#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

from nnscaler.ir.tensor import IRFullTensor, IRSubTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.graph import IRGraph


def test_graph_from_logic():

    node = IRFwOperation("test", "test",
                         inputs=[IRFullTensor([256, 256])],
                         num_outputs=1,
                         # kwargs
                         kw={
                           'a':[IRFullTensor([128, 256]),],
                           'b':IRFullTensor([128, 128])
                        },
                         t=IRFullTensor([128, 256]))
    output = IRFullTensor([128, 256])
    node.set_output(0, output)
    graph = IRGraph.from_logic_graph([node], [node.input(0)], [output], 'GenModule')
    assert len(graph.nodes()) == 1
    node = graph.node(0)
    print(node.kwargs)
    assert isinstance(node.input(0), IRSubTensor)
    assert isinstance(node.output(0), IRSubTensor)
    assert isinstance(node.kwargs['kw']['a'][0], IRSubTensor)
    assert isinstance(node.kwargs['kw']['b'], IRSubTensor)
    assert isinstance(node.kwargs['t'], IRSubTensor)


def test_graph_kwargs_track():

    node = IRFwOperation("test", "test",
                         inputs=[IRFullTensor([256, 256])],
                         num_outputs=1,
                         # kwargs
                         kw={
                           'a':[IRFullTensor([128, 256]),],
                           'b':IRFullTensor([128, 128])
                        },
                         t=IRFullTensor([128, 256]))
    output = IRFullTensor([128, 256])
    node.set_output(0, output)
    graph = IRGraph.from_logic_graph([node], [node.input(0), node.kwargs['t']], [output], 'GenModule')
    assert len(graph.nodes()) == 1
    assert len(graph.full_tensors()) == 5
    args = [IRFullTensor([256, 256]).tosub(), IRFullTensor([128, 256]).tosub()]
    # forward replace
    graph(*args)
    assert graph.input(0) == args[0]
    assert graph.input(1) == args[1]
    assert graph.node(0).kwargs['t'] == args[1]


def test_recompute_and_offload_reject_empty_nodes():
    graph = IRGraph([], [], [], 'GenModule')

    with pytest.raises(AssertionError, match='Cannot recompute empty list of nodes'):
        graph.recompute([])
    with pytest.raises(AssertionError, match='Cannot offload empty list of nodes'):
        graph.offload([])


def test_narrowed_alias_preserves_combined_memory_metadata():
    from nnscaler.graph.function.function import Identity
    from nnscaler.graph.segment import IRSegment, IRSegmentExpander

    tensor = IRFullTensor([8, 4]).tosub()
    alias = Identity(tensor)
    alias.set_output(0, IRFullTensor([8, 4]).tosub())
    alias.device = 0
    alias.recompute = 123
    alias.offload = 456
    segment = IRSegment([alias], [tensor], list(alias.outputs()))
    IRSegmentExpander(segment, set(), set())
    input_layout = tensor.parent.select(((0, 4), (0, 4)), (0, 1))
    output_layout = alias.output(0).parent.select(((0, 4), (0, 4)), (0, 1))
    replaced = {}
    nodes = list(IRSegmentExpander._fix_per_device_aliases(
        segment, {0: [input_layout]}, {0: [output_layout]}, replaced))
    assert nodes[0] is not alias
    assert nodes[0].input(0).shape == (4, 4)
    assert nodes[0].recompute == 123
    assert nodes[0].offload == 456
