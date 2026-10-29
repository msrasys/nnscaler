#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.ir.tensor import IRFullTensor, IRSubTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.graph import IRGraph


def test_copy_node_metadata_keeps_module_invocation_ids():
    from collections import OrderedDict
    source = IRFwOperation('source', 'source', inputs=[], num_outputs=0)
    target = IRFwOperation('target', 'target', inputs=[], num_outputs=0)
    source.module_call_stack = OrderedDict([('layers.0', 3), ('layers.0.reference', 4)])
    IRGraph.copy_node_meta_info(source, target)
    assert target.module_call_stack == source.module_call_stack


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
