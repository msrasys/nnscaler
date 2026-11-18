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


def test_graph_dump_keeps_cells(tmp_path):
    """
    A dump keeps `IRObject.cell`, the source of `IRObject.device`, including owners the graph does not hold
    (here the bare holder of `IR.set_object_device`); a dump of an older version, which drops cells, still loads.
    """
    import dill
    from nnscaler.ir.cten import IR
    from nnscaler.ir.unique import IDGenerator

    node = IRFwOperation('test', 'test', inputs=[IRFullTensor([4, 4])], num_outputs=1)
    node.set_output(0, IRFullTensor([4, 4]))
    graph = IRGraph.from_logic_graph([node], [node.input(0)], [node.output(0)], 'GenModule')
    graph.node(0).device = 1

    graph.dump(tmp_path / 'graph.ckp')
    loaded_node = IRGraph.load(tmp_path / 'graph.ckp').node(0)
    assert all(t.cell is loaded_node for t in loaded_node.iobjs() + loaded_node.oobjs())
    assert loaded_node.input(0).device == (1,)

    held = IR.set_object_device(IRFullTensor([2]).tosub(), 3)
    loaded_graph, loaded_held = IR.loads(IR.dumps((graph, held)))
    assert loaded_held.device == (3,) and loaded_graph.node(0).output(0).device == (1,)

    # older versions dumped `(id_state, graph)` with one plain `dill.dumps` and dropped every cell
    with IR.no_cell_pickle_context():
        (tmp_path / 'old.ckp').write_bytes(dill.dumps((IDGenerator().get_states(), graph)))
    old_node = IRGraph.load(tmp_path / 'old.ckp').node(0)
    assert old_node.input(0).cell is old_node and old_node.input(0).device == (1,)
    with pytest.raises(ValueError, match='carries no IRObject cells'):
        IR.loads((tmp_path / 'old.ckp').read_bytes())
