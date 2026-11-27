#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler.ir.tensor import IRFullTensor, IRSubTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.parallel import _gen_graph

from ..utils import replace_all_device_with


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


class _MLP(torch.nn.Module):
    def __init__(self, nlayers=4):
        super().__init__()
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(nlayers)])

    def forward(self, data):
        x = data['x']
        for fc in self.fcs:
            x = torch.relu(fc(x))
        return x.sum()


def _traced_graph(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    graph, _ = _gen_graph(
        _MLP(), {'data': {'x': torch.randn(4, 8)}}, outdir,
        constant_folding=True, end2end_mode=True,
    )
    fnodes = [n for n in graph.nodes() if isinstance(n, IRFwOperation) and not isinstance(n, IRGraphAnchor)]
    return graph, fnodes


def _assert_indexes_consistent(graph: IRGraph):
    """Check the membership indexes and lookups against a brute-force walk of `_nodes`."""
    def walk(seg, path):
        assert seg._node_set == set(seg._nodes)
        assert seg._sub_segments == {n for n in seg._nodes if isinstance(n, IRSegment)}
        for idx, node in enumerate(seg._nodes):
            yield node, seg, path + (idx,)
            if isinstance(node, IRSegment):
                yield from walk(node, path + (idx,))

    expected = list(walk(graph, ()))
    for node, parent, path in expected:
        assert graph.exist(node)
        assert graph.index(node).indices == path
        assert graph.segment(node) is parent
    nodes = [node for node, _, _ in expected]
    assert [pos.indices for pos in graph.multi_index(nodes)] == [path for _, _, path in expected]


@replace_all_device_with('cpu')
def test_segment_indexes_track_nested_transformations(tmp_path):
    graph, fnodes = _traced_graph(tmp_path)
    graph.staging([fnodes[0], fnodes[len(fnodes) // 2]])
    _assert_indexes_consistent(graph)

    # transform nodes that now live inside stage segments
    linears = [n for n in graph.select(ntype=IRFwOperation) if n.name == 'linear']
    replicated, partitioned = linears[0], linears[-1]
    assert graph.segment(replicated) is not graph and graph.segment(partitioned) is not graph
    graph.replicate(replicated, times=2)
    graph.partition(partitioned, partitioned.algorithm('dim'), idx=1, dim=0, num=2)
    _assert_indexes_consistent(graph)
    for node in (replicated, partitioned, partitioned.mirror):
        assert not graph.exist(node)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('primitive', ['blocking', 'staging'])
def test_repeated_stage_boundaries(tmp_path, primitive):
    graph, fnodes = _traced_graph(tmp_path / 'repeated')
    first, mid = fnodes[0], fnodes[len(fnodes) // 2]
    positions = graph.multi_index([mid, first, mid])
    assert positions[0] == positions[2] == graph.index(mid)
    assert positions[1] == graph.index(first)

    # a repeated boundary yields an empty stage, which is skipped
    getattr(graph, primitive)([first, first, mid])
    ref_graph, ref_fnodes = _traced_graph(tmp_path / 'reference')
    getattr(ref_graph, primitive)([ref_fnodes[0], ref_fnodes[len(ref_fnodes) // 2]])
    assert [type(n) for n in graph.nodes()] == [type(n) for n in ref_graph.nodes()]
    assert [s.nnodes for s in graph.select(ntype=IRSegment, flatten=False)] == \
        [s.nnodes for s in ref_graph.select(ntype=IRSegment, flatten=False)]
    _assert_indexes_consistent(graph)


@replace_all_device_with('cpu')
def test_segment_insert_rejects_duplicate_node(tmp_path):
    graph, fnodes = _traced_graph(tmp_path)
    nnodes = graph.nnodes
    with pytest.raises(ValueError, match='already in the segment'):
        graph.insert(fnodes[0], 0)
    assert graph.nnodes == nnodes
    _assert_indexes_consistent(graph)


@replace_all_device_with('cpu')
def test_load_graph_dump_without_membership_indexes(tmp_path):
    graph, _ = _traced_graph(tmp_path)
    # graph dumps from versions without the indexes only carry `_nodes`
    del graph._node_set
    del graph._sub_segments
    graph.dump(tmp_path / 'graph.ckp')

    loaded = IRGraph.load(tmp_path / 'graph.ckp')
    _assert_indexes_consistent(loaded)
