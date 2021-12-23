
from cube.graph.graph import IRGraph
from cube.graph.adapter.adapter import IRAdapter
from cube.graph.operator.operator import IRFwOperation
from cube.ir.cten import IRTensor


class AdapterGener:

    @staticmethod
    def gen(graph: IRGraph) -> IRGraph:
        for node in graph.nodes():
            if not isinstance(node, IRFwOperation):
                continue
            # adapter for input
            for input in node.inputs():
                if not isinstance(input, IRTensor):
                    continue
                # skip parameter
                if input.is_param():
                    continue
                adapter = IRAdapter(input)
                if not adapter.is_identity():
                    idx = graph.nodes().index(node)
                    graph._nodes.insert(idx, adapter)
        return graph
