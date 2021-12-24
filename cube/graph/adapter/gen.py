
from cube.graph.graph import IRGraph
from cube.graph.adapter.adapter import IRAdapter
from cube.graph.operator.operator import IRBpOperation, IRFwOperation
from cube.graph.tensor import IRSubTensor


class AdapterGener:

    @staticmethod
    def gen(graph: IRGraph) -> IRGraph:
        for node in graph.nodes():
            if isinstance(node, IRFwOperation):
                for input in node.inputs():
                    if not isinstance(input, IRSubTensor):
                        continue
                    # skip parameter
                    if input.is_param():
                        continue
                    adapter = IRAdapter(input)
                    if not adapter.is_identity():
                        idx = graph.nodes().index(node)
                        graph._nodes.insert(idx, adapter)
            if isinstance(node, IRBpOperation):
                for grad in node.grads():
                    if not isinstance(grad, IRSubTensor):
                        continue
                    # skip parameter
                    adapter = IRAdapter(grad)
                    if not adapter.is_identity():
                        idx = graph.nodes().index(node)
                        graph._nodes.insert(idx, adapter)
        return graph
