from typing import Dict, List, Tuple

from cube.graph.graph import IRGraph
from cube.graph.tensor import IRSubTensor, ValueMap
from cube.graph.adapter.adapter import IRAdapter, IRWeightReducer
from cube.graph.operator.operator import IRBpOperation, IRFwOperation


class AdapterGener:

    @staticmethod
    def gen(graph: IRGraph, eager=True) -> IRGraph:
        """
        Generate tensor adapter for both intermediate tensors and weights

        Args:
            graph: IRGraph.
            eager (Boolean):
                if True,
                    each adapter will be inserted right after it's ready to execute.
                if False (i.e., lazy),
                    each adatper will be inserted right before the tensor needs it.
            Note weight reducers are always append to last.
        Returns:
            graph (IRGraph)
        """
        # update the gradient before generate adapter
        for node in graph.nodes():
            if isinstance(node, IRBpOperation):
                node.update()
        graph = AdapterGener.gen_activation_adapter(graph, eager)
        graph = AdapterGener.gen_weight_reducer(graph)
        return graph

    @staticmethod
    def gen_activation_adapter(graph: IRGraph, eager=True) -> IRGraph:
        all_adapters = list()
        # generate adapter for non-weight values
        for node in graph.nodes():
            if isinstance(node, IRFwOperation):
                for input in node.inputs():
                    if not isinstance(input, IRSubTensor):
                        continue
                    # skip parameter
                    if input.is_param():
                        continue
                    adapter = IRAdapter.gen(input)
                    if not adapter.is_identity():
                        all_adapters.append(adapter)
                        idx = graph.nodes().index(node)
                        graph._nodes.insert(idx, adapter)
            if isinstance(node, IRBpOperation):
                for grad in node.inputs():
                    if not isinstance(grad, IRSubTensor):
                        continue
                    adapter = IRAdapter.gen(grad)
                    if not adapter.is_identity():
                        all_adapters.append(adapter)
                        idx = graph.nodes().index(node)
                        graph._nodes.insert(idx, adapter)
        graph.reset_dependency()
        if eager:
            seq = graph.nodes()
            for adapter in all_adapters:
                seq.remove(adapter)
            graph.partial_set_order(seq, eager=True)
        return graph


    @staticmethod
    def gen_weight_reducer(graph: IRGraph) -> IRGraph:
        # step 1: get weight and gradient
        # weights: Dict[weight_id: int, IRSubTensor]
        # grads  : Dict[weight_id: int, Dict[device: int, List[grad: IRSubTensor]]]
        grads = dict()
        weights = dict()
        for fnode in graph.nodes():
            if not isinstance(fnode, IRFwOperation):
                continue
            devid = fnode.device[0]
            for input in fnode.inputs():
                if isinstance(input, IRSubTensor) and input.is_param():
                    grad = input.grad
                    # nothing to sync
                    if grad.valmap == ValueMap(0, 1):
                        continue
                    if input._id not in grads:
                        grads[input._id] = dict()
                        weights[input._id] = input
                    if devid not in grads[input._id]:
                        grads[input._id][devid] = list()
                    if grad in grads[input._id][devid]:
                        raise RuntimeError(
                            "Find two same gradient (not expected). "
                            "This is usually due to replicated node assigned to same device. "
                            f"\nCheck node:\n\t{fnode}"
                        )
                    grads[input._id][devid].append(grad)
        # step 2: generate reducers.
        # reducers: tuple(ranks): List[weight]
        reducers: Dict[Tuple[int], List[IRSubTensor]] = dict()
        for wid in grads:
            ranks = list(grads[wid].keys())
            ranks.sort()
            ranks = tuple(ranks)  # ranks are used for group
            if len(ranks) == 1:
                continue
            if ranks not in reducers:
                reducers[ranks] = list()
            reducers[ranks].append(weights[wid])
        # generate reducer for each rank
        for ranks in reducers:
            weights = reducers[ranks]
            opt_op = IRWeightReducer(weights)
            opt_op.device = list(ranks)
            graph._nodes.append(opt_op)
        return graph