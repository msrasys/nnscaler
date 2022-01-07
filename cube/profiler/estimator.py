
from cube.graph.tensor import IRSubTensor, ValueMap
from cube.graph.adapter.adapter import IRAdapter
from cube.graph import IRGraph
from cube.ir.cten import IRCell, IRTensor


class Estimator:

    def __init__(self, graph: IRGraph):
        """
        Estimator for policy use
        """

        self.graph = graph

    def comm_volume(self, device: int) -> int:
        """
        Estimate message recv volume of device id.
        This has no requirement for generating adapters in graph.

        Node that is not assigned to a particular device will not
        be considered.
        """
        volume = 0
        for node in self.graph.nodes():
            if isinstance(node, IRAdapter):
                continue
            if device in node.device:
                volume += self.comm_volume_node(node)
        return volume

    def comm_volume_node(self, node: IRCell) -> int:
        """
        Estimate node message recv volume.
        This has no requirement for generating adapters in graph.
        """
        if node not in self.graph.nodes():
            raise KeyError(f"node {node} not in graph")
        if len(node.device) == 0:
            raise RuntimeError(f"node {node} device is not assigned")
        volume = 0
        for input in node.inputs():
            if isinstance(input, IRSubTensor):
                # reducer
                if input.is_param():
                    if input.grad.valmap != ValueMap(0, 1):
                        volume += input.nele() * (input.grad.valmap.chunk_num - 1)
                # adapter
                else:
                    local, remote = list(), list()
                    for ptensor in input.parent.ptensors:
                        if ptensor.device != input.device:
                            remote.append(ptensor)
                        else:
                            local.append(ptensor)
                    if input in local:
                        continue
                    else:
                        for ptensor in remote:
                            if input.overlap(ptensor):
                                intersection = input.common(ptensor)
                                volume += intersection.nele()
        return volume

    def flops(self) -> int:
        raise NotImplementedError

    def flops_node(self, node: IRCell) -> int:
        raise NotImplementedError
