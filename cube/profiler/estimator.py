
from cube.ir.operator import IRBpOperation, IRFwOperation
from cube.ir.tensor import IRSubTensor, ValueMap
from cube.ir.adapter import IRAdapter
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

        Note for intermediate tensor communication, the estimated
        communication volume is:
            Volume = 0 if local produced tensor can covor all the needed region.
                       else N#(remote produced overlapping region)
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
                    # check local producer
                    local_cover = False
                    for ptensor in local:
                        if input.overlap(ptensor):
                            intersection = input.common(ptensor)
                            if intersection == input:
                                local_cover = True
                                break
                    if local_cover:
                        continue
                    # check remote producer
                    remote_producer_volume = 0
                    for ptensor in remote:
                        if input.overlap(ptensor):
                            intersection = input.common(ptensor)
                            remote_producer_volume += intersection.nele()
                    # check remote consumer
                    # TODO: need to check if all consumers can be
                    # merged to input
                    remote_consumer_volume = None
                    index = input.parent.consumers.index(node)
                    for ctensor in input.parent.ctensors[:index]:
                        if input.overlap(ctensor):
                            if remote_consumer_volume is None:
                                remote_consumer_volume = 0
                            intersection = input.common(ctensor)
                            remote_consumer_volume += intersection.nele()
                        if intersection == input:
                            break
                    if remote_consumer_volume is None:
                        volume += remote_producer_volume
                    else:
                        volume += min(remote_consumer_volume, remote_producer_volume)
        # debug info
        # if isinstance(node, IRFwOperation):
        #     print(f'fw{node._id}-{node.device}-{node.name}: {volume}')
        # elif isinstance(node, IRBpOperation):
        #     print(f'bw{node._id}(fw{node.mirror._id}): {volume}')
        # else:
        #     print(f'cell{node._id}-{node.device}-{node.name}: {volume}')
        return volume

    def flops(self) -> int:
        raise NotImplementedError

    def flops_node(self, node: IRCell) -> int:
        raise NotImplementedError
