from typing import Tuple

from cube.graph.ir_comm import IRCommunication
from cube.graph.ir_graph import IRGraph


class Adapter:

    @staticmethod
    def adapt(graph: IRGraph) -> IRGraph:
        for src_node in graph.nodes():
            for out_idx, tensor in enumerate(src_node.outputs()):
                for dst_node in src_node.successors(out_idx):
                    if set(src_node.device) != set(dst_node.device):
                        from_rank = src_node.device
                        to_rank = dst_node.device
                        from_rank, to_rank = from_rank, to_rank
                        #TODO check if it is a tensor
                        send_node, recv_node = Adapter.create_tensor_move(
                            tensor = tensor,
                            from_rank = from_rank,
                            to_rank = to_rank
                        )
                        graph.insert(send_node, src_node=src_node)
                        graph.insert(recv_node, dst_node=dst_node,
                                     replaced_tensor=tensor)
        return graph

    @staticmethod
    def create_tensor_move(tensor, from_rank: int, to_rank: int) -> Tuple[IRCommunication, IRCommunication]:
        # send node
        ir_send_node = IRCommunication(
            send_tensors = [tensor],
            send_ranks   = [to_rank]
        )
        ir_send_node.device = from_rank
        # recv node
        ir_recv_node = IRCommunication(
            recv_tensors = [tensor],
            recv_ranks   = [from_rank]
        )
        ir_recv_node.device = to_rank
        return ir_send_node, ir_recv_node

