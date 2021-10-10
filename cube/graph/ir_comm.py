from typing import List
from enum import Enum

from cube.graph.ir_cten import IRCell


class IRCommType(Enum):

    Send = 'send'
    Recv = 'recv'
    SendRecv = 'sendrecv'


class IRCommunication(IRCell):
    """
    Communication cell for IRCell
    """
    
    def __init__(self,
                 send_tensors=list(), send_ranks: List[List[int]] = list(),
                 recv_tensors=list(), recv_ranks: List[List[int]] =list()):
        """
        Create a basic send, recv or sendrecv communication node
        """
        if len(send_tensors) != 0 and len(recv_tensors) != 0:
            comm_type = IRCommType.SendRecv
            signature = 'cube.runtime.collectives.sendrecv'
        elif len(send_tensors) != 0 and len(recv_tensors) == 0:
            comm_type = IRCommType.Send
            signature = 'cube.runtime.collectives.send'
        elif len(recv_tensors) != 0 and len(send_tensors) == 0:
            comm_type = IRCommType.Recv
            signature = 'cube.runtime.collectives.recv'
        else:
            raise ValueError(
                "Expected at least one of send_tensors and recv_tensors"
            )

        self.comm_type = comm_type
        self.send_tensors = list()
        self.send_ranks = list()
        self.recv_tensors = list()
        self.recv_ranks = list()

        super().__init__(
            name = comm_type.value,
            signature = signature,
            input_length = len(send_tensors),
            output_length = len(recv_tensors)
        )

        for idx, (tensor, to_device) in enumerate(zip(send_tensors, send_ranks)):
            self.set_input(idx, tensor)
            self.send_tensors.append(self.inputs(idx))
            self.send_ranks.append(to_device)

        for idx, (tensor, from_device) in enumerate(zip(recv_tensors, recv_ranks)):
            self.set_output(idx, tensor)
            self.recv_tensors.append(self.outputs(idx))
            self.recv_ranks.append(from_device)

    def merge(self, other):
        if not isinstance(other, IRCommunication):
            raise RuntimeError("Expected IRCommunication to merge")
        raise NotImplementedError
