from typing import List
from enum import Enum

from cube.ir.cten import IRCell, IRTensor


class IRCollType(Enum):

    AllReduce = 'all_reduce'
    AllGather = 'all_gather'
    ReduceScatter = 'reduce_scatter'
    Broadcast = 'broadcast'


class IRCollectives(IRCell):
    """
    Collective cell for IRCell
    """

    def __init__(self, inputs: List[IRTensor], outputs: List[IRTensor],
                 ranks: List[int], colltype: IRCollType):

        if not isinstance(colltype, IRCollType):
            raise TypeError("colltype Expected IRCollType")
        if not all([isinstance(rank, int) for rank in ranks]):
            raise TypeError("ranks should be List[int]")

        self.comm_type = colltype
        if colltype == IRCollType.AllReduce:
            signature = 'cube.runtime.collectives.all_reduce'
        if colltype == IRCollType.AllGather:
            signature = 'cube.runtime.collectives.all_gather'
        if colltype == IRCollType.ReduceScatter:
            signature = 'cube.runtime.collectives.reduce_scatter'
        if colltype == IRCollType.Broadcast:
            signature = 'cube.runtime.collectives.broadcast'

        self.ranks = ranks

        super().__init__(
            name = colltype.value,
            signature = signature,
            input_length = len(inputs),
            output_length = len(outputs)
        )
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        for idx, output in enumerate(outputs):
            self.set_output(idx, output)
