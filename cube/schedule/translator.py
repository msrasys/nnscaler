"""
Traning Logic Translator

The traning logic first translate the training logic into
Schedule Units, and then add Adapter ScheduleUnit
"""
from typing import List
import torch

from cube.ir.cten import IRCell, IRTensor
from cube.graph.tensor import IRFullTensor
from cube.graph.comm import IRCommunication
from cube.schedule.su import SUType, ScheduleUnit
from cube.schedule.pool import SchedulePool
from cube.schedule.sugraph import SUGraph


class IRDataLoader:

    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        return LogicTranslator.load_data(self)


class LogicTranslator:

    @staticmethod
    def load_data(dataloader: IRDataLoader):
        """
        Translator Action: Load data from data loaderw
        """
        datas = next(dataloader.dataloader)
        if not isinstance(datas, tuple):
            datas = (datas,)

        # data IRTensor
        outputs = list()
        for data in datas:
            if torch.is_tensor(data):
                data = IRFullTensor(shape=list(data.shape), name='data')
                data.requires_grad = False
            outputs.append(data)

        cell = IRCell(
            name='dataloader',
            signature='dataloader.__next__',
            input_length=0,
            output_length=len(datas)
        )
        for idx, output in enumerate(outputs):
            cell.set_output(idx, output)

        su = ScheduleUnit([cell], stype=SUType.Dataloader, name='DataLoader')
        SchedulePool().add_su(su)

        if    len(outputs) == 0: return
        elif  len(outputs) == 1: return outputs[0]
        else: return tuple(outputs)

    @staticmethod
    def forward(graph, *args):
        """
        Translator Action: forward an IRGraph
        """

        def _forward(graph, stype, *args):
            # set input
            for input, arg in zip(graph.inputs(), args):
                graph._replace_tensor(input, arg)
            # translate to SUs
            sus = list()
            for node in graph.nodes():
                su = ScheduleUnit([node], stype, name=str(stype))
                sus.append(su)
            return sus

        # forward graph
        fgraph = graph.copy(reverse=False)
        # backward graph
        bgraph = graph.copy(reverse=True)
        bgraph.tag = 'backward'

        # translate forward graph
        fsus = _forward(fgraph, SUType.Forward, *args)
        bsus = _forward(bgraph, SUType.Backward, *(fgraph.outputs()))
        for fsu, bsu in zip(fsus, bsus[::-1]):
            fsu.set_mirror(bsu)
            bsu.set_mirror(fsu)
            SchedulePool().add_su(fsu)
        
        for output in fgraph.outputs():
            output.set_trace(fsus)

        outputs = fgraph.outputs()
        if    len(outputs) == 1: return outputs[0]
        elif  len(outputs) == 0: return None
        else: return outputs

    @staticmethod
    def backward(tensor: IRTensor):
        """
        Translator Action: backward a tensor
        """
        if tensor.trace is None:
            return
        for fsu in tensor.trace[::-1]:
            SchedulePool().add_su(fsu.mirror)

    @staticmethod
    def gen_adapter(sus: List[ScheduleUnit]) -> List[ScheduleUnit]:
        """
        Each computation SU has adapters for its inputs
        """
        sugraph = SUGraph(sus)

        # clear adapters
        for su in sugraph.sus():
            su._clear_adapters()

        for su in sugraph.sus():
            for in_idx, input in enumerate(su.inputs()):
                if not isinstance(input, IRTensor):
                    continue
                pre_sus = su.predecessors(in_idx)
                for pre_su in pre_sus:
                    for out_idx, output in enumerate(pre_su.outputs()):
                        if output.overlap(input):
                            sub_tensor = output.common(input)
                            send_op = IRCommunication(
                                send_tensors=[sub_tensor],
                                send_ranks = [-1]
                            )
                            recv_op = IRCommunication(
                                recv_tensors=[sub_tensor],
                                recv_ranks = [-1]
                            )
                            send_op.pair(recv_op)
                            send_su = ScheduleUnit([send_op], SUType.Adapter, name='send')
                            recv_su = ScheduleUnit([recv_op], SUType.Adapter, name='recv')
                            su._add_in_adapter(in_idx, send_su, recv_su)
                            pre_su._add_out_adapter(out_idx, send_su, recv_su)

        sus_with_adapter = list()
        for su in sus:
            for idx in range(len(su.inputs())):
                send_adapters, recv_adapters = su.in_adapters(idx)
                for send_su, recv_su in zip(send_adapters, recv_adapters):
                    sus_with_adapter.append(send_su)
                    sus_with_adapter.append(recv_su)
            sus_with_adapter.append(su)
        return sus_with_adapter
