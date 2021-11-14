"""
Traning Logic Translator

The traning logic first translate the training logic into
Schedule Units, and then add Adapter ScheduleUnit
"""
import torch

from cube.graph.tensor import IRFullTensor, IRSubTensor
from cube.graph.operator import IRDataOperation
import cube.graph.gpass as gpass
from cube.schedule.pool import SchedulePool


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
                data = IRFullTensor(shape=list(data.shape), name='data').tosub()
                data.requires_grad = False
            outputs.append(data)

        data_op = IRDataOperation(
            data_num=len(datas)
        )
        for idx, output in enumerate(outputs):
            data_op.set_output(idx, output)

        SchedulePool().add_node(data_op)
        if    len(outputs) == 0: return
        elif  len(outputs) == 1: return outputs[0]
        else: return tuple(outputs)

    @staticmethod
    def forward(graph, *args):
        """
        Translator Action: forward an IRGraph
        """
        fgraph = gpass.forward(graph, *args)
        for node in fgraph.nodes():
            SchedulePool().add_node(node)
        for output in fgraph.outputs():
            SchedulePool().tape(output, fgraph.nodes())
        outputs = fgraph.outputs()
        if    len(outputs) == 1: return outputs[0]
        elif  len(outputs) == 0: return None
        else: return outputs

    @staticmethod
    def backward(loss: IRSubTensor):
        """
        Translator Action: backward a tensor
        """
        trace = SchedulePool().get_tape(loss)
        if trace is None:
            raise RuntimeError("No forward detected")
        # make grad to 1.0
        if not loss.shape == [1]:
            raise RuntimeError("backward can only perform on the scaler tensor")
        loss.parent.grad = None
        bnode = None
        for node in trace:
            for idx, output in enumerate(node.outputs()):
                if loss.overlap(output):
                    bnode = node.mirror
                    output.grad = None
                    bnode.set_grad(idx, None)
        for node in trace[::-1]:
            SchedulePool().add_node(node.mirror)
