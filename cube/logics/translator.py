from cube.ir.operator import IRBpOperation, IRFwOperation, IRDataOperation
from cube.ir.cten import IRCell
from cube.ir.tensor import IRFullTensor, IRSubTensor

from cube.graph.graph import IRGraph

from cube.logics.dataloader import IRDataLoader
from cube.logics import model
from cube.logics.pool import SchedulePool



class LogicTranslator:

    @staticmethod
    def gen_logic_graph(outputs=None) -> IRGraph:
        """
        Generate Training Logic Graph
        """
        nodes = SchedulePool().nodes()
        has_bp = any(n for n in nodes if isinstance(n, IRBpOperation))
        if has_bp:
            assert all(fnode.mirror in nodes for fnode in nodes if isinstance(fnode, IRFwOperation)), \
                "Training requires all nodes have backward."
        else:
            # remove backward nodes if no backward is called
            fnodes = [node for node in nodes if isinstance(node, IRFwOperation)]
            for fnode in fnodes:
                IRCell.make_pair(fnode, None)
            # remove node gradient
            for node in nodes:
                for itensor in node.inputs():
                    if isinstance(itensor, IRSubTensor):
                        itensor.parent.requires_grad = False
                        # ad hoc fix on operators with multiple same input tensors
                        itensor._dirty_grad = True
                for otensor in node.outputs():
                    if isinstance(otensor, IRSubTensor):
                        otensor.parent.requires_grad = False
        graph = IRGraph(nodes, inputs=[], outputs=outputs, module_name='LogicGraph')
        return graph

    @staticmethod
    def load_data(dataloader: IRDataLoader):
        """
        Translator Action: Load data from data loaderw
        """
        if not isinstance(dataloader, IRDataLoader):
            raise TypeError("Expected IRDataLoader")
        outputs = list()
        for dtype, shape in zip(dataloader.dtypes, dataloader.shapes):
            data = IRFullTensor(
                shape, 'data', requires_grad=False, dtype=dtype
            ).tosub()
            outputs.append(data)

        data_op = IRDataOperation(
            data_num=len(outputs), batch_dims=dataloader.get_batch_dims(),
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
        fgraph = model.forward(graph, *args)
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
        if loss.nelement() != 1:
            raise RuntimeError("backward can only perform on the scaler tensor")
        # loss tensor grad should be 1.0
        loss.parent.grad = 1.0
        for node in trace[::-1]:
            SchedulePool().add_node(node.mirror)

    @staticmethod
    def update(optimizer):
        raise NotImplementedError
