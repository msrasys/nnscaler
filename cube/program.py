from typing import List, Tuple
from cube.graph.torch_dtype_mapping import DType2IRDType

from cube.ir.cten import IRCell, IRTensor
from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.ir.operator import IRBpOperation, IRDataOperation

from cube.graph import IRGraph
from cube.graph import parser

from cube.runtime.syndata import CubeDataLoader
from cube.runtime.module import CubeModule
from cube.profiler.timer import print_each_rank

import torch


class Program:

    class __Program:

        def __init__(self):

            self._graph = IRGraph([], [], [], 'program')

    instance = None

    def __init__(self):
        if not Program.instance:
            Program.instance = Program.__Program()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def add_node(self, node: IRCell):
        self.instance._graph.insert(node, self.instance._graph.nnodes)

    def add_nodes(self, nodes: List[IRCell]):
        for node in nodes:
            self.add_node(node)

    def get_graph(self) -> IRGraph:
        return self.instance._graph

    def set_output(self, outputs: List[IRTensor]):
        for otensor in outputs:
            if not isinstance(otensor, IRTensor):
                raise NotImplementedError("Not support for non-tensor graph output")
        self.instance._graph.reset_outputs(len(outputs))
        for idx, otensor in enumerate(outputs):
            self.instance._graph.set_output(idx, otensor)

    def mirror_as_self(self):
        """
        Set mirror as self. This is called when a backward is triggered.
        """
        IRCell.make_pair(self.instance._graph, self.instance._graph)

    def clear(self):
        self.instance._graph = IRGraph([], [], [], 'program')

    def __repr__(self):
        return repr(self.instance._graph)


class SemanticDataLoader:

    def __init__(self, dataloader: CubeDataLoader):
        if not isinstance(dataloader, CubeDataLoader):
            raise TypeError("Expected data loader derived from CubeDataLoader")
        self.dataloader: CubeDataLoader = iter(dataloader)
        dtype_map = DType2IRDType
        self.dtypes = [dtype_map.map(dtype) for dtype in dataloader.dtypes]
        self.shapes = [list(shape) for shape in dataloader.shapes]

    def get_batch_dims(self) -> Tuple[int]:
        return tuple(self.dataloader.batch_dims)

    def get_batch_size(self) -> int:
        return self.dataloader.get_batch_size()

    def set_batch_size(self, bs: int):
        self.dataloader.set_batch_size(bs)
        return

    def __iter__(self):
        return self

    def __next__(self):
        outputs = list()
        for dtype, shape in zip(self.dtypes, self.shapes):
            data = IRFullTensor(
                shape, 'data', requires_grad=False, dtype=dtype
            ).tosub()
            outputs.append(data)

        data_op = IRDataOperation(
            data_num=len(outputs), batch_dims=self.get_batch_dims(),
        )
        for idx, output in enumerate(outputs):
            data_op.set_output(idx, output)

        Program().add_node(data_op)
        if    len(outputs) == 0: return
        elif  len(outputs) == 1: return outputs[0]
        else: return tuple(outputs)


class SemanticModel:

    def __init__(self, model: torch.nn.Module, input_shapes):
        """
        Create semantic model based on AI Scientist description.
        """
        dist = torch.distributed.is_initialized()
        if (not dist) or (dist and torch.distributed.get_rank() == 0):
            self.ir_graph = parser.convert_model(
                model, input_shapes=input_shapes
            )
        else:
            self.ir_graph = None
        self._loaded_module: CubeModule = None

    def get_graph(self):
        return self.ir_graph

    def load_module(self, filename: str, load_content=True):
        import importlib.util
        spec = importlib.util.spec_from_file_location("GenModel", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._loaded_module = module.GenModel().cuda()
        if load_content:
            print_each_rank("> loading parameter content...")
            # TODO: make hardcode ./fullmodel.pt programmable
            self._loaded_module.load_attr_content('./fullmodel.pt')

    def get_gen_module(self):
        return self._loaded_module

    def clear_module(self):
        self._loaded_module = None

    def __call__(self, *args):
        if self._loaded_module:
            return self._loaded_module(*args)
        else:
            return self.ir_graph(*args)