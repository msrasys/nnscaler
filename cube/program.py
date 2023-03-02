from typing import List, Tuple, Optional

from cube.ir.cten import IRCell, IRTensor
from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.ir.operator import IRBpOperation, IRDataOperation

from cube.graph import IRGraph
from cube.graph import parser
from cube.graph.parser.mapping import DType2IRDType

from cube.runtime.syndata import CubeDataLoader
from cube.runtime.module import CubeModule
from cube.runtime.device import DeviceGroup
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
    
    def finalize(self):
        """
        Close the recording of program.
        If the program doesn't do backward, set all tensors with requires_grad=False.
        """
        graph = self.get_graph()
        if not any(isinstance(node, IRBpOperation) for node in graph.nodes()):
            for ftensor in graph.full_tensors():
                ftensor.requires_grad = False
        

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

    def __init__(self, model: Optional[torch.nn.Module], input_shapes=None, dummy_input=None):
        """
        Create semantic model based on AI Scientist description.

        @param model Optional[torch.nn.Module]: Model description. Each device of local_rank == 0 needs to provide.
        @param input_shapes Any: to compatable with previous interface. No more need.
        """
        if DeviceGroup().local_rank == 0:
            assert isinstance(model, torch.nn.Module), f"device of local_rank == 0 must provide model"
        self.model = model
        self.input_shapes = None
        self.dummy_input = dummy_input
        self.ir_graph = None
        self._loaded_module: CubeModule = None
        self._save_content = True

    @property
    def save_content(self) -> bool:
        return self._save_content
    
    @save_content.setter
    def save_content(self, val: bool):
        self._save_content = val

    def get_graph(self):
        return self.ir_graph

    def load_module(self, filename: str):
        import importlib.util
        spec = importlib.util.spec_from_file_location("GenModel", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._loaded_module = module.GenModel().cuda()
        if self.save_content:
            print_each_rank("> loading parameter content...")
            # TODO: make hardcode ./fullmodel.pt programmable
            self._loaded_module.load_attr_content('./fullmodel.pt')

    def get_gen_module(self) -> Optional[torch.nn.Module]:
        return self._loaded_module

    def clear_module(self):
        self._loaded_module = None

    def __call__(self, *args):
        """
        Forward the model.
        This will trigger torch.jit.script to parse the model.
        """
        if self._loaded_module:
            return self._loaded_module(*args)
        else:
            assert all(isinstance(t, IRSubTensor) for t in args), f"Only support tensors as model inputs"
            input_shapes = [tuple(t.shape) for t in args]
            if DeviceGroup().local_rank == 0:
                if self.ir_graph is None:
                    self.ir_graph = parser.convert_model(
                        self.model, input_shapes=input_shapes, dummy_input=self.dummy_input, save_content=self.save_content
                    )
                    self.input_shapes = input_shapes
                else:
                    assert tuple(self.input_shapes) == tuple(input_shapes), \
                        f"Multiple forwarding of a same model, which require input shapes to be same."
            return self.ir_graph(*args)
