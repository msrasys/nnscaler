from typing import List, Tuple, Optional, Any, Dict
import inspect

from cube.ir.cten import IRCell, IRObject
from cube.ir.tensor import IRFullTensor
from cube.ir.operator import IRBpOperation, IRDataOperation

from cube.graph import IRGraph
from cube.graph import parser

from cube.runtime.syndata import CubeDataLoader
from cube.runtime.module import CubeModule
from cube.runtime.device import DeviceGroup

from cube.utils import load_model

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

    def set_input(self, inputs: Tuple[Any]):
        self.instance._graph.reset_inputs(len(inputs))
        for idx, obj in enumerate(inputs):
            self.instance._graph.set_input(idx, obj)

    def set_output(self, outputs: Tuple[Any]):
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

    def clear(self):
        Program.instance._graph = IRGraph([], [], [], 'program')

    def __repr__(self):
        return repr(self.instance._graph)


class SemanticDataLoader:

    def __init__(self, dataloader: CubeDataLoader):
        if not isinstance(dataloader, CubeDataLoader):
            raise TypeError("Expected data loader derived from CubeDataLoader")
        self.dataloader: CubeDataLoader = iter(dataloader)

    def get_batch_dims(self) -> Tuple[Optional[int]]:
        return tuple(self.dataloader.get_batch_dims())

    def get_batch_size(self) -> int:
        return self.dataloader.get_batch_size()

    def set_batch_size(self, bs: int):
        self.dataloader.set_batch_size(bs)
        return

    def get_runtime_sample(self):
        return next(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        def generate_output(sample):
            """Support complex of types: List, Tuple, torch.Tensor, object"""
            if isinstance(sample, tuple):
                return tuple(generate_output(t) for t in sample)
            if isinstance(sample, list):
                return list(generate_output(t) for t in sample)
            if isinstance(sample, dict):
                assert all(isinstance(key, (str, int)) for key in sample.keys())
                return {key:generate_output(val) for key, val in sample.items()}
            if isinstance(sample, set):
                return {generate_output(t) for t in sample}
            if isinstance(sample, torch.Tensor):
                tensor = IRFullTensor(list(sample.shape), 'data', dtype=sample.dtype).tosub()
                tensor._value = sample
                return tensor
            else:
                return IRObject('data', value=sample)

        sample = next(self.dataloader)
        outputs = generate_output(sample)

        # create dataloader
        if isinstance(outputs, (tuple, list)):
            data_num = len(outputs)
        elif isinstance(outputs, dict):
            data_num = len(outputs.keys())
        else:
            data_num = 1

        data_op = IRDataOperation(data_num=data_num, batch_dims=self.get_batch_dims())
        if not isinstance(outputs, tuple):
            data_op.set_output(0, outputs)
        else:
            for idx, t in enumerate(outputs):
                data_op.set_output(idx, t)
        Program().add_node(data_op)
        return outputs


class SemanticModel:

    def __init__(self, model: Optional[torch.nn.Module],
                 save_content: bool = True,
                 dynamic_shape: bool = False):
        """
        Create semantic model based on AI Scientist description.

        Args:
            model (Optional[torch.nn.Module]):
                single-device model description, only required for rank 0
            save_content (bool):
                whether to save the content of model and load it into generated model. Default True.
            dynamic_shape (bool):
                whether to use dynamic shape. Default False.
        """
        if DeviceGroup().local_rank == 0:
            assert isinstance(model, torch.nn.Module), f"device of local_rank == 0 must provide model"
        self.model = model
        self._dummy_input: Dict[str, Any] = None
        self._ir_graph = None
        self._loaded_module: CubeModule = None
        # parser configuration
        self.save_content: bool = save_content
        self.dynamic_shape: bool = dynamic_shape

    @property
    def dummy_input(self) -> Any:
        """Get dummy real-tensor input from on CPU"""
        return self._dummy_input

    @dummy_input.setter
    def dummy_input(self, val):

        def complex(val: Any):
            """Complex to CPU"""
            if isinstance(val, tuple):
                return tuple(complex(t) for t in val)
            if isinstance(val, list):
                return list(complex(t) for t in val)
            if isinstance(val, dict):
                return {complex(key):complex(val) for key, val in val.items()}
            if isinstance(val, set):
                return {complex(t) for t in val}
            if isinstance(val, torch.Tensor):
                return val.cpu()
            return val

        self._dummy_input = complex(val)

    def get_graph(self):
        return self._ir_graph

    def load_module(self, filename: str):
        """Load module from file."""
        self._loaded_module = load_model(filename, self.save_content)

    def get_gen_module(self) -> Optional[torch.nn.Module]:
        return self._loaded_module

    def clear_module(self):
        self._loaded_module = None

    def __call__(self, *args):
        """Forward the semantic model.

        This will parse the model into cube graph.

        Args:
            *args: input IRObjects

        Returns:
            graph outputs with IRObjects
        """
        assert self._ir_graph is None, \
            f"multiple forward on a semantic model is not allowed"
        if DeviceGroup().local_rank == 0:
            # collect dummy input
            if self.dummy_input is None:
                dummy_input = {}
                sig = inspect.signature(self.model.forward)
                # note: we don't support model forward arguments having complex data stucture
                # that contains tensor
                for name, arg in zip(sig.parameters.keys(), args):
                    if isinstance(arg, IRObject):
                        value = arg.value
                        arg._value = None  # remove tensor reference to release memory
                    else:
                        value = arg
                    dummy_input[str(name)] = value
                self.dummy_input = dummy_input
            # parse graph
            self._ir_graph = parser.convert_model(
                self.model,
                dummy_input=self.dummy_input,
                attr_savedir='./',
                dynamic_shape=self.dynamic_shape
            )
            return self._ir_graph(*args)
