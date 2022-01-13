"""
This operator class is highly inspired by eniops.
"""
import enum
import string
from typing import List, Optional, Tuple

from cube.ir.cten import IRTensor
from cube.graph.operator.operator import IRFwOperation
from cube.algorithm.factory import DistAlgorithmFactory


class EinDim:

    class ReduceType(enum.Enum):
        Stay = 0  # the dim is not allowed to be split
        Sum = 1

    def __init__(self, name: str, reduce=None):
        if not (str.isidentifier(name) or str.isnumeric(name) or name == '*'):
            raise ValueError("Einstein Axis name should be identifier")
        self.name: str = name
        self.reduce: Optional[EinDim.ReduceType] = reduce

    def __eq__(self, other):
        if isinstance(other, EinDim):
            if other.name == self.name:
                return True
        return False

    def is_reduce(self):
        return self.reduce == EinDim.ReduceType.Sum

    def __repr__(self):
        return self.name if not self.is_reduce() else self.name + "+"


class IREinops(IRFwOperation):
    """
    Einstein expression on operators like reshape, view, permute, reduce.
    """
    def __init__(self, name: str, signature: str, input_length: int, output_length:int):
        super().__init__(name, signature, input_length, output_length)
        self._ieins = [list() for _ in range(input_length)]
        self._oeins = [list() for _ in range(output_length)]

    def new(self, inputs, outputs, **kwargs):
        """
        Create a new same operation given the inputs and outputs

        Each operator needs to implement this.
        """
        raise NotImplementedError

    def make_expression(self):
        """
        Set einstein-like expression assuming input shapes are given.

        Each operator needs to implement this.
        """
        raise NotImplementedError
    
    def infer_shape(self):
        """
        Infer output value shape
        """
        for input in self.inputs():
            if isinstance(input, IRTensor) and input.shape is None:
                return False
        self.make_expression()
        # check expression
        for input, ein_dims in zip(self.inputs(), self._ieins):
            if len(ein_dims) == 0 or ein_dims is None:
                if isinstance(input, IRTensor):
                    raise RuntimeError(f"{self}: {input} has no ein-dims but is a tensor")
            if len(ein_dims) != 0:
                if not isinstance(input, IRTensor):
                    raise RuntimeError(f"{self}: {input} has ein-dims but is not a tensor")
                if len(input.shape) != len(ein_dims):
                    raise RuntimeError(f"input tensor ndims ({len(input.shape)}) != ein-dims ({len(ein_dims)})")
        # figure output shape
        for oidx in range(len(self._outputs)):
            output_shape = list()
            for oein in self._oeins[oidx]:
                if str.isdecimal(oein.name):
                    output_shape.append(int(oein.name))
                    continue
                for iidx in range(len(self._inputs)):
                    if oein in self._ieins[iidx]:
                        input = self.inputs(iidx)
                        dim = self._ieins[iidx].index(oein)
                        output_shape.append(input.shape[dim])
                        break
            self.outputs(oidx).shape = output_shape
        return True

    def set_input_ein(self, input_index: int, dims: List[EinDim]):
        """
        Set input einstein axis at input index
        """
        if not all([isinstance(dim, EinDim) for dim in dims]):
            raise TypeError("Expected Tuple[EinDim]")
        self._ieins[input_index] = tuple(dims)

    def set_output_ein(self, output_index: int, dims: Tuple[EinDim]):
        """
        Set output einstein axis at output index
        """
        if not all([isinstance(dim, EinDim) for dim in dims]):
            raise TypeError("Expected Tuple[EinDim]")
        self._oeins[output_index] = tuple(dims)

    def einexpr(self) -> str:
        inputs = list()
        outputs = list()
        for iein in self._ieins:
            inputs.append(' '.join([repr(ein) for ein in iein]))
        for oein in self._oeins:
            outputs.append(' '.join([repr(ein) for ein in oein]))
        return ', '.join(inputs) + ' -> ' + ', '.join(outputs)

    def algorithms(self, tag: Optional[str] = None):
        factory = DistAlgorithmFactory()
        if tag is None:
            algos = list()
            if factory.exist(type(self)):
                algos += [template(self) for template in factory.algorithms(type(self))]
            if factory.exist(IREinops):
                algos += [template(self) for template in factory.algorithms(IREinops)]
            return algos
        else:
            if factory.exist(type(self), tag):
                template = factory.algorithms(type(self), tag)
                return template(self)
            if factory.exist(IREinops, tag):
                template = factory.algorithms(IREinops, tag)
                return template(self)
            return None

    def parse(self, expr: str) -> Tuple[List[List[EinDim]], List[List[EinDim]]]:
        """
        parse string like:
            b m k, b k n -> b m n
        """
        if not isinstance(expr, str):
            raise TypeError("Expected string")
        # split to inputs and outputs
        if expr.count('->') != 1:
            raise ValueError("string must contain one ->")
        # split to each tensor
        input, output = expr.split('->')
        inputs = input.split(',')
        outputs = output.split(',')
        inputs = [[dim for dim in input.split(' ') if len(dim) != 0] for input in inputs]
        outputs = [[dim for dim in output.split(' ') if len(dim) != 0] for output in outputs]
        # parse each tensor
        input_axises = list()
        for input in inputs:
            axises = list()
            for dim in input:
                reduce = EinDim.ReduceType.Sum if dim not in output else None
                # a fixed numeric value indicates the axis is not splittable
                if str.isnumeric(dim):
                    reduce = EinDim.ReduceType.Stay
                axises.append(EinDim(dim, reduce))
            input_axises.append(axises)
        output_axises = list()
        for output in outputs:
            axises = [EinDim(dim) for dim in output]
            output_axises.append(axises)
        return input_axises, output_axises
