"""
This is the description interface to describe the 
segmentation requirement (restrictions).

The description includes two parts:

    1). restriction description on tensor segementation

    2). Translation procedure to translate such a restriction
        to the real segmentation on given logical tensor.
"""

from cube.tensor.segment import Segment
from cube.tensor.indices import TileIndices

import z3


class BaseOutline:
    """
    Basic class for declare outline

    To setup an attribute (requirement), use `inst_baseoutline.attribute_name = val`
    """
    def __init__(self, solver, tensor):
        if not isinstance(solver, z3.z3.Solver):
            raise TypeError("Expected solver to be an z3.z3.Solver")
        self.solver = solver
        self.shape = tensor.shape
        self.attributes = list()

    def get_attributes(self):
        return self.attributes

    def add_field(self, **kwargs):
        """
        Add a config field to current instance

        Usage: self.add_field(key=val):

        key is the name for the config attribute, val is the choices

        val type:
            list[int]: the key can only be the options from the val;
            int: the key can only be the val;
            range: the key can only be the val in the range;
            None: the key can be any integers
            z3.z3.ArithRef: the key is aligned with another attribute
        """
        for key in kwargs:
            if key in self.__dict__:
                raise RuntimeError("{} already in config field".format(key))
            val = kwargs[key]
            if isinstance(val, list):
                if not all([isinstance(arg, int) for arg in val]):
                    raise TypeError("{} only supports list[int] choices".format(key))
                self.__dict__[key] = z3.Int(key)
                self.attributes.append(self.__dict__[key])
                self.solver.add(z3.Or([self.__dict__[key] == v for v in val]))
            elif isinstance(val, int):
                self.__dict__[key] = z3.Int(str(id(self))+key)
                self.attributes.append(self.__dict__[key])
                self.solver.add(self.__dict__[key] == val)
            elif isinstance(val, range):
                self.__dict__[key] = z3.Int(str(id(self))+key)
                self.attributes.append(self.__dict__[key])
                self.solver.add(self.__dict__[key] >= val[0])
                raise NotImplementedError
            elif val is None:
                self.__dict__[key] = z3.Int(str(id(self))+key)
                self.attributes.append(self.__dict__[key])
            elif isinstance(val, z3.z3.ArithRef):
                self.__dict__[key] = val
            else:
                raise TypeError("{} can only be int, list[int], z3.Int()".format(key))
    
    def add_constraint(self, constraint):
        """
        Add a constraint
        """
        if not isinstance(constraint, z3.z3.BoolRef):
            raise TypeError("Expected z3.z3.BoolRef constraints")
        self.solver.add(constraint)

    def remove_config(self, config):
        if not isinstance(config, z3.z3.ModelRef):
            raise TypeError("Expected config from z3 model()")
        self.solver.add(z3.Or([z3.Not(attr == config[attr]) for attr in self.attributes]))

    def interpret(self, tensor, config):
        """
        Interpret to a list of segment based on the logical tensor and config

        Args:
            tensor (LogicalTensor)
            config (z3.z3.ModelRef)

        Returns:
            list[Segment]
        """
        raise NotImplementedError


class Full(BaseOutline):

    def __init__(self, solver, tensor):
        super().__init__(solver, tensor)

    def interpret(self, tensor, config):
        if not isinstance(config, z3.z3.ModelRef):
            raise TypeError("Expected config from z3 model()")
        indices = TileIndices([0] * len(self.shape), self.shape)
        segment = tensor.select(indices, None, self.shape)
        return [segment]


class SplitAxis(BaseOutline):

    def __init__(self, solver, tensor, axis, chunk_num, overlap):
        """
        Split the logical tensor uniformly in `axis` dimension

        TODO: support split axis with non-uniform chunk size

        shape: list / tuple int
            shape of input logical tensor
        axis: int
            which axis to split
        chunk_num: options (iterable int) / None / int:
            how many segments to produce
        overlap: options (iterable int) / int:
            overlap size on the boundary
        """
        if not isinstance(axis, int):
            raise RuntimeError("Expected axis to be an integer")
        super().__init__(solver, tensor)

        self.axis = axis
        
        self.add_field(overlap=overlap)
        self.add_constraint(self.overlap >= 0)

        self.add_field(chunk_num=chunk_num)
        self.add_constraint(self.chunk_num >= 0)

        # TODO: change to array to adapt with non-uniform cases
        self.add_field(chunk_size=None)
        
        # setup constraints
        total_size = self.shape[self.axis]
        self.add_constraint(
            self.chunk_num * self.chunk_size - self.overlap * (self.chunk_num - 1) == total_size
        )

    def interpret(self, tensor, config):
        if tuple(tensor.shape) != tuple(self.shape):
            raise RuntimeError("The logical tensor's shape doesn't match")
        if not isinstance(config, z3.z3.ModelRef):
            raise TypeError("Expected config from z3 model()")
        chunk_num = config[self.chunk_num].as_long()
        chunk_size = config[self.chunk_size].as_long()
        shape = list(self.shape)
        shape[self.axis] = chunk_size
        anchor = [0] * len(shape)
        segments = list()
        for cid in range(chunk_num):
            indices = TileIndices(anchor, shape)
            segment = tensor.select(indices, None, shape)
            segments.append(segment)
            anchor[self.axis] += shape[self.axis]
        return segments


class SplitValue(BaseOutline):

    def __init__(self, solver, tensor, chunk_num, val_op):
        """
        Split the whole tensor in value dimension.

        Each segment shape will be same with logical tensor.

        Each segment value will be modified by `val_op`.
        """
        super().__init__(solver, tensor)
        self.add_field(chunk_num=chunk_num)
        self.add_constraint(self.chunk_num >= 1)
        self.val_op = val_op

    def interpret(self, tensor, config):
        if tuple(tensor.shape) != tuple(self.shape):
            raise RuntimeError("The logical tensor's shape doesn't match")
        chunk_num = config[self.chunk_num].as_long()
        segments = list()
        for cid in range(chunk_num):
            indices = TileIndices([0] * len(self.shape), self.shape)
            segment = tensor.select(indices, self.val_op, self.shape)
            segments.append(segment)
        for segment in segments:
            segment.val_op_segs.append(segments)
        return segments
