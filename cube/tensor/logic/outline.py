"""
This is the description interface to describe the 
segmentation requirement (restrictions).

The description includes two parts:

    1). restriction description on tensor segementation

    2). Translation procedure in runtime to translate such a restriction
        to the real segmentation on given logical tensor shape.
"""

from cube.tensor.segment import Segment
from cube.tensor.indices import TileIndices

import z3


# interface to setup restrictions on the segmentation

class BaseOutline:
    """
    Basic class for declare outline

    To setup an attribute (requirement), use `inst_baseoutline.attribute_name = val`
    """
    def __init__(self, solver, shape):
        super().__init__()
        self.solver = solver
        self.shape = shape
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
                self.solver.add(z3.Or([self.__dict__[key] == val for val in val]))
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
    
    def remove_config(self, config):
        if not isinstance(config, z3.z3.ModelRef):
            raise TypeError("Expected config from z3 model()")
        self.solver.add(z3.Or([z3.Not(attr == config[attr]) for attr in self.attributes]))

    def interpret(self, logical_tensor, config):
        raise NotImplementedError


class Full(BaseOutline):

    def __init__(self, solver, shape):
        super().__init__(solver, shape)

    def interpret(self, logical_tensor, config):
        if not isinstance(config, z3.z3.ModelRef):
            raise TypeError("Expected config from z3 model()")
        indices = TileIndices([0] * len(self.shape), self.shape)
        segment = logical_tensor.select(indices, None, self.shape)
        return [segment]


class SplitAxis(BaseOutline):

    def __init__(self, solver, shape, axis, chunk_num, overlap):
        """
        Split the logical tensor spatially in `axis` dimension

        TODO: support split axis with non-uniform chunk size

        shape: list / tuple int
            shape of input logical tensor
        axis: int
            which axis to split
        chunk_num: options (iterable int) / None / int:
            how many segments to produce
        uniform: Boolean
            whether restrict to uniform split
        overlap: options (iterable int) / int:
            overlap size on the boundary
        """
        if not isinstance(axis, int):
            raise RuntimeError("Expected axis to be an integer")

        super().__init__(solver, shape)
        self.axis = axis
        
        self.add_field(overlap=overlap)
        self.solver.add(self.overlap >= 0)

        self.add_field(chunk_num=chunk_num)
        self.solver.add(self.chunk_num >= 0)

        # TODO: change to array to adapt with non-uniform cases
        self.add_field(chunk_size=None)
        
        # setup constraints
        total_size = self.shape[self.axis]
        self.solver.add(self.chunk_num * self.chunk_size - self.overlap * (self.chunk_num - 1) == total_size)

    def interpret(self, logical_tensor, config):
        """
        Get segments from config

        Args:
            logical_tensor (LogicalTensor): 
                the logical tensor
            config:
                Config searched by model output

        """
        if tuple(logical_tensor.shape) != tuple(self.shape):
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
            segment = logical_tensor.select(indices, None, shape)
            segments.append(segment)
            anchor[self.axis] += shape[self.axis]
        return segments


class SplitValue(BaseOutline):

    def __init__(self, solver, shape, chunk_num, val_map_op):
        """
        Split the whole tensor in value dimension.

        Each segment shape will be same with logical tensor.

        Each segment value will be modified by `val_map_op`.
        """
        if not callable(val_map_op):
            raise TypeError("Expected val_map_op a callable function")
        super().__init__(solver, shape)
        self.add_field(chunk_num=chunk_num)
        self.solver.add(self.chunk_num >= 1)
        self.val_map_op = val_map_op

    def interpret(self, logical_tensor, config):
        if tuple(logical_tensor.shape) != tuple(self.shape):
            raise RuntimeError("The logical tensor's shape doesn't match")
        chunk_num = config[self.chunk_num].as_long()
        segments = list()
        for cid in range(chunk_num):
            # full tensor shape
            indices = TileIndices([0] * len(self.shape), self.shape)
            segment = logical_tensor.select(indices, self.val_map_op, self.shape)
            segments.append(segment)
        return segments
