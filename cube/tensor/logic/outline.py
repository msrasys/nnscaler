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


# interface to setup restrictions on the segmentation

class BaseOutline:
    """
    Basic class for declare outline

    To setup an attribute (requirement), use `inst_baseoutline.attribute_name = val`
    """
    def __init__(self):
        self.reduction = reduction
        # decide how to generate segmentation given the requirement
        self.policy_fn = None

    def set_policy(self, policy_fn):
        if not callable(policy_fn):
            raise TypeError("Expected a function to take BaseOutline instance")
        self.policy_fn = policy_fn

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key].set(val)
        #TODO: Align semantics will not allow setting val on child, need a new class
        elif isinstance(val, MutableContainer) or isinstance(val, ConstantContainer):
            self.__dict__[key] = val
        elif val is None or isinstance(val, range) or isinstance(val, set):
            self.__dict__[key] = MutableContainer(val)
        else:
            self.__dict__[key] = ConstantContainer(val)

    def interpret(self, logical_tensor):
        raise NotImplementedError

    def __call__(self, logical_tensor):
        if not isinstance(logical_tensor, LogicalTensor):
            raise TypeError("Expected logical_tensor is instance of LogicalTensor")

        #TODO: merge out to fuse in configurable space
        if self.policy_fn is not None:
            self.policy_fn.get()(self)

        self.interpret(logical_tensor)


class Full(ConfigTemplate):

    def __init__(self):
        pass

    def interpret(self, logical_tensor):
        shape = logical_tensor.shape
        indices = TileIndices([0] * len(shape), shape)
        segment = Segment(logical_tensor, indices, self.reduction.get())
        return [segment]


class SplitAxis(ConfigTemplate):

    def __init__(self, axis, chunk_num=None, overlap=0, uniform=True):
        """
        Segmentation Pattern Requirement (parameters):

        axis (int): the axis to split

        chunk_num (None, int, tuple(int, int)):
            valid chunk numbers to split.
            If None, then any chunk number is valid;
            If an integer, only the specified chunk number is valid;
            If a tuple(min, max), the chunk number wihtin the scope [min,max] is valid


        overlap (0, int, tuple(int, int)):
            valid size for overlaping on the boundary of each splitted chunks.
            If None, any overlapping is valid
            If an integer, each overlap size is valid;
            if a tuple(min, max), the overlap size wihtin the scope [min,max] is valid

        """
        super().__init__()
        self.axis = axis
        self.chunk_num = chunk_num
        self.uniform = uniform
        self.overlap = overlap

    def interpret(self, logical_tensor):
        """
        Runtime segment generation given the logical tensor shape

        This is the policy that how to do the translation.
        """
        segments = list()
        shape = list(logical_tensor.shape)
        shape[self.axis.get()] = shape[self.axis.get()] // self.chunk_num.get()
        anchor = [0] * len(shape)
        #TODO: support list of reductions
        for cid in range(self.chunk_num.get()):
            indices = TileIndices(anchor, shape)
            segment = Segment(logical_tensor, indices)
            segments.append(segment)
            anchor[self.axis.get()] += shape[self.axis.get()]
        return segments


class SplitValue(ConfigTemplate):

    def __init__(self, chunk_num=None, val_map_op=None):
        ##TODO
        self.chunk_num = chunk_num
        self.val_map_op = val_map_op
