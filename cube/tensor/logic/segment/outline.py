"""
This is the description interface to describe the 
segementation requirement (restrictions).

The description includes two parts:

    1). restriction description on tensor segementation

    2). Translation procedure in runtime to translate such a restriction
        to the real segmentation on given logical tensor shape.
"""

from cube.tensor.logic.segment.segment import TileSegment, ReductionOp


class MutableContainer:

    def __init__(self, scope):
        self.__val = None
        self.__scope = scope

    def get(self, scope=False):
        if scope:
            return self.__scope
        else:
            return self.__val
    
    def set(self, val):
        if self.__scope is not None:
            if val not in self.__scope:
                raise ValueError("Fail to set container, out of range")
        self.__val = val


class ConstantContainer:

    def __init__(self, val):
        self.__val = val
    
    def get(self):
        return self.__val

    def set(self, val):
        raise RuntimeError("Cannot set a ConstantContainer")


# interface to setup restrictions on the segmentation


class BaseOutline:
    """
    Basic class for declare outline

    To setup an attribute (requirement), use `inst_baseoutline.attribute_name = val`
    """
    def __init__(self, reduction=None):
        self.reduction = reduction

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


class Full(BaseOutline):

    def __init__(self, reduction=None):
        super().__init__(reduction)

    def __call__(self, shape):
        segment = TileSegment([0] * len(shape), list(shape), self.reduction.get())
        return [segment]


class SplitAxis(BaseOutline):

    def __init__(self, axis, chunk_num=None, overlap=0, reduction=None, uniform=True):
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
        super().__init__(reduction)
        self.axis = axis
        self.chunk_num = chunk_num
        self.uniform = uniform
        self.overlap = overlap

    def __call__(self, shape):
        """
        Runtime segment generation given the logical tensor shape

        This is the policy that how to do the translation.
        """ 
        segments = list()
        shape = list(shape)
        shape[self.axis.get()] = shape[self.axis.get()] // self.chunk_num.get()
        anchor = [0] * len(shape)
        #TODO: support list of reductions
        for cid in range(self.chunk_num.get()):
            segment = TileSegment(
                list(anchor), list(shape), reduction=self.reduction)
            anchor[self.axis.get()] += shape[self.axis.get()]
            segments.append(segment)
        return segments
