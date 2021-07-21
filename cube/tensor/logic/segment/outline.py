"""
This is the description interface to describe the 
segementation requirement (restrictions).

The description includes two parts:

    1). restriction description on tensor segementation

    2). Translation procedure in runtime to translate such a restriction
        to the real segmentation on given logical tensor shape.
"""


# interface to setup restrictions on the segmentation

class Full:

    def __init__(self):
        pass

    def __call__(self, shape):
        pass

class SplitAxis:

    def __init__(self, axis, chunk_num=None, overlap=0):
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
        self.axis = axis
        self.chunk_num = chunk_num
        self.overlap = overlap
    
    def __call__(self, shape):
        """
        Runtime segment generation given the logical tensor shape
        """
        pass
