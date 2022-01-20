"""
This operator class is highly inspired by einops.

* Annotating Dimensions:

  e.g., 'a+', 'ab^', 'cd', '(ab+ c^ d)', '64'

A dimension of a tensor can be annotated by {identifier}{reduce} template.

An `identifier` must be one of:
  1) symbolic annotation that must match with the criteria of python str.isidentifier.
  2) numeric string that must match with python str.isnumeric. This indicates the shape is the same value
     numeric string will always have '^' reduction type
  3) '*': this special value indicates the dimension is dynamic will automatically get expanded given the shape

A `reduce` can be a set of {'', '+', '^'}:
  '' indicates this dimension will apear in output.
  '+' indicates no this dimension will be reduced in output using sume
  '^' means this dimension is out of scope, Einops will not handle this (cannot do split on it)

A complex annotation for a dimension is using brackets, i.e., '(' and ')', to include
more inner-dimensions. The value of inner dimension must be (partially) indicated by function args (of same name)
so that letting system know (infer).

* Annotating Operator:

e.g., 'm k+, n k+ -> m n', '4 k+, k+ d -> 8 d', '* d^, s -> * s'

An operator dimension can be annoted with input dimensions and output dimensions.
Same identifier indicates the same shape and semantically same dimension propagation.

'->' seperates the inputs (left) and outputs (right) and ',' separates each input and output.
A shape needs to be annotated using dimension annotations with delimiters of (mulitple) space ' '.

Dimension annotations in Output must apear in inputs, or using numeric string

* Splitting Rule:

Spatial Splitting (dimension with '' reduce type):
    tensors that have this dimension will be splitted spatially.
    tensors that don't have this dimension will be replicated.

Numerical Splitting (dimension with '+' reduce type):
    tensors that have this dimension will be splitted spatially,
    tensors that don't have this dimension will be splitted numerically

Illegal Splitting (dimension with '^' reduce type):
    Illegal splitting algorithm on this dimension.

"""

from typing import Callable, Dict, List, Union
from typing import Optional, Set, Tuple, Optional
import enum
import re
import copy
import string

from cube.ir.cten import IRTensor
from cube.graph.operator.operator import IRFwOperation
from cube.algorithm.factory import DistAlgorithmFactory


class EinDim:
    """
    To represent a dimension, name = {identifier}{reducetype}
    e.g.,
        ab^ means the dimension name is 'ab' and is a frozen dimension (cannot be split)
        ab+ means the dimension name is 'ab' and this dimension is a reduce dimension
        ['b', 'c+', 'd^'] means the dimension is composed by b, c, d
        where b can be spatially partitioned (apear in output), c is a reduce dimension,
        d is a frozen dimension (cannot be split)
    """

    class ReduceType(enum.Enum):
        Spatial=''
        Sum = '+'
        Stay = '^'  # the dim is not allowed to be split

    def __init__(self, name: Union[str, List[str]]):
        if isinstance(name, str):
            name = [name]
        self._name: List[str] = list()
        self._reduce: List[EinDim.ReduceType] = list()
        self._length: Dict[str, Optional[int]] = dict()
        for n in name:
            # complex name cannot have *
            if len(name) > 1 and '*' in n:
                raise ValueError("Einstein Axis name cannot have * for multiple inner-dimension")
            # get reduce type
            reduce = EinDim.ReduceType.Spatial
            if n[-1] == EinDim.ReduceType.Sum.value:
                reduce = EinDim.ReduceType.Sum
                n = n[:-1]
            elif n[-1] == EinDim.ReduceType.Stay:
                reduce = EinDim.ReduceType.Stay
                n = n[:-1]
            # get identifier name
            if len(n) == 0 or not (str.isidentifier(n) or str.isnumeric(n) or n == '*'):
                raise ValueError(f"EinDim name {n} should be identifier")
            if str.isnumeric(n):
                reduce = EinDim.ReduceType.Stay
            self._name.append(n)
            self._reduce.append(reduce)
        for n in self._name:
            self._length[n] = None

    @property
    def name(self) -> str:
        if len(self._name) == 1:
            return self._name[0]
        return '(' + ' '.join(self._name) + ')'

    def names(self) -> List[str]:
        return copy.copy(self._name)

    @property
    def reduce(self) -> str:
        return self._reduce

    def setlen(self, anno: str, dim: int):
        if anno not in self._name:
            raise KeyError(f"Cannot find anno: {anno} in {self.name}")
        self._length[anno] = dim

    def __eq__(self, other):
        if isinstance(other, EinDim):
            if other.name == self.name:
                return True
        return False

    def is_reduce(self):
        return self.reduce == EinDim.ReduceType.Sum

    def __repr__(self):
        name_reduce = [name + reduce.value for name, reduce in zip(self._name, self._reduce)]
        if len(self._name) == 0:
            return self._name[0] + self._reduce[0].value
        return '(' + ' '.join(name_reduce) + ')'


class EinopAnno:

    def __init__(self, anno: str):
        """
        initializing annotations specfied in str, e.g.,
            a (b c) d, d k -> a (b c) k
        """
        if not isinstance(anno, str):
            raise TypeError("Expected anno to be str")
        self.anno = anno
        if '->' not in self.anno:
            raise ValueError("Expected -> in anno")
        # to inputs and outputs
        inputs, outputs = self.anno.split('->')
        inputs = inputs.split(',')
        outputs = outputs.split(',')
        # to eindims
        self._identifiers: Set[str] = set()
        self.inputs: List[List[EinDim]] = [
            self.parse_shape(shape) for shape in inputs
        ]
        self.outputs: List[List[EinDim]] = [
            self.parse_shape(shape) for shape in outputs
        ]

    def parse_shape(self, shape: str) -> List[EinDim]:
        """
        parsing annotations like of a single shape, e.g.,
            a (b dim)  d
        """
        # => ['a', '(', 'b', 'dim', ')', 'd']
        shapes = list()
        for group in re.split('\ +', shape):
            if len(group) == 0:
                continue
            if '(' in group or ')' in group:
                for group in re.split('([\(\)])', group):
                    if len(group) != 0:
                        shapes.append(group)
            else:
                shapes.append(group)
        identifiers: List[List[str]] = list()
        current_identifier = list()
        bracket_group = False
        for w in shapes:
            if w == '(':
                if bracket_group:
                    raise RuntimeError("brackets inside brackets not allowed")
                bracket_group = True
                identifiers.append(current_identifier)
                current_identifier = list()
            elif w == ')':
                if not bracket_group:
                    raise RuntimeError("backets are not balanced at (")
                bracket_group = False
                identifiers.append(current_identifier)
                current_identifier = list()
            else:
                if bracket_group:
                    current_identifier.append(w)
                    self._identifiers.add(w)
                else:
                    if len(current_identifier) != 0:
                        identifiers.append(current_identifier)
                    current_identifier = [w]
                    self._identifiers.add(w)
        if bracket_group:
            raise RuntimeError("brackets are not balanced at )")
        if len(current_identifier) != 0:
            identifiers.append(current_identifier)
        identifiers = [EinDim(identifer) for identifer in identifiers]
        return identifiers

    def identifiers(self) -> Set[str]:
        return copy.copy(self._identifiers)

    def __repr__(self) -> str:
        inputs = ', '.join([repr(input) for input in self.inputs])
        outputs = ', '.join(repr(output) for output in self.outputs)
        return inputs + ' -> ' + outputs



class IREinops(IRFwOperation):
    """
    Einstein-inspired notation operations
    """
    def __init__(self, signature: str, annos: List[Union[str, Tuple[str, Callable]]],
                 inputs: List, name: str, **kwargs):        
        noutputs = set()
        self._annos: List[EinopAnno] = list()
        self._adapt: List[Union[Callable, None]] = list()
        for anno in annos:
            if isinstance(anno, tuple):
                anno, adapt = anno
            elif isinstance(anno, str):
                adapt = None
            else:
                raise TypeError("Expected annos to be list of tuples of list of str")
            anno = EinopAnno(anno)
            self._annos.append(anno)
            self._adapt.append(adapt)
            noutputs.add(len(anno.outputs))
        self._iannos: List[List[EinDim]] = None
        self._oannos: List[List[EinDim]] = None

        if len(noutputs) != 1:
            raise ValueError("Annotations should have same output length")
        super().__init__(name, signature, len(inputs), list(noutputs)[0])
        # set input
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        for name in kwargs:
            self.kwargs[name] = kwargs[name]

    def infer_shape(self) -> bool:
        """
        Shape inference by mathcing dimension annotations.
        Assume input shape is given
        """
        # try parsing given anno candidates
        ret = False
        for anno, adapt in zip(self._annos, self._adapt):
            if adapt is not None:
                anno = adapt(anno, self)
            ret, iannos, oannos = self.parse(anno)
            self._iannos = iannos
            self._oannos = oannos
            if ret: break
        if not ret:
            raise RuntimeError("No matching anno for given annos")
        dimlen: Dict[str, int] = dict()
        for input, ishape in zip(self.inputs(), self._iannos):
            if not ((ishape is None and not isinstance(input, IRTensor)) or
                    len(ishape) == len(input.shape)):
                raise RuntimeError(f"node {self._id} {self.signature}: error match input: {input.shape} and einshape: {ishape}")
            for tdim, edim in zip(input.shape, ishape):
                if len(edim._name) == 1:
                    if edim.name in dimlen and dimlen[edim.name] != tdim:
                        raise RuntimeError(f"op: {self.signature} has different shape for same dim annotation {edim.name}")
                    dimlen[edim.name] = tdim
                    edim.setlen(edim.name, tdim)
                else:
                    toinfer = list()
                    accum = 1
                    for name in edim._name:
                        if str.isnumeric(name):
                            accum *= int(name)
                            edim.setlen(name, int(name))
                            dimlen[name] = int(name) 
                        elif name in self.kwargs:
                            accum *= self.kwargs[name]
                            edim.setlen(name, self.kwargs[name])
                            dimlen[name] = self.kwargs[name]
                        else:
                            toinfer.append(name)
                    if len(toinfer) > 1:
                        raise RuntimeError(f"Expected indication of dimension {toinfer} from kwargs")
                    if len(toinfer) == 1:
                        edim.setlen(toinfer[0], tdim // accum)
                        dimlen[toinfer[0]] = tdim // accum
        # figure output shape
        for oidx in range(len(self._outputs)):
            output_shape = list()
            for odim in self._oannos[oidx]:
                accum = 1
                for name in odim._name:
                    if str.isdecimal(name):
                        accum *= int(name)
                    else:
                        if name not in dimlen:
                            raise KeyError(f"Dim annotation {name} not in input")
                        accum *= dimlen[name]
                        odim.setlen(name, dimlen[name])
                output_shape.append(accum)
            self.outputs(oidx).shape = output_shape
        return ret

    def new(self, inputs: List, outputs: List):
        """
        construct a new operator sharing same kwargs with new inputs
        and outputs
        """
        annos = list()
        for anno, adapt in zip(self._annos, self._adapt):
            annos.append((anno.anno, adapt))
        op = IREinops(self.signature, annos, inputs, self.name, **self.kwargs)
        for idx, output in enumerate(outputs):
            op.set_output(idx, output)
        return op

    def parse(self, anno: EinopAnno):
        """
        parse annotations, assuming input tensor shape is given
        """
        if len(anno.inputs) != len(self.inputs()):
            return False, None, None
        identifiers = anno.identifiers()
        # expand *
        expand_dims = None
        if '*' in identifiers:
            for idx in range(len(anno.inputs)):
                shape = anno.inputs[idx]
                shape_anno = [dim.name for dim in shape]
                if '*' in shape_anno:
                    start = shape_anno.index('*')
                    span = len(self.inputs(idx).shape) - len(shape) + 1
                    if span <= 0:
                        if expand_dims is None:
                            expand_dims = list()
                        if len(expand_dims) > 0:
                            return False, None, None
                        anno.inputs[idx].remove(EinDim('*'))
                    if span > 0:
                        if expand_dims is None:
                            expand_dims = list()
                            unused_annos = [c for c in string.ascii_lowercase if c not in identifiers]
                            if len(unused_annos) < span:
                                raise RuntimeError("Too many introduced dimensions")
                            for dim in range(span):
                                expand_dims.append(EinDim([unused_annos[dim]]))
                        if len(expand_dims) != span:
                            return False, None, None
                        anno.inputs[idx] = anno.inputs[idx][:start] + expand_dims + anno.inputs[idx][start+1:]
            for idx in range(len(anno.outputs)):
                shape = anno.outputs[idx]
                shape_anno = [dim.name for dim in shape]
                if '*' in shape_anno:
                    if expand_dims is None:
                        raise RuntimeError("* should appear in inputs")
                    start = shape_anno.index('*')
                    span = len(expand_dims)
                    anno.outputs[idx] = anno.outputs[idx][:start] + expand_dims + anno.outputs[idx][start+1:]
        # check dimension consistency
        dimlen: Dict[str, int] = dict()
        for shape, input in zip(anno.inputs, self.inputs()):
            if not isinstance(input, IRTensor):
                if not (len(shape) != 1 and shape[0].name != '1'):
                    return False, None, None
            else:
                if len(input.shape) != len(shape):
                    return False, None, None
                for edim, nele in zip(shape, input.shape):
                    if edim.name in dimlen:
                        if nele != dimlen[edim.name]:
                            return False, None, None
                    dimlen[edim.name] = nele
        return True, anno.inputs, anno.outputs

    def einexpr(self) -> str:
        inputs = list()
        outputs = list()
        for shape in self._iannos:
            inputs.append(' '.join([repr(edim) for edim in shape]))
        for shape in self._oannos:
            outputs.append(' '.join([repr(edim) for edim in shape]))
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
