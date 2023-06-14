from typing import Generator, Iterable, List, Any, Optional, Tuple

from cube.ir.cten import IRCell, IRTensor, IRObject
from cube.ir.dtype import IRDType
from cube.ir.tensor import IRSubTensor
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.ir.adapter import IRWeightReducer, IRAdapter
from cube.ir.adapter.prim import IRAdapterPrim

from cube.graph.segment import IRSegment

from cube.codegen.frontend_mapping import Sign2EmitRule

from cube.flags import CompileFlag


class IRValue:

    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return self.name


class CodeEmission:
    """
    Basic emission
    """

    @staticmethod
    def dtype_map(dtype: IRDType) -> str:
        if not isinstance(dtype, IRDType):
            raise TypeError("Expected IRDType")
        return 'torch.' + dtype.value

    @staticmethod
    def node_name(node: IRCell) -> str:
        return f"{node.name}{node.cid}"
    
    @staticmethod
    def tensor_name(tensor: Any, prefix_attr: Optional[str] = None) -> str:
        """
        Return the var name.
        For tensor, return the {prefix}{tensor.name}_{tensor.tid}
        For non-tensor, return its string

        @param tensor Any: any value
        @attr_prefix Optional[str]: prefix for a attributed tensor

        @return str
        """
        if isinstance(tensor, IRObject):
            tensor_name = tensor.name
            if '.' in tensor_name:
                tensor_name = tensor_name.split('.')[0]
            name = '_'.join([tensor_name, str(tensor.tid)])
            if prefix_attr is not None and tensor.is_attr():
                name = prefix_attr + name
        else:
            name = str(tensor)
        return name

    @staticmethod
    def complex_name(val: Any, prefix_attr: Optional[str]=None) -> str:
        """
        Return the val name with complex data type over IRObject
        Currently support complex data type of Dict, List, Tuple, IRObject
        """
        modifier = lambda t: IRValue(CodeEmission.tensor_name(t, prefix_attr))
        val = IRSegment.modify_objects_of_complex(val, modifier)
        return str(val)

    @staticmethod
    def tuple_name(tensors: List[Any],
                   skip_attr: bool = False, prefix_attr: Optional[str] = None) -> str:
        """
        Return the tupled tensor name.

        @param tensors List[Any]: list of any value
        @param skip_attr bool: whether to skip graph attribute in the tensors
        @param prefix_attr bool: whether to add a prefix for graph attribute

        @return name str: the tupled tensor name
        """
        names = []
        for t in tensors:
            if isinstance(t, IRTensor) and skip_attr and t.is_attr():
                continue
            names.append(CodeEmission.tensor_name(t, prefix_attr))
        name = '(' + ', '.join(names + ['']) + ')'
        return name
    
    @staticmethod
    def return_name(tensors: List[Any],
                    skip_attr: bool = False, prefix_attr: Optional[str] = None) -> str:
        names = []
        for t in tensors:
            if isinstance(t, IRTensor) and skip_attr and t.is_attr():
                continue
            names.append(CodeEmission.tensor_name(t, prefix_attr))
        names = '_' if len(names) == 0 else ', '.join(names)
        return names
    
    @staticmethod
    def return_name_complex(vals: List[Any],
                            skip_attr: bool = False, prefix_attr: Optional[str] = None) -> str:
        names = []
        for t in vals:
            if isinstance(t, IRObject) and skip_attr and t.is_attr():
                continue
            names.append(CodeEmission.complex_name(t, prefix_attr))
        names = '_' if len(names) == 0 else ', '.join(names)
        return names
    
    @staticmethod
    def kwargs_name(**kwargs) -> str:
        names = []
        for name, val in kwargs.items():
            # TODO: Ad-hoc patch for amp
            if CompileFlag.use_amp and val == 'torch.float32':
                val = 'torch.float16'
            names.append(f'{name}={val}')
        name = ', '.join(names)
        return name
    

class FuncEmission(CodeEmission):

    @staticmethod
    def emit_dataloader(node: IRDataOperation) -> List[str]:
        return ['next(dataloader)']
    
    @staticmethod
    def emit_fnode(node: IRFwOperation, prefix_attr: str = None) -> List[str]:
        """
        Emit the statement to call the op in the forward code
        (e.g. in Segments, Adapter or CodeGen.Main)

        The result will look like (the lines are split into `List[str]`)
        ```
        tensor_3333 = torch.view(tensor_2222, [1,2,3,4,5])
        ```

        The fields storing intermediate codes that are populated by this method:
        -   NONE
        """
        assert isinstance(node, IRFwOperation)
        codes = []
        # insert comment
        if node.comment is not None:
            codes.append(f'# {node.comment}')
        signature = node.signature
        inputs = [FuncEmission.tensor_name(t, prefix_attr=prefix_attr) for t in node.inputs()]
        kwargs = {}
        for key in node.kwargs:
            val = node.kwargs[key]
            if isinstance(val, str) and 'self.' not in val:
                val = '"' + val + '"'
            kwargs[key] = val

        emit_rule = Sign2EmitRule.map(signature)
        body = emit_rule(node, inputs, kwargs)

        if len(node.outputs()) == 0:
            code = body
        else:
            outputs = [FuncEmission.tensor_name(t) for t in node.outputs()]
            outputs = ', '.join(outputs)
            code = f'{outputs} = {body}'
        codes.append(code)
        return codes
    
    @staticmethod
    def emit_adapter(node: IRAdapter, prefix_attr: Optional[str] = None) -> List[str]:
        """
        Emit the statment of the adapter call
        
        The resultant `List[str]` will be lines of the statements of the final
        Python method for the targeted Segment, 
        without the method signature and the return statement.

        The fields storing intermediate codes that are populated by this method:
        -   NONE
        """
        codes = []
        assert len(node.device) == 1, f"Expected adapter to be dispatched:\n{node.extra_repr()}"
        prims = [node] if node.differentiable and node.custom else [prim for prim in node.prims]

        # only adapter that is non-differentiable can be executed as async
        async_op = CompileFlag.async_comm and (not node.differentiable)
        if async_op:
            for idx, prim in enumerate(prims):
                if isinstance(prim, IRAdapterPrim) and prim.volume() == 0:
                    continue
                break
            #TODO: support more general cases: independent same-group primitives
            async_op = False if len(prims[idx:]) != 1 else async_op

        for prim in prims:
            if len(prim.inputs()) == 1:
                itensors = FuncEmission.tensor_name(prim.inputs()[0], prefix_attr=prefix_attr)
            else:
                itensors = FuncEmission.tuple_name(prim.inputs(), prefix_attr=prefix_attr)
            prim_kwargs = dict(prim.kwargs)
            if async_op:
                prim_kwargs['async_op'] = True
            kwargs = FuncEmission.kwargs_name(**prim_kwargs)
            outputs = FuncEmission.return_name(prim.outputs())
            code = f'{outputs} = {prim.signature}({itensors}, {kwargs})'
            codes.append(code)
        return codes
    
    @staticmethod
    def emit_reducer(node: IRWeightReducer) -> List[str]:
        """
        Emit the statment to invoke a reducer object.
        
        The fields storing intermediate codes that are populated by this method:
        -   NONE
        """
        reducer_name = f'self.wreducer{node._id}'
        code = f'{reducer_name}.sync_grads()'
        return [code]

    @staticmethod
    def emit_release(tensors: Iterable[IRTensor]) -> str:
        tnames : Generator = (FuncEmission.tensor_name(t) for t in tensors)
        return 'del ' + ', '.join(tnames)
    
    @staticmethod
    def get_backward_callsite_io_tensors(bwop: IRCell) -> Tuple:
        """
        Get backward inputs and outputs
        ```
        (input_tensors, output_tensors, output_grads, input_grads)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~
        #inputs to 'backward'                         outputs of 'backward'
        ```
        
        @return input_tensors List[IRSubTensor]: forward input tensors (backward input)
        @return output_tensors List[IRSubTensor]: forward output tensors (backward output)
        @return output_grads List[IRSubTensor]: gradient of forward output tensors
            (backward input)
        @return input_grads List[IRSubTensor]: gradient of forward input tensors
            (backward output)
        """
        assert not bwop.isfw()
        fwop: IRCell = bwop.mirror

        grad2tensor = {}
        for t in fwop.inputs() + fwop.outputs():
            if isinstance(t, IRSubTensor) and t.grad is not None:
                grad2tensor[t.grad] = t

        input_grads = [t for t in bwop.outputs() if isinstance(t, IRSubTensor)]
        output_grads = [t for t in bwop.inputs() if isinstance(t, IRSubTensor)]
        input_tensors = [grad2tensor[g] for g in input_grads]
        output_tensors = [grad2tensor[g] for g in output_grads]

        return input_tensors, output_tensors, output_grads, input_grads
