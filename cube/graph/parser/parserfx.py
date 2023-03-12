import torch
import enum
import re
from typing import Any, List, Tuple, Optional, Callable, Union

from cube.ir.operator import IRFwOperation
from cube.ir.tensor import IRFullTensor
from cube.ir.cten import IRObject, IRCell
from cube.graph.parser.frame import Frame
from cube.graph.parser.mapping import DType2IRDType
from cube.graph.parser.mappingfx import SignFx2Op
from cube.graph.function.pyfunc import IRPyFunc

import torch.fx

class ErasedDevice:
    pass

class FxNodeKind(enum.Enum):
    PrimGetAttr = 1
    PrimCallMethod = 2
    PrimCallFunction = 3  # -> the parser may end here
    PrimConstant = 4
    AtenOp = 5  # -> the parser may end here
    PrimIf = 6  # dynamic
    PrimListConstruct = 7
    PrimListUnpack = 8
    PrimTupleUnpack = 9
    PrimPythonOp = 10
    PrimDevice = 11  # erased
    PrimLoop = 12
    PrimCallModule = 13
    # for torch.fx
    Placeholder = 14
    Output = 15


class FxFuncOpTracer(torch.fx.Tracer):
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True
        # capture torch.nn.functional return
        return m.__module__.startswith('torch.nn.functional') and not isinstance(m, torch.nn.Sequential)


class FxModuleParser:
    save_content: bool = True

    @staticmethod
    def shape_refine(shape: torch.Size) -> torch.Size:
        """
        replacing scale shape [] to [1]
        :param shape:
        :return:
        """
        # TODO update
        return torch.Size([1]) if shape == torch.Size([]) else shape


    @staticmethod
    def parse(module: torch.fx.GraphModule,
              input_shapes: Optional[Tuple[List[int],]] = None,
              dummy_inputs = None,
              frame: Frame = None) \
            -> Tuple[List[IRFullTensor], List[IRFwOperation], List[IRFullTensor]]:
        """
        The overall entry to parse a torch.fx graph module
        """
        frame = frame if frame is not None else Frame()
        frame.push_var()
        frame.push_attr()

        inputs = [node for node in module.graph.nodes if node.op == 'placeholder']
        print(f'inputs = {inputs}')

        if input_shapes is not None and len(input_shapes) != len(inputs):
            print(f'module(type = {type(module)}.__dict__.keys() = {module.__dict__.keys()}')
            print(f'input shape mismatch (got {len(input_shapes)} != {len(inputs)})')
            # TODO fixme raise RuntimeError(f"Module {module.original_name} input shape mismatch (got {len(input_shapes)} != {len(inputs)})")
        
        default_dtype = torch.get_default_dtype()
        kDefaultType = DType2IRDType.map(default_dtype) # TODO specify dtype
        if input_shapes is not None:
            # shape propagation
            sample_inputs = dummy_inputs if dummy_inputs else [torch.ones(shape, dtype=default_dtype) for shape in input_shapes]
            sample_input_tensors = [sample_inputs[input] for input in sample_inputs] if type(sample_inputs) is dict else sample_inputs

            # from torch.fx.passes.shape_prop import ShapeProp
            # ShapeProp(module).propagate(*sample_input_tensors)  # TODO fixme ShapeProp(module).propagate(*sample_inputs)
            from cube.graph.parser.concrete_trace_utils.kwargs_shape_prop.kwargs_shape_prop import KwargsShapeProp as ShapeProp
            ShapeProp(module).propagate(sample_inputs)

            # handle graph inputs
            for idx, input in enumerate(inputs):
                assert isinstance(input, torch.fx.Node)
                shape = None if (input_shapes is None or len(input_shapes) <= idx) else input_shapes[idx]
                dtype = kDefaultType
                val = IRFullTensor(shape=shape, requires_grad=False, dtype=dtype, name=input.name)
                frame.add_var(input.name, val, graph_arg=idx)
        else:
            assert dummy_inputs is not None, 'input_shapes and dummy_inputs cannot be None at the same time.'
            # remove dead nodes
            # from nni.common.concrete_trace_utils.kwargs_shape_prop.kwargs_shape_prop import DCEHandler
            from cube.graph.parser.concrete_trace_utils.kwargs_shape_prop.kwargs_shape_prop import DCEHandler
            DCEHandler(module).eliminate_dead_code()
            # shape propagation
            # from nni.common.concrete_trace_utils.kwargs_shape_prop.kwargs_shape_prop import KwargsShapeProp
            # KwargsShapeProp(module).propagate(dummy_inputs)
            from cube.graph.parser.concrete_trace_utils.kwargs_shape_prop.kwargs_shape_prop import KwargsShapeProp as ShapeProp
            ShapeProp(module).propagate(dummy_inputs)
            # handle graph inputs
            for idx, input in enumerate(inputs):
                assert isinstance(input, torch.fx.Node)
                if isinstance(dummy_inputs, dict):
                    if input.name in dummy_inputs:
                        shape = input.meta['tensor_meta'].shape
                    else:
                        shape = None
                else:
                    # FIXME: this part is only for transformers.tokenization_utils_base.BatchEncoding,
                    # extend to other input types
                    if hasattr(dummy_inputs, input.name):
                        print(f'dummy_inputs has {input.name}')
                        shape = getattr(dummy_inputs, input.name).size()
                    else:
                        # FIXME: seems the kwargs name (e.g., _deprecated_arguments) is not aligned with input.name
                        print(f'dummy_inputs does not have {input.name}')
                        shape = None
                dtype = kDefaultType
                val = IRFullTensor(shape=shape, requires_grad=False, dtype=dtype, name=input.name)
                frame.add_var(input.name, val, graph_arg=idx)
        input_val = [frame.get_var(input.name) for input in inputs]

        for node in module.graph.nodes:
            if 'tensor_meta' in node.meta:
                if node.meta['type'] is type(tuple()):
                    print(f'{node.name} is tuple type')
                elif node.meta['type'] is type(torch.fx.immutable_collections.immutable_dict()):
                    print(f'{node.name} is immutable_dict type')
                    assert isinstance(node.meta['tensor_meta'], dict)
                else:
                    if node.meta['type'] is type(torch.Tensor()) or node.meta['type'] is type(torch.nn.parameter.Parameter()):
                        print(node.name, node.meta['tensor_meta'].dtype, node.meta['tensor_meta'].shape)
                    else:
                        print(f'WARNING node {node.name} is neither Tensor nor Parameter')
            else:
                print(f'{node.name} does not has tensor_meta')

        # add activations to frame, including call_func/call_method output and final output
        activation_op_strs = {'call_function', 'output', 'call_method'}
        activation_nodes = [node for node in module.graph.nodes if node.op in activation_op_strs]
        for node in activation_nodes:
            if hasattr(node, 'meta') and node.meta.get('tensor_meta') and hasattr(node.meta['tensor_meta'], 'dtype'):
                assert isinstance(node, torch.fx.Node)
                shape = node.meta['tensor_meta'].shape
                shape = FxModuleParser.shape_refine(shape)
                dtype = DType2IRDType.map(node.meta['tensor_meta'].dtype)
                requires_grad = node.meta['tensor_meta'].requires_grad
                val = IRFullTensor(shape=shape, requires_grad=requires_grad, dtype=dtype, name=node.name)
                frame.add_var(node.name, val)
            else:
                frame.add_var(node.name, IRObject())

        # handle nodes
        all_ir_nodes: List[IRFwOperation] = list()
        total_node_num = len(module.graph.nodes)
        node_idx = 1
        for node in module.graph.nodes:
            print(f'[{node_idx}/{total_node_num}]')
            node_idx += 1

            ir_nodes = FxModuleParser.parse_node(node, module, frame)
            if ir_nodes is not None:
                all_ir_nodes += ir_nodes

        output_val = [frame.get_var(node.name) for node in module.graph.nodes if node.op == 'output']
    
        frame.pop_var()
        frame.pop_attr()
        if FxModuleParser.save_content:
            frame.save_attr_content()

        return input_val, all_ir_nodes, output_val


    @staticmethod
    def ntype(node: torch.fx.Node):
        if node.op == 'call_module':
            return FxNodeKind.PrimCallModule
        if node.op == 'call_function':
            return FxNodeKind.PrimCallFunction
        if node.op == 'get_attr':
            return FxNodeKind.PrimGetAttr
        if node.op == 'placeholder':
            return FxNodeKind.Placeholder
        if node.op == 'output':
            return FxNodeKind.Output
        if node.op == 'call_method':
            return FxNodeKind.PrimCallMethod
        raise RuntimeError(f"Unknown node kind {node.kind()} from torchscript module")

    @staticmethod
    def parse_node(node: torch.fx.Node, module, frame: Frame) -> List[IRFwOperation]:
        """
        Parse the node and return the IRFwOperation nodes
        """
        node_type = FxModuleParser.ntype(node)
        try:
            if node_type == FxNodeKind.Placeholder:
                return []
            if node_type == FxNodeKind.Output:
                return FxModuleParser.parse_prim_output_node(node, module, frame)
            if node_type in (FxNodeKind.PrimCallFunction, FxNodeKind.PrimCallMethod):
                return FxModuleParser.parse_prim_function_method(node, module, frame)
            if node_type == FxNodeKind.PrimGetAttr:
                return FxModuleParser.parse_prim_attr_node(node, module, frame)
            if node_type == FxNodeKind.PrimCallModule:
                raise RuntimeError(f"parse_prim_module is not supported.")

            # TODO bother assigning all ignored prim functions new NodeKinds?
            if node_type == FxNodeKind.PrimDevice:
                return FxModuleParser.parse_value_erased_node(node, module, frame, [ErasedDevice()])
            raise NotImplementedError(f"Un-supported node type {node_type}")
        except Exception:
            raise RuntimeError(f"\n\nParsing error at node:\n\t{node}\n")

    @staticmethod
    def fetch_attr(mod: torch.fx.GraphModule, target: str):
        target_atoms = target.split('.')
        attr_itr = mod
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    @staticmethod
    def parse_prim_function_method(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRFwOperation]:
        # get signature
        fsig = FxModuleParser._get_qualified_name(node.target, node)
        if isinstance(node.target, str):
            print(f'parse_prim_method_node: {fsig}')
        else:
            print(f'parse_prim_function_node: {fsig}')

        def get_complex_data(val: Any) -> Any:
            """Change inner fx.Node into IRObject"""
            if isinstance(val, tuple):
                return tuple(get_complex_data(t) for t in val)
            if isinstance(val, list):
                return list(get_complex_data(t) for t in val)
            if isinstance(val, torch.fx.Node):
                return frame.get_var(val.name)
            return val
    
        # get inputs
        input_vals = [get_complex_data(val) for val in node.args]
        kwargs = {key: get_complex_data(val) for key, val in node.kwargs.items()}

        # map to IR operator
        if SignFx2Op.exist(fsig):
            ir_node = SignFx2Op.map(fsig)(*input_vals, **kwargs)
        else:
            # FIXME: handle cases for IRObject in kwargs
            # case1: unknown torch operator
            if FxModuleParser._is_torch_autograd_op(node, frame, fsig):
                print(f'>>> Find unknown pytorch operation: {fsig}')
                fname = fsig.split('.')[-1] if '.' in fsig else fname
                ir_node = IRFwOperation(fname, fsig, len(input_vals), 1)
                ir_node.kwargs = kwargs
                for idx, t in enumerate(input_vals):
                    ir_node.set_input(idx, t)
            # case2: python runtime function
            else:
                print(f'>>> Set python runtime function: {fsig}')
                ir_node = IRPyFunc(fsig, input_vals, [None], **kwargs)

        if isinstance(ir_node, IRCell):
            # TODO gracefully set output
            output_name = node.name
            output_val = frame.get_var(output_name)
            ir_node.set_output(0, output_val)
            return [ir_node]
        else:
            frame.set_var(node.name, ir_node)
            return []

    @staticmethod
    def parse_prim_attr_node(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRFwOperation]:
        """
        There are two types of get_attr, one is `FxNodeKind.PrimGetAttr` which is dealt with in this function.
        The other is `FxNodeKind.PrimCallFunction ` (i.e., <built-in function getattr>)
        which is dealt with by parse_prim_function_method.
        """
        assert node is not None
        tensor_name = node.name
        if 'tensor_meta' in node.meta:
            tensor_shape = node.meta['tensor_meta'].shape
            # tensor_dtype = node.meta['tensor_meta'].dtype
            #TODO assume it is weight
            default_dtype = torch.get_default_dtype()
            kDefaultType = DType2IRDType.map(default_dtype)  # TODO specify dtype
            ir_tensor = IRFullTensor(tensor_shape, tensor_name, requires_grad=True, dtype=kDefaultType)
            ir_tensor.as_param()
            frame.add_var(tensor_name, ir_tensor)
        else:
            var = FxModuleParser.fetch_attr(module, node.target)
            frame.add_var(tensor_name, var)

        return None

    @staticmethod
    def parse_prim_output_node(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRCell]:
        assert len(node.args) == 1 and len(node.kwargs) == 0
        ir_nodes = []

        # handle complex outputs
        def generate_outputs(val: Any, _ops: List) -> IRObject:
            """Support complex data type of List, Tuple, Dict, Tensor/Object"""
            if isinstance(val, list):
                inputs = tuple(generate_outputs(sub_node, _ops) for sub_node in val)
                output = IRObject()
                _ops.append(IRPyFunc('(lambda *args: list(args))', inputs, [output]))
                return output
            if isinstance(val, tuple):
                inputs = tuple(generate_outputs(sub_node, _ops) for sub_node in val)
                output = IRObject()
                _ops.append(IRPyFunc('(lambda *args: args)', inputs, [output]))
                return output
            if isinstance(val, dict):
                output = IRObject()
                assert all(not isinstance(key, torch.fx.Node) for key in val.keys()), f"output dict cannot have torch.fx.Node is key"
                keys = tuple(str(key) for key in val.keys())
                values = generate_outputs(tuple(generate_outputs(value, _ops) for value in val.values()), _ops)
                _ops.append(IRPyFunc('(lambda vals, keys: {key:val for key,val in zip(keys,vals)})', [values], [output], keys=keys))
                return output
            if isinstance(val, torch.fx.Node):
                return frame.get_var(val.name)
            return val
        
        output = generate_outputs(node.args[0], ir_nodes)
        frame.set_var(node.name, output)
        return ir_nodes

    # # NOTE: this is a function in torch.fx
    # @staticmethod
    # def _get_qualified_name(func: Callable[..., Any]) -> str:
    #     # things like getattr just appear in builtins
    #     if getattr(builtins, func.__name__, None) is func:
    #         return func.__name__
    #     # torch.Tensor.{fn}
    #     if isinstance(func, types.MethodDescriptorType) and func is getattr(torch.Tensor, func.__name__, None):
    #         return f"torch.Tensor.{func.__name__}"
    #     name = func.__name__
    #     module = FxModuleParser._find_module_of_method(func)
    #     module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
    #     # Fixup segment_reduce mismatch
    #     if module == "torch" and name == "segment_reduce":
    #         name = "_" + name
    #     return f'{module}.{name}'

    @staticmethod
    def _get_qualified_name(node_target: Union[str, Callable[..., Any]], node: torch.fx.Node = None) -> str:
        if isinstance(node_target, str):
            assert node is not None
            return FxModuleParser._get_qualified_name_of_call_method(node_target, node)
        else:
            return FxModuleParser._get_qualified_name_of_call_function(node_target)

    @staticmethod
    def _get_qualified_name_of_call_function(node_target: Callable[..., Any]) -> str:
        """
        The target field of call_function node must be an callable object.
        """
        # # things like getattr just appear in builtins
        # if getattr(builtins, func.__name__, None) is func:
        #     return func.__name__
        # TODO(yizhu1): find a general solution
        assert callable(node_target)
        name = node_target.__name__
        module = FxModuleParser._find_module_of_method(node_target)
        module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
        return f'{module}.{name}'

    @staticmethod
    def _get_qualified_name_of_call_method(node_target: str, node: torch.fx.Node) -> str:
        """
        The target field of call_method node must be a string.
        """
        assert isinstance(node_target, str)
        for module, module_name in [(torch, 'torch'), (torch.Tensor, 'torch.Tensor')]:
            lib_func = getattr(module, node_target, None)
            if lib_func is not None and callable(lib_func):
                return f'{module_name}.{node_target}'
        assert len(node.args) == 1, f'invalid args {node.args} in {node.name}, {node.target}, {node.meta}'
        assert len(node.kwargs) == 0, f'invalid kwargs {node.kwargs} in {node.name}, {node.target}, {node.meta}'
        # example node.args[0].meta is {'type': <class 'dict'>}
        in_type = node.args[0].meta['type']
        assert node_target in in_type().__dir__(), f'node_target = {node_target}, in_type().__dir__() = {in_type().__dir__()}'
        sig = f'{in_type.__name__}.{node_target}'
        print(f'The method is not torch or Tensor, but {sig}')
        return sig

    # this is fixed on master, WAR for 1.5
    @staticmethod
    def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
        name = orig_method.__name__
        module = orig_method.__module__
        if module is not None:
            return module
        for guess in [torch, torch.nn.functional]:
            if getattr(guess, name, None) is orig_method:
                return guess.__name__
        raise RuntimeError(f'cannot find module for {orig_method}')
    
    @staticmethod
    def _is_torch_autograd_op(node: torch.fx.Node, frame: Frame, signature: str) -> bool:
        """Check whether the node is of a pytorch autograd operation."""
        # note: some python operations like torch.Tensor.size() doesn't return
        # an IRTensor, thus cannot be considered as a pytorch autograd operator.
        return signature.startswith('torch.') and \
               isinstance(frame.get_var(node.name), IRFullTensor)
