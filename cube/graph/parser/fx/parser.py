import torch
import enum
import logging
from pathlib import Path
from typing import Any, List, Tuple, Callable, Union, Dict, Type

from cube.ir.operator import IRFwOperation
from cube.ir.tensor import IRFullTensor
from cube.ir.cten import IRObject, IRCell
from cube.graph.parser.frame import Frame
from cube.graph.parser.fx.mapping import SignFx2Op
from cube.graph.function.pyfunc import IRPyFunc
from cube.graph.function.dimops import IRDimops

import torch.fx

_logger = logging.getLogger(__name__)


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


def get_complex_data(val: Any, frame: Frame) -> Any:
    """Change inner fx.Node into IRObject"""
    if isinstance(val, tuple):
        return tuple(get_complex_data(t, frame) for t in val)
    if isinstance(val, list):
        return list(get_complex_data(t, frame) for t in val)
    if isinstance(val, torch.fx.Node):
        return frame.get_var(val.name)
    return val


class FxModuleParser:
    """
    torch.fx module parser
    """

    ATTR_CONTENT_FILE = 'fullmodel.pt'
    ATTR_MAP_FILE = 'dist_param_map.pt'

    @staticmethod
    def shape_refine(shape: torch.Size) -> torch.Size:
        """Replacing scale shape [] to [1]

        Args:
            shape (torch.Size): tensor shape

        Returns:
            torch.Size: refined shape
        """
        return torch.Size([1]) if shape == torch.Size([]) else shape


    @staticmethod
    def parse(module: torch.fx.GraphModule,
              dummy_inputs: Dict[str, Any],
              frame: Frame = None,
              attr_savedir='./',
              *,
              save_content: bool = True,
              dynamic_shape: bool = True
        ) -> Tuple[List[IRFullTensor], List[IRFwOperation], List[IRFullTensor]]:
        """Parse torch.fx module into cube IR

        The overall entry to parse a torch.fx graph module

        Args:
            save_content (bool): whether to save the content of the module
            dynamic_shape (bool): whether to parse the module with dynamic shape
        """
        from cube.graph.parser.fx.concrete_trace_utils.kwargs_shape_prop.kwargs_shape_prop import KwargsShapeProp as ShapeProp

        frame = frame if frame is not None else Frame()
        frame.push_var()
        frame.push_attr()

        assert isinstance(dummy_inputs, dict), "Expected dummy inputs to parse module"

        inputs = [node for node in module.graph.nodes if node.op == 'placeholder']
        _logger.info(f'> torch.fx parser: graph inputs: {inputs}')

        # shape propagation
        ShapeProp(module).propagate(dummy_inputs)
        # handle graph inputs
        for idx, input in enumerate(inputs):
            assert isinstance(input, torch.fx.Node)
            # dealing with different types of dummy_inputs
            if isinstance(dummy_inputs, dict):
                if input.name not in dummy_inputs:
                    val = IRObject(input.name)
                else:
                    if 'tensor_meta' in input.meta:
                        shape = input.meta['tensor_meta'].shape
                        if len(shape) == 0:
                            shape = [1]
                        dtype = input.meta['tensor_meta'].dtype
                        val = IRFullTensor(shape=shape, requires_grad=False, dtype=dtype, name=input.name)
                    else:
                        val = IRObject(input.name, value=dummy_inputs[input.name])
            else:
                # FIXME: this part is only for transformers.tokenization_utils_base.BatchEncoding,
                # extend to other input types
                if hasattr(dummy_inputs, input.name):
                    shape = getattr(dummy_inputs, input.name).size()
                else:
                    # FIXME: seems the kwargs name (e.g., _deprecated_arguments) is not aligned with input.name
                    shape = None
                dtype = input.meta['tensor_meta'].dtype
                val = IRFullTensor(shape=shape, requires_grad=False, dtype=dtype, name=input.name)
            frame.add_var(input.name, val, graph_arg=idx)

        input_val = [frame.get_var(input.name) for input in inputs]

        # add activations to frame, including call_func/call_method output and final output
        # call_module corresponds to leaf torch.nn.module
        from cube.graph.parser.fx.concrete_trace_utils.kwargs_shape_prop.kwargs_shape_prop import TensorMetadata
        activation_op_strs = {'call_function', 'output', 'call_method', 'call_module'}
        activation_nodes = [node for node in module.graph.nodes if node.op in activation_op_strs]
        def parse_complex_out(meta_out):
            if isinstance(meta_out, TensorMetadata):
                shape = meta_out.shape
                assert shape == torch.Size([]), f'{meta_out}'
                return IRFullTensor(shape=shape, requires_grad=meta_out.requires_grad, dtype=meta_out.dtype)
            elif isinstance(meta_out, dict):
                ret = {}
                for k, v in meta_out.items():
                    ret[k] = parse_complex_out(v)
                return ret
            else:
                return meta_out
        for node in activation_nodes:
            if hasattr(node, 'meta') and node.meta.get('tensor_meta'):
                assert isinstance(node, torch.fx.Node)
                if isinstance(node.meta['tensor_meta'], TensorMetadata):
                    meta_outs = (node.meta['tensor_meta'],)
                else:
                    meta_outs = node.meta['tensor_meta']
                vals = list()
                for meta_out in meta_outs:
                    if isinstance(meta_out, TensorMetadata):
                        shape = meta_out.shape
                        shape = FxModuleParser.shape_refine(shape)
                        dtype = meta_out.dtype
                        requires_grad = meta_out.requires_grad
                        val = IRFullTensor(shape=shape, requires_grad=requires_grad, dtype=dtype, name=node.name)
                    else:
                        val = IRObject(value=parse_complex_out(meta_out))
                    vals.append(val)
                if len(vals) == 1:
                    frame.add_var(node.name, vals[0])
                else:
                    frame.add_var(node.name, vals)
            else:
                frame.add_var(node.name, IRObject())

        # handle nodes
        all_ir_nodes: List[IRFwOperation] = list()
        total_node_num = len(module.graph.nodes)
        for nidx, node in enumerate(module.graph.nodes):
            _logger.info(f'[{nidx}/{total_node_num}] parsing node {node}...')
            ir_nodes = FxModuleParser.parse_node(node, module, dynamic_shape, frame)
            if ir_nodes is not None:
                all_ir_nodes += ir_nodes

        # output_nodes = [node for node in module.graph.nodes if node.op == 'output']
        # assert len(output_nodes) == 1, f"get mutiple {len(all_ir_nodes)} output nodes"
        # output_val = frame.get_var(output_nodes[0].name)
        output_val = [frame.get_var(node.name) for node in module.graph.nodes if node.op == 'output']

        if save_content:
            attr_savedir = Path(attr_savedir)
            frame.save_attr_content(attr_savedir / FxModuleParser.ATTR_CONTENT_FILE)
            frame.save_attr_map(attr_savedir / FxModuleParser.ATTR_MAP_FILE)

        frame.pop_var()
        frame.pop_attr()

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
    def parse_node(node: torch.fx.Node, module, dynamic_shape: bool, frame: Frame) -> List[IRFwOperation]:
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
                return FxModuleParser.parse_prim_function_method(node, module, dynamic_shape, frame)
            if node_type == FxNodeKind.PrimGetAttr:
                return FxModuleParser.parse_prim_attr_node(node, module, frame)
            if node_type == FxNodeKind.PrimCallModule:
                return FxModuleParser.parse_prim_module(node, module, frame)

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
    def parse_prim_module(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRFwOperation]:
        prim_module = FxModuleParser.fetch_attr(module, node.target)
        if prim_module.__class__.__module__.startswith('torch.nn.modules'):
            raise RuntimeError(f'{prim_module.__class__.__module__} can not be parsed as leaf nodes')
        else:
            raise RuntimeError(f'unknown module: {prim_module.__class__.__module__}')

    @staticmethod
    def parse_prim_function_method(node: torch.fx.Node, module: torch.fx.GraphModule, dynamic_shape: bool, frame: Frame) -> List[IRFwOperation]:
        # get signature
        fsig = FxModuleParser._get_qualified_name(node.target, node)

        # get inputs
        input_vals = [get_complex_data(val, frame) for val in node.args]
        kwargs = {key: get_complex_data(val, frame) for key, val in node.kwargs.items()}

        return FxModuleParser._parse_node(fsig, node, input_vals, kwargs, dynamic_shape, frame)

    @staticmethod
    def _parse_node(fsig: str, node: torch.fx.Node, input_vals: list, kwargs: dict, dynamic_shape: bool, frame: Frame) -> List[IRFwOperation]:
        # map to IR operator
        if SignFx2Op.exist(fsig):
            ir_node = SignFx2Op.map(fsig)(*input_vals, **kwargs)
        else:
            # FIXME: handle cases for IRObject in kwargs
            # case1: unknown torch operator
            if FxModuleParser._is_torch_autograd_op(node, frame, fsig):
                _logger.warning(f'Find unknown pytorch operation: {fsig}')
                fname = fsig.split('.')[-1] if '.' in fsig else fsig
                ir_node = IRFwOperation(fname, fsig, input_vals, 1, **kwargs)
            # case2: python runtime function
            else:
                _logger.warning(f'Set python runtime function: {fsig}')
                ir_node = IRPyFunc(fsig, input_vals, [IRObject()], **kwargs)

        if isinstance(ir_node, IRCell):
            comment = str(node.meta.get('frame_record', ''))
            if comment:
                ir_node.comment = comment

        ir_nodes = []
        if isinstance(ir_node, IRCell):
            ir_nodes.append(ir_node)
            if len(ir_node.outputs()) > 1:
                vals = frame.get_var(node.name)
                assert len(vals) == len(ir_node.outputs()), f'{vals}, {ir_node.outputs()}'
                for i in range(len(vals)):
                    ir_node.set_output(i, vals[i])
            elif ir_node.output(0).value is not None:
                if dynamic_shape:
                    frame.set_var(node.name, ir_node.output(0))
                    ir_node.output(0).name = node.name
                else:
                    # if use static shape graph, all IRObject will be converted to real traced value.
                    # the ir_node will be folded and not appeared in the final graph
                    frame.set_var(node.name, ir_node.output(0).value)
                    ir_nodes.pop(-1)
            else:
                output_val = frame.get_var(node.name)
                if isinstance(ir_node, IRDimops):
                    ir_node.infer_shape()
                    assert output_val.shape == ir_node.output(0).shape, (
                        f'find shape inference not match: {output_val.shape} vs {ir_node.output(0).shape}'
                        f'\nnode: {node}'
                    )
                ir_node.set_output(0, output_val)
        else:
            frame.set_var(node.name, ir_node)

        _logger.info(f'parsing result: {ir_node}')
        return ir_nodes

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
            dtype = node.meta['tensor_meta'].dtype
            requires_grad = node.meta['tensor_meta'].requires_grad

            # check if existing param
            if requires_grad and frame.has_attr_value(node.target):  # existing param
                prev_tensor_name = frame.get_attr_key(node.target)
                frame.add_var(tensor_name, frame.get_var(prev_tensor_name))
            else:  # new param / activation
                ir_tensor = IRFullTensor(tensor_shape, tensor_name, requires_grad=requires_grad, dtype=dtype)
                if requires_grad:  # case for registered parameters
                    ir_tensor.as_param()
                else:  # case for registered buffers
                    ir_tensor.as_buffer()
                frame.add_var(tensor_name, ir_tensor)
                value = FxModuleParser.fetch_attr(module, node.target)
                frame.add_attr_content(ir_tensor.tid, value)
                frame.add_attr_map(ir_tensor.name, node.target)
        else:
            var = FxModuleParser.fetch_attr(module, node.target)
            frame.add_var(tensor_name, var)

        return None

    @staticmethod
    def parse_prim_output_node(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRCell]:
        assert len(node.args) == 1 and len(node.kwargs) == 0
        ir_nodes = []

        def generate_outputs(val: Any) -> Any:
            """Support complex data type of List, Tuple, Dict, Tensor/Object"""
            if isinstance(val, list):
                return list(generate_outputs(item) for item in val)
            if isinstance(val, tuple):
                return tuple(generate_outputs(item) for item in val)
            if isinstance(val, dict):
                return {generate_outputs(key) : generate_outputs(value) for key, value in val.items()}
            if isinstance(val, torch.fx.Node):
                return frame.get_var(val.name)
            # for other types like int, float, ...
            return val
        output = generate_outputs(node.args[0])

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
        return sig

    @staticmethod
    def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
        if getattr(orig_method, '__name__', None) == 'apply' and isinstance(getattr(orig_method, '__self__', None), Type) \
            and issubclass(orig_method.__self__, torch.autograd.Function):
            # for torch.autograd.Function
            return f'{orig_method.__self__.__module__}.{orig_method.__self__.__name__}'

        name = orig_method.__name__
        module = orig_method.__module__
        # if hasattr(orig_method, '__qualname__') and isinstance(orig_method.__qualname__, str):
        #     # if there has '.' in '__qualname__', it means this function is in a nested structure,
        #     #
        #     # for example, it is a method / function in a class:
        #     # torch.nn.Linear.forward.__module__ = torch.nn
        #     # torch.nn.Linear.forward.__name__ = forward
        #     # torch.nn.Linear.forward.__qualname__ = Linear.forward
        #     #
        #     # And in fx.node qualified name creating rule, the module also should include the class name,
        #     # in this example, the returned module should be `torch.nn.Linear`.
        #     splited_names = orig_method.__qualname__.split('.')
        #     class_name, name = splited_names[:-1], splited_names[-1]
        #     module = '.'.join([module] + class_name)
        if module == 'torch.autograd.grad_mode' and name in ['__enter__', '__exit__']:
            return 'torch.autograd.grad_mode.no_grad'
        if module is not None:
            return module
        if hasattr(orig_method, '__qualname__')\
            and isinstance(orig_method.__qualname__, str) and orig_method.__qualname__.startswith('_VariableFunctionsClass.'):
            return 'torch._C._VariableFunctions'
        for guess in [torch, getattr(torch.nn, 'functional')]:
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
