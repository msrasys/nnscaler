import torch
import enum
import re
from typing import Any, List, Tuple, Optional

from cube.ir.operator import IRFwOperation
from cube.ir.tensor import IRFullTensor
import cube.ir as ir
from cube.graph.parser.frame import Frame
from cube.graph.parser.mapping import DType2IRDType
from cube.graph.parser.mappingfx import SignFx2Op

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
            raise RuntimeError(f"Module {module.original_name} input shape mismatch (got {len(input_shapes)} != {len(inputs)})")

        ## shape propagation
        default_dtype = torch.get_default_dtype()
        kDefaultType = DType2IRDType.map(default_dtype) # TODO specify dtype
        sample_inputs = [torch.ones(shape, dtype=default_dtype) for shape in input_shapes]
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(module).propagate(*sample_inputs)
        for node in module.graph.nodes:
            print(node.name, node.meta['tensor_meta'].dtype, node.meta['tensor_meta'].shape)

        # handle graph input -- Assuming all the inputs are tensors
        for idx, input in enumerate(inputs):
            assert isinstance(input, torch.fx.Node)
            shape = None if input_shapes is None else input_shapes[idx]
            dtype = kDefaultType
            val = IRFullTensor(shape=shape, requires_grad=False, dtype=dtype, name=input.name)
            frame.add_var(input.name, val, graph_arg=idx)
        input_val = [frame.get_var(input.name) for input in inputs]

        # add activations to frame, including call_func output and final output
        activation_op_strs = {'call_function', 'output', 'call_method'}
        activation_nodes = [node for node in module.graph.nodes if node.op in activation_op_strs]
        for node in activation_nodes:
            assert isinstance(node, torch.fx.Node)
            shape = node.meta['tensor_meta'].shape
            shape = FxModuleParser.shape_refine(shape)
            dtype = DType2IRDType.map(node.meta['tensor_meta'].dtype)
            val = IRFullTensor(shape=shape, requires_grad=True, dtype=dtype, name=node.name)
            frame.add_var(node.name, val)

        # handle nodes
        all_ir_nodes: List[IRFwOperation] = list()
        for node in module.graph.nodes:
            ir_nodes = FxModuleParser.parse_node(node, module, frame)
            all_ir_nodes += ir_nodes

        # handle outputs
        output_nodes = [node.all_input_nodes for node in module.graph.nodes if node.op == 'output']
        print(f'outputs = {output_nodes}')
        output_var_name = [output.name for output in [item for sublist in output_nodes for item in sublist]]
        output_val = [frame.get_var(var_name) for var_name in output_var_name]

        # flatten output_val
        outputs = list()
        for val in output_val:
            if isinstance(val, list):
                outputs += val
            else:
                outputs.append(val)
        output_val = outputs

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
        # if node.kind() == 'prim::CallMethod':
        #     return FxNodeKind.PrimCallMethod
        # if node.kind() == 'prim::CallFunction': # the op call
        #     return FxNodeKind.PrimCallFunction
        # if node.kind() == 'prim::Constant':
        #     return FxNodeKind.PrimConstant
        # if node.kind().startswith('aten::'):
        #     return FxNodeKind.AtenOp
        # if node.kind() == 'prim::If':
        #     return FxNodeKind.PrimIf
        # if node.kind() == 'prim::Loop':
        #     return FxNodeKind.PrimLoop
        # if node.kind() == 'prim::ListConstruct':
        #     return FxNodeKind.PrimListConstruct
        # if node.kind() == 'prim::TupleConstruct':
        #     return FxNodeKind.PrimListConstruct
        # if node.kind() == 'prim::ListUnpack':
        #     return FxNodeKind.PrimListUnpack
        # if node.kind() == 'prim::TupleUnpack':
        #     return FxNodeKind.PrimListUnpack
        # if node.kind() == 'prim::PythonOp':
        #     return FxNodeKind.PrimPythonOp
        # if node.kind() == 'prim::device':
        #     return FxNodeKind.PrimDevice
        raise RuntimeError(f"Unkown node kind {node.kind()} from torchscript module")

    @staticmethod
    def parse_node(node: torch.fx.Node, module, frame: Frame) -> List[IRFwOperation]:
        # print("### parse_node {}".format(node))
        """
        Parse the node and return the IRFwOperation nodes
        """
        node_type = FxModuleParser.ntype(node)
        try:
            if node_type == FxNodeKind.Placeholder:
                return []
            if node_type == FxNodeKind.Output:
                return []

            if node_type == FxNodeKind.PrimCallFunction:
                return FxModuleParser.parse_prim_function_node(node, module, frame)
            if node_type == FxNodeKind.PrimCallMethod:
                return FxModuleParser.parse_prim_method_node(node, module, frame)
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
    def parse_prim_module(node: torch.fx.Node, module, frame: Frame) -> List[IRFwOperation]:
        """
        :param node:
        :param module:
        :param frame:
        :return:
        """
        raise RuntimeError(f"parse_prim_module needs update")

        input_nodes = node.all_input_nodes
        for input_node in input_nodes:
            var_name = input_node.name
            val = frame.get_var(var_name)
            frame.push_param(var_name)

        # TODO skip self_module in torchscript
        call_module = module

        label = node.name
        node_target_stack = node.target.split('.')

        #TODO check leaf module, iterate if not

        leaf_module = module
        for node_target_stack_iter in node_target_stack:
            leaf_module = getattr(leaf_module, node_target_stack_iter)

        _, ir_nodes, outputs_val = FxModuleParser.parse_nn_module(node, leaf_module, frame=frame)

        # pop out the frame
        frame.pop_param(times=len(input_nodes) - 1)

        # # handle outputs
        # # TODO outputs vs output
        # outputs = [node]
        # # outputs = [output for output in node.outputs()]
        # for output, val in zip(outputs, outputs_val):
        #     frame.add_var(output.name, val)

        return ir_nodes

    @staticmethod
    def parse_nn_module(node: torch.fx.Node, method: torch.nn.Module, frame: Frame):
        """
        Parse module method
        """

        input_var_name = [input_node.name for input_node in node.all_input_nodes]
        input_val = [frame.get_var(var_name) for var_name in input_var_name]

        all_ir_nodes: List[IRFwOperation] = list()

        # handle graph output

        fsig = type(method).__name__
        # ir_node = SignFx2Op.map(fsig)(input=input_val)
        func = SignFx2Op.map(fsig)
        weights_names = None #TODO obtain parameter name list
        if weights_names is not None:
            for idx, weight_name in enumerate(weights_names):
                #create FullTensor
                weight = getattr(method, weight_name)
                if weight is not None:
                    weight_fulltensor_name = node.name + "-" + weight_name
                    weight_fulltensor = IRFullTensor(weight.shape, weight_fulltensor_name, requires_grad=False)
                    # frame.add_attr(weight_fulltensor_name, weight_fulltensor)
                    # tmp_ones_tensor = torch.ones(weight.shape, dtype=torch.get_default_dtype())
                    # frame.add_attr_content(weight_fulltensor.tid, tmp_ones_tensor)
                    frame.add_var(weight_fulltensor_name, weight_fulltensor)
                    input_val.append(weight_fulltensor)
                else:
                    input_val.append(None)

        ir_node = func(inputs=input_val)
        all_ir_nodes += [ir_node]

        outputs = [node]
        output_var_name = [output.name for output in outputs]
        output_val = [frame.get_var(var_name) for var_name in output_var_name]

        # frame.pop_var()
        return input_val, all_ir_nodes, output_val

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
    def parse_prim_function_node(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRFwOperation]:
        """
        parse node like:
            Tensor = prim::CallFunction(%5, %input.1, %3, %4)
            %5 : Function = prim::Constant[name="linear"]()
            %12 : (Tensor, Tensor) = prim::CallFunction(%5, %x1.1, %x2.1)
        """
        # get signature
        fsig = FxModuleParser._get_qualified_name(node.target)
        print(f'parse_prim_function_node: {fsig}')

        # get inputs
        input_nodes = [input_node for input_node in node.args]
        input_vals = list()
        for index, input_node in enumerate(input_nodes):
            if isinstance(input_node, torch.fx.Node):
                var_name = input_node.name
                val = frame.get_var(var_name)
                input_vals.append(val)
            elif isinstance(input_node, (int, float)):
                input_vals.append(input_node)
            else:
                input_vals.append(None)

        # map to IR operator
        ir_node = SignFx2Op.map(fsig)(inputs=input_vals)

        # TODO gracefully set output
        output_name = node.name
        output_val = frame.get_var(output_name)
        ir_node.set_output(0, output_val)

        # # push output in the frame
        # # help: >>> a = torch._C.TupleType([torch._C.TensorType.getInferred()])
        # #     : >>> dir(a)
        # #     : >>> a.elements()  # [TensorType, TensorType]
        # cnt = 0
        # for output in node.outputs():
        #     if isinstance(output.type(), torch._C.TupleType):
        #         tuplen = len(output.type().elements())
        #         ir_output = [ir_node.output(idx) for idx in range(cnt, cnt + tuplen)]
        #         cnt += tuplen
        #     else:
        #         ir_output = ir_node.output(cnt)
        #         cnt += 1
        #     frame.add_var(output.debugName(), ir_output)
        #
        # if cnt != len(ir_node.outputs()):
        #     raise RuntimeError(
        #         f"Parse fail: {fsig} has {cnt} outputs != pre-defined {len(ir_node.outputs())}"
        #     )

        return [ir_node]

    @staticmethod
    def parse_prim_method_node(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRFwOperation]:
        # get signature
        fsig = FxModuleParser._get_qualified_name(node.target)
        print(f'parse_prim_method_node: {fsig}')

        # get inputs
        input_nodes = [input_node for input_node in node.args]
        input_vals = list()
        for index, input_node in enumerate(input_nodes):
            if isinstance(input_node, torch.fx.Node):
                var_name = input_node.name
                val = frame.get_var(var_name)
                input_vals.append(val)
            elif isinstance(input_node, (int, float)):
                input_vals.append(input_node)
            else:
                input_vals.append(None)

        # map to IR operator
        ir_node = SignFx2Op.map(fsig)(inputs=input_vals)

        output_name = node.name
        output_val = frame.get_var(output_name)
        ir_node.set_output(0, output_val)

        return [ir_node]

    @staticmethod
    def parse_prim_attr_node(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRFwOperation]:
        assert node is not None
        tensor_name = node.name

        tensor_shape = node.meta['tensor_meta'].shape
        # tensor_dtype = node.meta['tensor_meta'].dtype
        #TODO assume it is weight
        default_dtype = torch.get_default_dtype()
        kDefaultType = DType2IRDType.map(default_dtype)  # TODO specify dtype
        ir_tensor = IRFullTensor(tensor_shape, tensor_name, requires_grad=True, dtype=kDefaultType)
        ir_tensor.as_param()
        frame.add_var(tensor_name, ir_tensor)

        return list()


    from typing import Callable
    # import python_stubs.buildins
    @staticmethod
    def _get_qualified_name(func: Callable[..., Any]) -> str:
        # # things like getattr just appear in builtins
        # if getattr(builtins, func.__name__, None) is func:
        #     return func.__name__
        # TODO(yizhu1): find a general solution
        if isinstance(func, str):
            for module, module_name in [(torch, 'torch'), (torch.Tensor, 'torch.Tensor')]:
                lib_func = getattr(module, func, None)
                if lib_func is not None and callable(lib_func):
                    return f'{module_name}.{func}'
            raise RuntimeError(f'cannot find module for {func}')
        name = func.__name__
        module = FxModuleParser._find_module_of_method(func)
        module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
        return f'{module}.{name}'

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