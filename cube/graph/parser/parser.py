import torch
import enum
import re
from typing import List, Tuple, Optional, Union

from cube.graph import IRFwOperation
from cube.graph.tensor import IRFullTensor
from cube.graph.parser.frame import Frame
from cube.graph.parser.mapping import Sign2Op


_refmodule = torch.nn.Module()


class ScriptNodeKind(enum.Enum):
    PrimGetAttr = 1
    PrimCallMethod = 2
    PrimCallFunction = 3  # -> the parser may end here
    PrimConstant = 4
    AtenOp = 5            # -> the parser may end here
    PrimIf = 6            # dynamic
    PrimListConstruct = 7
    PrimListUnpack = 8
    PrimTupleUnpack = 9
    PrimPythonOp = 10


class ScriptModuleParser:

    @staticmethod
    def parse_module(module,
                     input_shapes: Optional[ Tuple[List[int],] ] = None,
                     frame: Frame = Frame()) \
        -> Tuple[List[IRFullTensor], List[IRFwOperation], List[IRFullTensor]]:
        """
        The overall entry to parse a torchscript graph module
        """
        frame.push()

        # handle graph input -- Assuming all the inputs are tensors
        input_var_name = [input.debugName() for input in module.graph.inputs()]
        for index, var_name in enumerate(input_var_name[1:]): # omit self
            frame.add_var(var_name, IRFullTensor(name=var_name), graph_arg=index)
        input_val = [frame.get_var(var_name) for var_name in input_var_name[1:]]

        # handle input shape
        if input_shapes:
            if len(input_val) != len(input_shapes):
                raise RuntimeError(
                    f"Module {module.original_name} input shape mismatch (got {len(input_shapes)} != {len(input_val)})"
                )
            for shape, val in zip(input_shapes, input_val):
                if isinstance(val, IRFullTensor):
                    val.shape = shape

        all_ir_nodes: List[IRFwOperation] = list()
        for node in module.graph.nodes():
            # debug info
            # print(f'on parsing:\n\t{node}')
            ir_nodes = ScriptModuleParser.parse_node(node, module, frame)
            # print(f'> {frame}')
            # print(f'> {ir_nodes}')
            # _ = input('>>>')
            if len(ir_nodes) != 0:
                for ir_node in ir_nodes:
                    ret = ir_node.infer_shape()
                    if not ret:
                        print(f'warning: {ir_node} cannot infer shape')
                all_ir_nodes += ir_nodes
        
        # handle graph output -- Assuming all the output are tensors
        output_var_name = [output.debugName() for output in module.graph.outputs()]
        output_val = [frame.get_var(var_name) for var_name in output_var_name]

        frame.pop()
        return input_val, all_ir_nodes, output_val

    @staticmethod
    def ntype(node: torch._C.Node):
        if node.kind() == 'prim::GetAttr':
            return ScriptNodeKind.PrimGetAttr
        if node.kind() == 'prim::CallMethod':
            return ScriptNodeKind.PrimCallMethod
        if node.kind() == 'prim::CallFunction': # the op call
            return ScriptNodeKind.PrimCallFunction
        if node.kind() == 'prim::Constant':
            return ScriptNodeKind.PrimConstant
        if node.kind().startswith('aten::'):
            return ScriptNodeKind.AtenOp
        if node.kind() == 'prim::If':
            return ScriptNodeKind.PrimIf
        if node.kind() == 'prim::ListUnpack':
            return ScriptNodeKind.PrimListUnpack
        if node.kind() == 'prim::ListConstruct':
            return ScriptNodeKind.PrimListConstruct
        if node.kind() == 'prim::TupleUnpack':
            return ScriptNodeKind.PrimTupleUnpack
        if node.kind() == 'prim::PythonOp':
            return ScriptNodeKind.PrimPythonOp
        raise RuntimeError(f"Unkown node kind {node.kind()} from torchscript module")

    @staticmethod
    def parse_node(node: torch._C.Node, module, frame: Frame) -> List[IRFwOperation]:
        """
        Parse the node and return the IRFwOperation nodes
        """
        try:
            node_type = ScriptModuleParser.ntype(node)
        except RuntimeError:
            print(module.graph)
            raise RuntimeError("Unsupported node kind {node.kind()} found in parsing. See above graph.")
        if node_type == ScriptNodeKind.PrimCallFunction:
            return ScriptModuleParser.parse_prim_function_node(node, module, frame)
        if node_type == ScriptNodeKind.AtenOp:
            return ScriptModuleParser.parse_aten_node(node, module, frame)
        if node_type == ScriptNodeKind.PrimCallMethod:
            return ScriptModuleParser.parse_prim_method_node(node, module, frame)
        if node_type == ScriptNodeKind.PrimGetAttr:
            return ScriptModuleParser.parse_prim_attr_node(node, module, frame)
        if node_type == ScriptNodeKind.PrimConstant:
            return ScriptModuleParser.parse_prim_constant_node(node, module, frame)
        if node_type == ScriptNodeKind.PrimListConstruct:
            return ScriptModuleParser.parse_prim_list_construct_node(node, module, frame)
        if node_type == ScriptNodeKind.PrimListUnpack:
            return ScriptModuleParser.parse_prim_listunpack_node(node, module, frame)
        if node_type == ScriptNodeKind.PrimTupleUnpack:
            return list() # tuple unpack should only be used in prim function node
        if node_type == ScriptNodeKind.PrimPythonOp:
            return ScriptModuleParser.parse_prim_python_op_node(node, module, frame)
        raise NotImplementedError(f"Un-supported node type {node_type}")

    @staticmethod
    def parse_prim_function_node(node, module, frame: Frame) -> List[IRFwOperation]:
        """
        parse node like:
            Tensor = prim::CallFunction(%5, %input.1, %3, %4)
            %5 : Function = prim::Constant[name="linear"]()
        """
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]

        outputs: List[Union[torch._C.Value, IRFullTensor]] = list()
        for output in node.outputs():
            # unpack the output type
            if isinstance(output.type(), torch._C.TupleType):
                for unpack_node in module.graph.nodes():
                    if ScriptModuleParser.ntype(unpack_node) == ScriptNodeKind.PrimTupleUnpack:
                        if output in unpack_node.inputs():
                            ScriptModuleParser.parse_prim_tupleunpack_node(unpack_node, module, frame)
                            break
                tuple_outputs = frame.get_var(output.debugName())
                outputs += tuple_outputs
            else:
                outputs.append(output)

        # handle function node
        fnode = node.inputsAt(0).node()
        if not ScriptModuleParser.ntype(fnode) == ScriptNodeKind.PrimConstant:
            raise RuntimeError(f"Found unexpected function call node: {fnode}")
        fsig = frame.get_var(inputs[0].debugName())

        # handle inputs
        input_vals = list()
        for index, input in enumerate(inputs[1:]):
            var_name = input.debugName()
            val = frame.get_var(var_name)
            input_vals.append(val)

        ir_node = Sign2Op.map(fsig)(inputs=input_vals, n_outputs=len(outputs))
        if len(ir_node.outputs()) != len(outputs):
            raise RuntimeError(
                f"Parse fail: {fsig} has {len(outputs)} outputs != pre-defined {len(ir_node.outputs())}"
            )

        # handle outputs
        for index, output in enumerate(outputs):
            if isinstance(output, IRFullTensor):
                ir_node.set_output(index, output)
            else:
                frame.add_var(output.debugName(), ir_node.outputs(index))

        return [ir_node]

    @staticmethod
    def parse_aten_node(node, module, frame: Frame) -> List[IRFwOperation]:
        """
        Parse script module node like:
            %13 : Tensor = aten::gt(%output1.1, %output2.1)
        """
        fsig = node.kind()
        fsig = re.sub('aten::', 'torch.', fsig)
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]

        # handle inputs:
        input_val = list()
        for input in inputs:
            var_name = input.debugName()
            val = frame.get_var(var_name)
            input_val.append(val)

        # create IR node
        ir_node = Sign2Op.map(fsig)(inputs=input_val, n_outputs=len(outputs))
        if len(ir_node.outputs()) != len(outputs):
            raise RuntimeError(
                f"Parse fail: {fsig} has {len(outputs)} outputs != pre-defined {len(ir_node.outputs())}"
            )

        # handle outputs
        for index, output in enumerate(outputs):
            frame.add_var(output.debugName(), ir_node.outputs(index))

        return [ir_node]

    @staticmethod
    def parse_prim_method_node(node, module, frame: Frame) -> List[IRFwOperation]:
        """
        Parse script module node like:
            %output.1 : Tensor = prim::CallMethod[name="forward"](%2, %x.1)

        prim::CallMethod has a underlying submodule
        """
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]

        # forward
        label = node.s('name')
        if label != 'forward':
            raise RuntimeError(f"{node} is calling function {label} that is not `forward`")

        # handle inputs -- in stack with reverse order
        for input in inputs[1:][::-1]:
            var_name = input.debugName()
            val = frame.get_var(var_name)
            frame.push_param(var_name)

        # print(f'> {frame}')

        # recursively parse the module
        module_label = node.inputsAt(0).node().s('name')
        call_module = getattr(module, module_label)
        _, ir_nodes, outputs_val = ScriptModuleParser.parse_module(call_module, frame=frame)

        # pop out the frame
        frame.pop_param(times=len(inputs)-1)

        # handle outputs
        outputs = [output for output in node.outputs()]
        for output, val in zip(outputs, outputs_val):
            frame.add_var(output.debugName(), val)

        return ir_nodes

    @staticmethod
    def parse_prim_attr_node(node, module, frame) -> List[None]:
        """
        Parse script module node like:
            %2 :__torch__.torch.nn.modules.linear.___torch_mangle_0.Linear = prim::GetAttr[name="linear1"](%self)
            %3 : Tensor = prim::GetAttr[name="weight"](%self)
        The __torch__.torch.nn.modules.* will be ignored

        This will add frame with the variable name and it's value

        The value can be:
            1). (IRFullTensor) the tensor edge in graph
            2). (str code) symbolic value based on runtime info (e.g., self.training)
            3). (str) Function or torch.nn.moudles

        Returns:
            Empty list
        """
        global _refmodule
        if node.inputsAt(0).debugName() != 'self':
            raise RuntimeError(f"Fail to parse {node} due to missing %self")

        label = node.s('name')
        var_name = node.outputsAt(0).debugName()
        dtype = node.outputsAt(0).type().str()

        # this usually means weight (nn.Parameter in torch)
        if dtype == 'Tensor':
            shape = list(getattr(module, label).shape)
            ir_tensor = IRFullTensor(name=label, shape=shape)
            frame.add_var(var_name, ir_tensor)
        # symbolic attributes
        elif dtype in ['bool', 'int', 'float']:
            if hasattr(_refmodule, label):
                val = 'self.' + label
            else:
                val = getattr(module, label)
            # print(f'get: var_name {var_name}: {val}')
            frame.add_var(var_name, val)
        # NoneType
        elif dtype == 'NoneType':
            frame.add_var(var_name, None)
        # module name or other things cannot handle
        else:
            frame.add_var(var_name, label)
        return list()

    @staticmethod
    def parse_prim_constant_node(node, module, frame) -> List[None]:
        """
        Parse script module node like:
            %6 : Function = prim::Constant[name="dropout"]()
            %5 : bool = prim::Constant[value=0]()
        
        This will add frame with the variable name and it's value

        Returns:
            Empty list
        """
        if len(list(node.inputs())) != 0:
            raise RuntimeError(f"prim::Constant node: {node} has inputs")
        var_name = node.outputsAt(0).debugName()
        dtype = node.outputsAt(0).type().str()

        if dtype == 'Function':
            signature = repr(node.outputsAt(0).type())
            if '__torch__.' in signature:
                signature = re.sub('__torch__.', '', signature)
            frame.add_var(var_name, signature)
        else:
            val = node.outputsAt(0).toIValue()
            frame.add_var(var_name, val)
        return list()

    @staticmethod
    def parse_prim_if_node(node, module, frame: Frame) -> List[IRFwOperation]:
        """
        Parse script module node like 
            %output2 : Tensor = prim::If(%15) # /tmp/ipykernel_27188/2459450745.py:13:8
                block0():
                    -> (%output1.1)
                block1():
                    -> (%output2.1)
        """
        raise NotImplementedError("Dynamic Graph is not supported yet")

    @staticmethod
    def parse_prim_list_construct_node(node, module, frame: Frame) -> List[None]:
        """
        Parse script module node like
            %8 : int[] = prim::ListConstruct(%3)
        """
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]
        assert len(outputs) == 1
        output = outputs[0]
        out_val = list()
        for input in inputs:
            out_val.append(frame.get_var(input.debugName()))
        frame.add_var(output.debugName(), out_val)
        return list()

    @staticmethod
    def parse_prim_listunpack_node(node, module, frame: Frame) -> List[None]:
        raise NotImplementedError

    @staticmethod
    def parse_prim_tupleunpack_node(node, module, frame) -> List[None]:
        """
        Parse script module node like:
            %q.1 : Tensor, %k.1 : Tensor, %v.1 : Tensor = prim::TupleUnpack(%11)
        """
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]
        if len(inputs) != 1:
            raise RuntimeError("Find UnpackTuple has more than one input")
        if len(outputs) == 1:
            raise RuntimeError("Find UnpackTuple has only one output")
        tuple_outs = list()
        for output in outputs:
            dtype = output.type().str()
            var_name = output.debugName()
            if dtype == 'Tensor':
                ir_tensor = IRFullTensor(name=var_name)
                tuple_outs.append(ir_tensor)
                frame.add_var(var_name, ir_tensor)
            else:
                raise NotImplementedError
        frame.add_var(inputs[0].debugName(), tuple_outs)
        return list()

    @staticmethod
    def parse_prim_python_op_node(node, module, frame):
        raise NotImplementedError("Cannot support torch.jit.ignore")
        print(dir(node))

    @staticmethod
    def flatten(smodule, depth=0):
        """
        Flatten the recursive script module to function and aten primitives
        """
        # stashed_module = list()
        inputs = [input for input in smodule.graph.inputs()]
        print('    '*depth, f'graph inputs: {inputs}')
        if len(list(smodule.children())) == 0:
            for node in smodule.graph.nodes():
                print('    '*depth, node)
        else:
            for node in smodule.graph.nodes():
                print('    '*depth, node)
                if node.kind() == 'prim::CallMethod':
                    label = node.inputsAt(0).node().s('name')
                    submodule = getattr(smodule, label)
                    ScriptModuleParser.flatten(submodule, depth+1)


