import torch
import enum
import re
from typing import List, Tuple

from cube.graph.graph import IROperation, IRTensor
from cube.graph.frame import Frame

class ScriptNodeKind(enum.Enum):
    PrimGetAttr = 1
    PrimCallMethod = 2
    PrimCallFunction = 3  # -> the parser may end here
    PrimConstant = 4
    AtenOp = 5            # -> the parser may end here
    PrimIf = 6            # dynamic


class ScriptModuleParser:

    @staticmethod
    def parse_module(module,
                     frame: Frame = Frame()) \
        -> Tuple[List[IRTensor], List[IROperation], List[IRTensor]]:
        """
        The overall entry to parse a torchscript graph module
        """
        frame.push()

        # handle graph input -- Assuming all the inputs are tensors
        input_var_name = [input.debugName() for input in module.graph.inputs()]
        # [1:] is to omit self
        for index, var_name in enumerate(input_var_name[1:]):
            frame.add_var(var_name, IRTensor(name=var_name), graph_arg=index)
        input_val = [frame.get_var(var_name) for var_name in input_var_name[1:]]

        all_ir_nodes: List[IROperation] = list()
        for node in module.graph.nodes():
            # debug info
            # print(f'on parsing:\n\t{node}')
            ir_nodes = ScriptModuleParser.parse_node(node, module, frame)
            # print(f'> {frame}')
            # print(f'> {ir_nodes}')
            # _ = input('>>>')
            if len(ir_nodes) != 0:
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
        raise RuntimeError(f"Unkown node kind {node.kind()} from torchscript module")

    @staticmethod
    def parse_node(node: torch._C.Node, module, frame: Frame) -> List[IROperation]:
        """
        Parse the node and return the IROperation nodes
        """
        node_type = ScriptModuleParser.ntype(node)
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

    @staticmethod
    def parse_prim_function_node(node, module, frame: Frame) -> List[IROperation]:
        """
        parse node like:
            Tensor = prim::CallFunction(%5, %input.1, %3, %4)
            %5 : Function = prim::Constant[name="linear"]()
        """
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]

        # handle function node
        fnode = node.inputsAt(0).node()
        if not ScriptModuleParser.ntype(fnode) == ScriptNodeKind.PrimConstant:
            raise RuntimeError(f"Found unexpected function call node: {fnode}")
        fsig = frame.get_var(inputs[0].debugName())

        # create IR node
        ir_node = IROperation(
            signature = fsig,
            name = fnode.s('name'),
            input_length=len(inputs) - 1,
            output_length=len(outputs),
        )

        # handle inputs -- in stack with reverse order
        for index, input in enumerate(inputs[1:]):
            var_name = input.debugName()
            val = frame.get_var(var_name)
            ir_node.set_input(index, val)
        
        # handle outputs
        for index, output in enumerate(outputs):
            frame.add_var(output.debugName(), ir_node.outputs(index))

        return [ir_node]

    @staticmethod
    def parse_aten_node(node, module, frame: Frame) -> List[IROperation]:
        """
        Parse script module node like:
            %13 : Tensor = aten::gt(%output1.1, %output2.1)
        """
        fsig = node.kind()
        fsig = re.sub('aten::', 'torch.', fsig)
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]

        # create IR node
        ir_node = IROperation(
            signature = fsig,
            name = fsig,
            input_length = len(inputs),
            output_length = len(outputs)
        )

        # handle inputs
        inputs = [input for input in node.inputs()]
        # in stack with reverse order
        for index, input in enumerate(inputs):
            var_name = input.debugName()
            val = frame.get_var(var_name)
            ir_node.set_input(index, val)

        # handle outputs
        outputs = [output for output in node.outputs()]
        for index, output in enumerate(outputs):
            frame.add_var(output.debugName(), ir_node.outputs(index))

        return [ir_node]

    @staticmethod
    def parse_prim_method_node(node, module, frame: Frame) -> List[IROperation]:
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
        _, ir_nodes, outputs_val = ScriptModuleParser.parse_module(call_module, frame)

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
            1). (IRTensor) the tensor edge in graph
            2). (str code) symbolic value based on runtime info (e.g., self.training)
            3). (str) Function or torch.nn.moudles

        Returns:
            Empty list
        """
        if node.inputsAt(0).debugName() != 'self':
            raise RuntimeError(f"Fail to parse {node} due to missing %self")

        label = node.s('name')
        var_name = node.outputsAt(0).debugName()
        dtype = node.outputsAt(0).type().str()

        # this usually means weight (nn.Parameter in torch)
        if dtype == 'Tensor':
            shape = list(getattr(module, label).shape)
            ir_tensor = IRTensor(name=label, shape=shape)
            frame.add_var(var_name, ir_tensor)
        # symbolic attributes
        elif dtype in ['bool', 'int', 'float']:
            frame.add_var(var_name, 'self.' + label)
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
    def parse_prim_if_node(node, module, frame: Frame) -> List[IROperation]:
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
                ntype = ScriptModuleParser.ntype(node)
                print('    '*depth, node)
                if ntype == ScriptNodeKind.PrimCallMethod:
                    label = node.inputsAt(0).node().s('name')
                    submodule = getattr(smodule, label)
                    ScriptModuleParser.flatten(submodule, depth+1)


