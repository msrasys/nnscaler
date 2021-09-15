import torch
import enum
from cube.graph.graph import IROperation

from typing import Dict, List

class ScriptNodeKind(enum.Enum):
    PrimGetAttr = 1
    PrimCallMethod = 2
    PrimCallFunction = 3  # -> the parser may end here
    PrimConstant = 4
    AtenOp = 5            # -> the parser may end here
    PrimIf = 6            # dynamic


class ScriptModuleParser:

    @staticmethod
    def parse_module(module: torch.jit.RecursiveModule) -> List[IROperation]:
        """
        The overall entry to parse a torchscript graph module
        """
        all_ir_nodes: List[IROperation] = list()
        for node in module.graph.nodes():
            ir_nodes = None
            node_type = ScriptModuleParser.ntype(node)
            if node_type == ScriptNodeKind.PrimCallFunction:
                ir_nodes = ScriptModuleParser.parse_prim_function_node(node, module)
            if node_type == ScriptNodeKind.AtenOp:
                ir_nodes = ScriptModuleParser.parse_aten_node(node, module)
            if node_type == ScriptNodeKind.PrimCallMethod:
                ir_nodes = ScriptModuleParser.parse_prim_method_node(node, module)
            if ir_nodes is not None:
                all_ir_nodes.append(ir_nodes)
        return all_ir_nodes

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
    def parse_prim_function_node(node, module) -> IROperation:
        """
        parse node like:
            Tensor = prim::CallFunction(%5, %input.1, %3, %4)
            %5 : Function = prim::Constant[name="linear"]()
        """
        fnode = node.inputsAt(0).node()  # function node
        if not ScriptModuleParser.ntype(fnode) == ScriptNodeKind.PrimConstant:
            raise RuntimeError(f"Found unexpected function call node: {fnode}")
        var_name, fsig, _ = ScriptModuleParser.parse_prim_constant_node(fnode, module)
        input_length = len(list(fnode.inputs())) - 1  # -1 for the first signature
        output_length = len(list(fnode.outputs()))
        ir_node = IROperation(
            signature = fsig,
            name = fnode.s('name'),
            input_length=input_length,
            output_length=output_length,
        )
        return ir_node

    @staticmethod
    def parse_aten_node(node, module) -> IROperation:
        """
        Parse script module node like:
            %13 : Tensor = aten::gt(%output1.1, %output2.1)
        """
        input_nodes = list()
        for input in node.inputs():
            input_node = input.node()
            input_nodes.append(input_node)


    @staticmethod
    def parse_node(node: torch._C.Node, module: torch.jit.RecursiveScriptModule):
        """
        Parse the node and get the inputs, module type, outputs

        Returns:
            Inputs: list[int]
            Modulelin
        """
        ntype = ScriptNodeKind.get_node_type(node)
        if ntype == ScriptNodeKind.PrimGetAttr:
            return ScriptModuleParser.parse_prim_attr_node(node, module)
        if ntype == ScriptNodeKind.PrimCallMethod:
            return ScriptModuleParser.parse_prim_method_node(node, module)
        if ntype == ScriptNodeKind.PrimCallFunction:
            return ScriptModuleParser.parse_prim_function_node(node, module)
        if ntype == ScriptNodeKind.AtenOp:
            return ScriptModuleParser.parse_aten_node(node, module)
        if ntype == ScriptNodeKind.PrimIf:
            return ScriptModuleParser.parse_prim_if_node(node, module)
        raise RuntimeError(f"Unkown node kind {node.kind()} from torchscript module")
    
    @staticmethod
    def parse_prim_method_node(node, module) -> List[IROperation]:
        """
        Parse script module node like:
            %output.1 : Tensor = prim::CallMethod[name="forward"](%2, %x.1)

        Find the module and 
        """
        # forward
        label = node.s('name')
        if label != 'forward':
            raise RuntimeError(f"{node} is calling function {label} that is not `forward`")
        call_module = getattr(module, label)
        ir_nodes = ScriptModuleParser.parse_module(call_module)
        # TODO: rename nodes
        return ir_nodes

    @staticmethod
    def parse_prim_attr_node(node, module) -> Dict:
        """
        Parse script module node like:
            %2 :__torch__.torch.nn.modules.linear.___torch_mangle_0.Linear = prim::GetAttr[name="linear1"](%self)
            %3 : Tensor = prim::GetAttr[name="weight"](%self)
        The __torch__.torch.nn.modules.* will be ignored
        """
        if node.inputsAt(0).debugeName() != 'self':
            raise RuntimeError(f"Fail to parse {node} due to missing %self")
        # user defined var name: linear1
        label = node.s('name')

        # output names: [2]
        output_vars : List[str] = list()
        for output in node.outputs():
            output_vars.append(output.debugName())

        # module type name
        module_mangle = node.outputsAt(0).type().str()
        if 'torch.nn.modules' in module_mangle:
            return None
        module_name_demangle = list()
        for mname in module_mangle.split('.'):
            if mname == '__torch__' or '_mangle' in mname:
                continue
            module_name_demangle.append(mname)
        module_name_demangle = '.'.joint(module_name_demangle)

        return dict(
            input_ids=list(), output_names=output_names,
            module=module_name_demangle,
            label=label
        )

    @staticmethod
    def parse_prim_constant_node(node):
        output = node.outputsAt(0)
        if output.type().str() == 'Function':
            func_name = repr(output.type())

    @staticmethod
    def parse_prim_if_node(node, module):
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
        if len(list(smodule.children())) == 0:
            for node in smodule.graph.nodes():
                print('    '*depth, node)
        else:
            for node in smodule.graph.nodes():
                ntype = ScriptModuleParser.get_node_type(node)
                print('    '*depth, node)
                if ntype == ScriptNodeKind.PrimCallMethod:
                    label = node.inputsAt(0).node().s('name')
                    submodule = getattr(smodule, label)
                    ScriptModuleParser.flatten(submodule, depth+1)


