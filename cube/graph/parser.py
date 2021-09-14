import torch
import enum

class ScriptNodeKind(enum.Enum):
    PrimGetAttr = 1
    PrimCallMethod = 2


class ScriptModuleParser:

    @staticmethod
    def get_node_type(node: torch._C.Node):
        if node.kind() == 'prim::GetAttr':
            return ScriptNodeKind.PrimGetAttr
        if node.kind() == 'prim::CallMethod':
            return ScriptNodeKind.PrimCallMethod

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
            return ScriptModuleParser.parse_prim_call_node(node, module)

    @staticmethod
    def parse_prim_attr_node(node, module):
        """
        Parse script module node like:
            %2 :__torch__.torch.nn.modules.linear.___torch_mangle_0.Linear = prim::GetAttr[name="linear1"](%self)
        """
        name = node.s('name')
        module_mangle = node.outputsAt(0).type().str()
        module_name_demangle = list()
        for mname in module_mangle.split('.'):
            if mname == '__torch__' or '_mangle' in mname:
                continue
            module_name_demangle.append(mname)
        module_name_demangle = '.'.joint(module_name_demangle)
        # TODO


    def parse_prim_call_node(node, module):
        """
        Parse script module node like:
            %output.1 : Tensor = prim::CallMethod[name="forward"](%2, %x.1) # /tmp/ipykernel_27188/97711738.py:11:17
        """
        #TODO