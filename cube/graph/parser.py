import torch
import enum
import re
from collections import OrderedDict
from typing import List, Any, Tuple

from cube.graph.graph import IROperation, IRTensor


class _Frame:
    """
    Frame to save call stack and variable
    """
    def __init__(self):

        # var name -> value (IRTesnor, deterministic)
        self._vars: List[dict[str, Any]] = list()
        self._var_stack: List[str] = list()

    def push(self):
        """
        This should only be called when step in a module
        """
        self._vars.append(OrderedDict())

    def pop(self):
        """
        This should only be called step out a module
        """
        if len(self._vars) == 0:
            raise RuntimeError("Try to pop stack with 0 depth")
        self._vars.pop()

    def add_var(self, var_name: str, val: Any, graph_arg: int = 0):
        """
        Add variable to the current frame

        Args:
            var_name (str): variable name (unique)
            val: variable content
            graph_arg (int):
                indicate whether it is an argument of the graph.
                If is 0, is not a graph arg.
                If > 0, is a graph arg, will try to find 
                val from previous frame
        """
        if not isinstance(var_name, str):
            raise RuntimeError("Expected var_name is str")
        if var_name in self._vars[-1]:
            raise KeyError("Try to insert an already existed variable")
        if graph_arg == 0:
            self._vars[-1][var_name] = val
        elif graph_arg > 0:
            # root graph entry
            if self.depth() == 1:
                self._vars[-1][var_name] = val
            # fucnton call
            else:
                prev_frame = self._vars[-2]
                param_name = self._var_stack[0-graph_arg]
                val = prev_frame[param_name]
                self._vars[-1][var_name] = val
        else:
            raise ValueError("graph_arg (int) must be >= 0")

    def get_var(self, var_name: str) -> Any:
        """
        Get variable value according to var_name

        Special mapping between frames (function calls):

            input.x will be mapped to output.k at the about 1-hop frame

        Returns:
            val (Any)
        """
        # first check whether we have variable in this frame
        if var_name in self._vars[-1]:
            return self._vars[-1][var_name]
        raise KeyError(f"Cannot find var name {var_name}")

    def push_param(self, var_name):
        """
        push var name to the method stack

        Args:
            var_name (str): variable name
        """
        if var_name not in self._vars[-1]:
            raise KeyError(f"push {var_name} not declared")
        self._var_stack.append(var_name)

    def pop_param(self, times=1):
        """
        pop var name from the method stack
        """
        for _ in range(times):
            self._var_stack.pop()

    def depth(self):
        return len(self._vars)

    def __repr__(self):
        dscp = f'frame: depth: {self.depth()}\n  var table:'
        for var_name in self._vars[-1].keys():
            dscp += f'\n    {var_name} : {self._vars[-1][var_name]}'
        dscp += f'\n  var stack:'
        for var_name in self._var_stack:
            dscp += f'\n    {var_name}'
        return dscp


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
                     frame: _Frame = _Frame()) -> Tuple[List[IROperation], List[IRTensor]]:
        """
        The overall entry to parse a torchscript graph module
        """
        frame.push()

        # handle graph input -- Assuming all the inputs are tensors
        input_var_name = [input.debugName() for input in module.graph.inputs()]
        # [1:] is to omit self
        for var_name in input_var_name[1:][::-1]:
            frame.add_var(var_name, IRTensor(name=var_name), graph_arg=True)

        all_ir_nodes: List[IROperation] = list()
        for node in module.graph.nodes():
            # debug info
            print(f'on parsing:\n\t{node}')
            ir_nodes = ScriptModuleParser.parse_node(node, module, frame)
            print(f'> {frame}')
            print(f'> {ir_nodes}')
            _ = input('>>>')
            if len(ir_nodes) != 0:
                all_ir_nodes.append(ir_nodes)
        
        # handle graph output -- Assuming all the output are tensors
        output_var_name = [output.debugName() for output in module.graph.outputs()]
        output_val = [frame.get_var(var_name) for var_name in output_var_name]

        frame.pop()
        return all_ir_nodes, output_val

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
    def parse_node(node: torch._C.Node, module, frame: _Frame) -> List[IROperation]:
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
    def parse_prim_function_node(node, module, frame: _Frame) -> List[IROperation]:
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
    def parse_aten_node(node, module, frame: _Frame) -> List[IROperation]:
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
    def parse_prim_method_node(node, module, frame: _Frame) -> List[IROperation]:
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

        # cell node that will not appear in final graph
        signature = frame.get_var(node.inputsAt(0).debugName())
        ir_node = IROperation(
            name = signature + '.' + label,
            signature = signature,
            input_length = len(inputs) - 1,
            output_length = len(outputs)
        )

        # handle inputs -- in stack with reverse order
        for index, input in enumerate(inputs[1:][::-1]):
            var_name = input.debugName()
            val = frame.get_var(var_name)
            ir_node.set_input(-1-index, val)
            frame.push_param(var_name)

        print(f'> {frame}')

        # recursively parse the module
        module_label = node.inputsAt(0).node().s('name')
        call_module = getattr(module, module_label)
        ir_nodes, outputs_val = ScriptModuleParser.parse_module(call_module, frame)

        # pop out the frame
        frame.pop_param(times=len(inputs)-1)

        # handle outputs
        outputs = [output for output in node.outputs()]
        for index, (output, val) in enumerate(zip(outputs, outputs_val)):
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
            ir_tensor = IRTensor(name=label)
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
            frame.add_var(var_name, signature)
        else:
            val = node.outputsAt(0).toIValue()
            frame.add_var(var_name, val)
        return list()

    @staticmethod
    def parse_prim_if_node(node, module, frame: _Frame) -> List[IROperation]:
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


