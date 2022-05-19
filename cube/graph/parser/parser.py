import torch
import enum
import re
from typing import Any, List, Tuple, Optional

from cube.graph import IRFwOperation
from cube.graph.tensor import IRFullTensor
from cube.graph.parser.frame import Frame
from cube.graph.parser.mapping import Sign2Op, DType2IRDType


_refmodule = torch.nn.Module()

class ErasedDevice:
    pass


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
    PrimDevice = 11       # erased


class ScriptModuleParser:

    @staticmethod
    def parse_module(module,
                     input_shapes: Optional[ Tuple[List[int],] ] = None,
                     frame: Frame = None) \
        -> Tuple[List[IRFullTensor], List[IRFwOperation], List[IRFullTensor]]:
        """
        The overall entry to parse a torchscript graph module
        """
        frame = frame if frame is not None else Frame()
        frame.push()

        inputs = list(module.graph.inputs())[1:]
        if input_shapes is not None and len(input_shapes) != len(inputs):
            raise RuntimeError(f"Module {module.original_name} input shape mismatch (got {len(input_shapes)} != {len(inputs)})")

        # handle graph input -- Assuming all the inputs are tensors
        kDefaultType = DType2IRDType.map(torch.get_default_dtype())
        for idx, input in enumerate(inputs):
            if isinstance(input.type(), torch._C.TensorType):
                shape = None if input_shapes is None else input_shapes[idx]
                dtype = kDefaultType
                val = IRFullTensor(shape=shape, requires_grad=False, dtype=dtype, name=input.debugName())
            else:
                raise NotImplementedError("Graph inputs only accepts Tensor")
            frame.add_var(input.debugName(), val, graph_arg=idx)
        input_val = [frame.get_var(input.debugName()) for input in inputs]

        # handle nodes
        all_ir_nodes: List[IRFwOperation] = list()
        for node in module.graph.nodes():
            ir_nodes = ScriptModuleParser.parse_node(node, module, frame)
            for ir_node in ir_nodes:
                try:
                    ret = ir_node.infer_shape()
                    if not ret:
                        print(f'warning: {ir_node} cannot infer shape')
                except Exception:
                    raise RuntimeError(
                        f"====== Shape Infer Error ====\n\n\n"
                        f"IR Node: {ir_node}\n\n"
                        f"Node:\n{node}\n"
                        f"====== Shape Infer Error ====\n\n\n"
                    )
            all_ir_nodes += ir_nodes

        # handle outputs
        output_var_name = [output.debugName() for output in module.graph.outputs()]
        output_val = [frame.get_var(var_name) for var_name in output_var_name]
        
        # flatten output_val
        outputs = list()
        for val in output_val:
            if isinstance(val, list):
                outputs += val
            else:
                outputs.append(val)
        output_val = outputs

        frame.pop()
        return input_val, all_ir_nodes, output_val

    @staticmethod
    def parse_module_method(module, method: torch._C.ScriptMethod, frame: Frame):
        """
        Parse module method
        """
        frame.push()

        input_var_name = [input.debugName() for input in method.graph.inputs()]
        kDefaultType = DType2IRDType.map(torch.get_default_dtype())

        for index, var_name in enumerate(input_var_name[1:]): # omit self
            frame.add_var(var_name, IRFullTensor(name=var_name, requires_grad=False, dtype=kDefaultType), graph_arg=index)

        input_val = [frame.get_var(var_name) for var_name in input_var_name[1:]]

        all_ir_nodes: List[IRFwOperation] = list()
        for node in method.graph.nodes():
            ir_nodes = ScriptModuleParser.parse_node(node, module, frame)
            for ir_node in ir_nodes:
                try:
                    ret = ir_node.infer_shape()
                    if not ret:
                        print(f'warning: {ir_node} cannot infer shape')
                except Exception:
                    raise RuntimeError(
                        f"====== Shape Infer Error ====\n\n\n"
                        f"IR Node: {ir_node}\n\n"
                        f"Module:\n{module.code}\n\n"
                        f"Node:\n{node}\n"
                        f"====== Shape Infer Error ====\n\n\n"
                    )
            all_ir_nodes += ir_nodes

        # handle graph output
        output_var_name = [output.debugName() for output in method.graph.outputs()]
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
        if node.kind() == 'prim::ListConstruct':
            return ScriptNodeKind.PrimListConstruct
        if node.kind() == 'prim::TupleConstruct':
            return ScriptNodeKind.PrimListConstruct
        if node.kind() == 'prim::ListUnpack':
            return ScriptNodeKind.PrimListUnpack
        if node.kind() == 'prim::TupleUnpack':
            return ScriptNodeKind.PrimListUnpack
        if node.kind() == 'prim::PythonOp':
            return ScriptNodeKind.PrimPythonOp
        if node.kind() == 'prim::device':
            return ScriptNodeKind.PrimDevice
        raise RuntimeError(f"Unkown node kind {node.kind()} from torchscript module")

    @staticmethod
    def parse_node(node: torch._C.Node, module, frame: Frame) -> List[IRFwOperation]:
        # print("### parse_node {}".format(node))
        """
        Parse the node and return the IRFwOperation nodes
        """
        node_type = ScriptModuleParser.ntype(node)
        try:
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
                return ScriptModuleParser.parse_prim_list_unpack_node(node, module, frame)
            if node_type == ScriptNodeKind.PrimPythonOp:
                return ScriptModuleParser.parse_prim_python_op_node(node, module, frame)

            # TODO bother assigning all ignored prim functions new NodeKinds?
            if node_type == ScriptNodeKind.PrimDevice:
                return ScriptModuleParser.parse_value_erased_node(node, module, frame, [ErasedDevice()])

            raise NotImplementedError(f"Un-supported node type {node_type}")
        except Exception:
            raise RuntimeError(f"\n\nParsing error at node:\n\t{node}\n")

    @staticmethod
    def parse_prim_function_node(node, module, frame: Frame) -> List[IRFwOperation]:
        """
        parse node like:
            Tensor = prim::CallFunction(%5, %input.1, %3, %4)
            %5 : Function = prim::Constant[name="linear"]()
            %12 : (Tensor, Tensor) = prim::CallFunction(%5, %x1.1, %x2.1)
        """
        inputs = [input for input in node.inputs()]

        # get signature
        fnode = node.inputsAt(0).node()
        if not ScriptModuleParser.ntype(fnode) == ScriptNodeKind.PrimConstant:
            raise RuntimeError(f"Found unexpected function call node: {fnode}")
        fsig = frame.get_var(inputs[0].debugName())

        # get inputs
        input_vals = list()
        for index, input in enumerate(inputs[1:]):
            var_name = input.debugName()
            val = frame.get_var(var_name)
            input_vals.append(val)

        # map to IR operator
        ir_node = Sign2Op.map(fsig)(inputs=input_vals)
        
        # push output in the frame
        # help: >>> a = torch._C.TupleType([torch._C.TensorType.getInferred()])
        #     : >>> dir(a)
        #     : >>> a.elements()  # [TensorType, TensorType]
        cnt = 0
        for output in node.outputs():
            if isinstance(output.type(), torch._C.TupleType):
                tuplen = len(output.type().elements())
                ir_output = [ir_node.outputs(idx) for idx in range(cnt, cnt+tuplen)]
                cnt += tuplen
            else:
                ir_output = ir_node.outputs(cnt)
                cnt += 1
            frame.add_var(output.debugName(), ir_output)

        if cnt != len(ir_node.outputs()):
            raise RuntimeError(
                f"Parse fail: {fsig} has {cnt} outputs != pre-defined {len(ir_node.outputs())}"
            )

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
        input_val = [frame.get_var(input.debugName()) for input in inputs]

        # special handling on aten::size(tensor: tensor, dim: int)
        if fsig == 'torch.size':
            if len(inputs) == 2:
                tensor, dim = input_val
                output: int = tensor.shape[dim]
            else:
                tensor = input_val[0]
                output: List[int] = list(tensor.shape)
            frame.add_var(outputs[0].debugName(), output)
            return []

        # aten::__getitem__.t(t[](a) list, int idx) -> t(*)"
        # REMARK List-type only. '__getitem__' cannot serve as accessor to tensor element.
        elif fsig == 'torch.__getitem__':
            # NOTE there are other overloadings of '__getitem__' for 'str'(i.e. char list), 'Dict(t)' in TorchScript
            container, index = input_val
            frame.add_var(outputs[0].debugName(), container[index])
            return []

        # aten::tensor(elems: List^{n:Nat}[T], dtype:Optional[ScalarType], device:Device, requires_grad:bool) -> Tensor
        elif fsig == 'torch.tensor':
            # originally 'aten::tensor' 
            var_name = outputs[0].debugName()
            elems, dtype, erased_device, requires_grad = input_val

            # dtype may be None, in PyTorch it's to infer dtype from 'elems'.
            if dtype == None:
                dtype = DType2IRDType.map(torch.get_default_dtype())

            ir_tensor = IRFullTensor(shape=[len(elems)], name=var_name, requires_grad=requires_grad, dtype=dtype)
            frame.add_var(var_name, ir_tensor)
            return []

        # May be a symbolic object i.e. IRFwOperation,
        # or, occasionally this node can be statically evaluated, therefore a concrete value
        result = Sign2Op.map(fsig)(inputs=input_val)

        if isinstance(result, IRFwOperation):
            # to create IR node

            ir_node = result
            if len(ir_node.outputs()) != len(outputs):
                raise RuntimeError(
                    f"Parse fail: {fsig} has {len(outputs)} outputs != pre-defined {len(ir_node.outputs())}"
                )

            # handle outputs
            for index, output in enumerate(outputs):
                frame.add_var(output.debugName(), ir_node.outputs(index))

            return [ir_node]

        else:
            # concrete value.
            assert len(outputs) == 1, "Cases with multiple outputs are only List/Tuple-Unpack and handled specially"
            frame.add_var(outputs[0].debugName(), result)
            return []

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
        # handle inputs -- in stack with reverse order
        for input in inputs[1:][::-1]:
            var_name = input.debugName()
            val = frame.get_var(var_name)
            frame.push_param(var_name)

        # recursively parse the module
        if node.inputsAt(0).debugName() == 'self':
            call_module = module
        else:
            call_module = getattr(module, node.inputsAt(0).debugName())

        call_method = getattr(call_module, label)
        _, ir_nodes, outputs_val = ScriptModuleParser.parse_module_method(call_module, call_method, frame=frame)

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
            tensor = getattr(module, label)
            shape = list(tensor.shape)
            ir_tensor = IRFullTensor(
                name=label, shape=shape,
                requires_grad=tensor.requires_grad,
                dtype=DType2IRDType.map(tensor.dtype)
            )
            if isinstance(tensor, torch.nn.Parameter):
                ir_tensor.as_param()
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
        elif dtype == '__torch__.einops.einops.TransformRecipe':
            recipe = getattr(module, label)
            frame.add_var(var_name, recipe)
        else:
            # print("### parse_prim_attr_node unknown: {}".format(dtype))
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
    def parse_prim_list_unpack_node(node, module, frame: Frame) -> List[None]:
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
        tuple_inputs = frame.get_var(inputs[0].debugName())
        if len(tuple_inputs) != len(outputs):
            raise RuntimeError("Expected unpacked tuple number have same length of tupled input")
        for output, val in zip(outputs, tuple_inputs):
            frame.add_var(output.debugName(), val)
        return list()

    @staticmethod
    def parse_prim_python_op_node(node, module, frame):
        raise NotImplementedError("Cannot support torch.jit.ignore")
        print(dir(node))

    @staticmethod
    def parse_value_erased_node(node, module, frame, erased_vals: List[Any]):
        outputs = list(node.outputs())

        assert len(outputs) == len(erased_vals)
        for output, ev in zip(outputs, erased_vals):
            frame.add_var(output.debugName(), ev)
        return []



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


