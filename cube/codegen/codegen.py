"""
Generate Pytorch code given the model DAG and the transformation config
"""
from typing import Dict, List, Any, Tuple, Union
import torch
import copy
from cube.graph.parser.mapping import Sign2Op

from cube.ir.cten import IRCell, IRTensor
from cube.ir.dtype import IRDType

from cube.ir.tensor import IRSubTensor
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation
from cube.ir.adapter import IRWeightReducer, IRAdapter
from cube.ir.adapter.prim import CollectivePrim
from cube.graph.graph import IRGraph, IRSegment
from cube.graph.schedule import IRScheduleStrategy

from cube.execplan import ExectuionPlan

from cube.codegen.syntax.symtable import SymbolTable
from cube.codegen.syntax.blocks import ClassBlock, FunctionBlock
from cube.codegen.register import VarManager


class CodeGen:
    """
    Generate code for the model
    """
    def __init__(self, execplan: ExectuionPlan):
        if not isinstance(execplan, ExectuionPlan):
            raise TypeError("execplan should be ExecutionPlan")
        self.execplan = execplan

    def dtype_map(self, dtype: IRDType) -> str:
        if not isinstance(dtype, IRDType):
            raise TypeError("Expected IRDType")
        return 'torch.' + dtype.value

    def node_naming(self, node: IRCell) -> str:
        return f"{node.name}{node._id}"

    def tensor_naming(self, tensor: Any) -> str:
        """
        Return the var name (unique for different variable)
        """
        if isinstance(tensor, IRTensor):
            tensor_name = tensor.name
            if '.' in tensor_name:
                tensor_name = tensor_name.split('.')[0]
            name = '_'.join([tensor_name, str(tensor._id)])
        else:
            name = str(tensor)
        return name


class ModelCodeGen(CodeGen):
    """
    Generate model code
    """

    def __init__(self, execplan: ExectuionPlan):
        super().__init__(execplan)
        # model full code
        self.init_code: List[str] = [
            '\n\n########## Generated Model Code ###########',
            'import torch', 'import cube', '', '']
        # customized op code
        for _, op_impl in Sign2Op.kOpCodeDef.items():
            self.init_code.append(op_impl)
            self.init_code += ['', '']
        # module init code
        self.declare_region: List[str] = list()
        # module forward code
        self.forward_region_units: List[List[str]] = list()
        self.forward_region: List[str] = list()
        # module member name
        self.symbols = SymbolTable()
        # ref module to check shared variables
        self._ref_module = torch.nn.Module()

    def init_comm_groups(self):
        """
        Get all communication groups.

        Creating communication group requires all the devices
        enter the same call.
        """
        graph = self.execplan.graph
        sign = 'self.init_group(ranks={ranks})'
        # collect groups from weight reducer
        comm_groups: Dict[Tuple[int]] = list()
        for node in graph.nodes():
            if isinstance(node, IRWeightReducer):
                ranks = list(node.device)
                ranks.sort()
                ranks = tuple(ranks)
                if ranks not in comm_groups:
                    comm_groups.append(ranks)
        # collect groups from p2p fusion
        adapters = [n for n in graph.flatten() if isinstance(n, IRAdapter)]
        for adapter in adapters:
            for prim in adapter.prims:
                if isinstance(prim, CollectivePrim):
                    ranks = list(prim.kwargs['ranks'])
                    ranks.sort()
                    ranks = tuple(ranks)
                    if ranks not in comm_groups:
                        comm_groups.append(ranks)
        # create communication group
        self.declare_region.append('# communication groups')
        for ranks in comm_groups:
            code = sign.format(ranks=list(ranks))
            self.declare_region.append(code)
        self.declare_region.append(' ')

    def gen(self, device: int, outfile=None, attach=False) -> str:
        """
        Generate model implementation code based on the given graph.
        """
        gencode = copy.copy(self.init_code)
        node_args: List[List[str]] = list()
        gen_nodes: List[IRCell] = list()

        # initialize communication groups
        self.init_comm_groups()

        # parse graph body
        for node in self.execplan.seq(device):
            if isinstance(node, IRSegment):
                # skip backward ir graph
                if not node.forward:
                    continue
                self.emit_segment_call(node)
            elif isinstance(node, IRFwOperation):
                self.emit_op_call(node)
            elif isinstance(node, IRAdapter):
                self.emit_adapter_call(node)
            elif isinstance(node, IRWeightReducer):
                self.emit_reducer_init(node)
                self.emit_reducer_call(node)
            elif isinstance(node, IRBpOperation):
                continue
            elif isinstance(node, IRDataOperation):
                continue
            else:
                raise RuntimeError(f"Un-recognized IRCell type: {type(node)}")
            # emit node tensor declaration
            self.emit_node_declare(node)
            # emit node code
            self.forward_region_units.append(self.forward_region)
            self.forward_region = list()
            gen_nodes.append(node)
            args = list()
            for t in node.inputs():
                if isinstance(t, IRSubTensor):
                    if not t.is_param():
                        args.append(self.tensor_naming(t))
                else:
                    args.append(self.tensor_naming(t))
            node_args.append(args)

        # generate full code
        with ClassBlock(class_name='GenModel', derived=['cube.runtime.module.CubeModule']) as cb:
            with FunctionBlock(func_name='__init__', args=['self']) as ib:
                ib.insert_body(self.declare_region)
            cb.insert_body('')
            cb.insert_body(ib.code)
            for idx, node in enumerate(gen_nodes):
                name = self.node_naming(node)
                input_args = ['self'] + node_args[idx]
                forward_code = self.forward_region_units[idx]
                with FunctionBlock(func_name=name, args=input_args) as fb:
                    fb.insert_body(forward_code)
                    # generate output
                    outputs = [self.tensor_naming(t) for t in node.outputs()]
                    return_code = f"return {', '.join(outputs)}"
                    fb.insert_body(return_code)
                cb.insert_body('')
                cb.insert_body(fb.code)
        gencode += cb.code
        gencode += ['']

        code = '\n'.join(gencode)
        # write to file
        if outfile:
            with open(outfile, 'a' if attach else 'w') as f:
                f.write(code)
        
        # clear used buffer
        self.clear()
        return code

    def emit_node_declare(self, node: IRCell):
        """
        Emit tensor declaration code
        """
        sign = 'torch.nn.Parameter(torch.empty({shape}, dtype={dtype}))'
        for input in node.inputs():
            name = self.tensor_naming(input)
            if isinstance(input, IRTensor):
                if input.is_param() and not self.symbols.exist(name):
                    self.symbols.create(name)
                    code = f'{name} = {sign.format(shape=tuple(input.shape), dtype=self.dtype_map(input.dtype))}'
                    self.declare_region.append(code)
            if isinstance(input, str):
                if name.startswith('self.'):
                    if not hasattr(self._ref_module, name[5:]):
                        raise NotImplementedError("member attribute is not added")
        for output in node.outputs():
            self.symbols.create(self.tensor_naming(output))
        return

    def emit_segment_call(self, graph: IRGraph):
        """
        Emit IRSegment code
        """
        for node in graph.nodes():
            if isinstance(node, IRFwOperation):
                self.emit_op_call(node)
            elif isinstance(node, IRAdapter):
                self.emit_adapter_call(node)
            else:
                raise RuntimeError(f"unexpected type {type(node)} in forward graph:\n{graph.extra_repr()}")

    def emit_op_call(self, node: IRFwOperation):
        """
        Emit op forward code
        """
        op_code = node.signature
        inputs = [self.tensor_naming(t) for t in node.inputs()]
        kwargs = list()
        for key in node.kwargs:
            val = node.kwargs[key]
            if isinstance(val, str) and 'self.' not in val:
                val = '"' + val + '"'
            code = f'{key}={val}'
            kwargs.append(code)
        inputs += kwargs
        inputs = ', '.join(inputs)
        body = f'{op_code}({inputs})'
        if len(node.outputs()) == 0:
            code = body
        else:
            outputs = [self.tensor_naming(t) for t in node.outputs()]
            outputs = ', '.join(outputs)
            code = f'{outputs} = {body}'
        self.forward_region.append(code)

    def emit_adapter_call(self, node: IRAdapter):
        """
        Emit adapter call
        """
        assert len(node.device) == 1, f"Expected adapter to be dispatched:\n{node.extra_repr()}"
        for prim in node.prims:
            if len(prim.inputs()) == 1:
                itensors = self.tensor_naming(prim.inputs()[0])
            else:
                itensors = self.tuple_naming(prim.inputs())
            kwargs = list()
            for name, val in prim.kwargs.items():
                kwargs.append(f'{name}={val}')
            kwargs = ', '.join(kwargs)
            outputs = self.return_naming(prim.outputs())
            code = f'{outputs} = {prim.signature}({itensors}, {kwargs})'
            self.forward_region.append(code)

    def emit_reducer_init(self, node: IRWeightReducer):
        # reducer init interface
        reducer_init = '{reducer} = cube.runtime.adapter.Reducer(ranks={ranks})'
        reducer_add = 'self.add_reducer({reducer})'
        add_param = '{reducer}.add_param({weight})'
        # create reducer in declare region
        weights = node.inputs()
        reducer_name = f'self.wreducer{node._id}'
        self.declare_region.append('')
        init_code = reducer_init.format(reducer=reducer_name, ranks=node.device)
        self.declare_region.append(init_code)
        weights = [self.tensor_naming(t) for t in weights]
        for weight in weights:
            add_param_code = add_param.format(reducer=reducer_name, weight=weight)
            self.declare_region.append(add_param_code)
        add_code = reducer_add.format(reducer=reducer_name)
        self.declare_region.append(add_code)

    def emit_reducer_call(self, node: IRWeightReducer):
        reducer_name = f'self.wreducer{node._id}'
        call_code = f'{reducer_name}.allreduce()'
        self.forward_region.append(call_code)

    def return_naming(self, tensors: List[Any]) -> str:
        tensors = [self.tensor_naming(t) for t in tensors]
        if len(tensors) == 0:
            tensors = '_'
        else:
            tensors = ', '.join(tensors)
        return tensors

    def tuple_naming(self, tensors: List[Any]) -> str:
        tensors = [self.tensor_naming(t) for t in tensors]
        tensors = '(' + ', '.join(tensors + ['']) + ')'
        return tensors

    def tensor_naming(self, tensor: Any):
        """
        Generate tensor name.

        Will add prefix 'self.' for parameters
        """
        name = super().tensor_naming(tensor)
        if isinstance(tensor, IRSubTensor):
            if tensor.is_param():
                name = 'self.' + name
        return name

    def clear(self):
        """
        Clear buffer that used for generating code
        """
        # module init code
        self.declare_region: List[str] = list()
        # module forward code
        self.forward_region_units: List[List[str]] = list()
        self.forward_region: List[str] = list()
        # module member name
        self.symbols = SymbolTable()


class ScheduleCodeGen(CodeGen):

    def __init__(self, execplan: ExectuionPlan):
        super().__init__(execplan)
        # model full code
        self.init_code: List[str] = [
            '\n\n########## Generated Schedule Code ###########',
            'import torch', 'import cube', '']
        # module member name
        self.symbols = SymbolTable()
        self.vars = VarManager()

    def gen(self, device: int, outfile=None, attach=False) -> str:
        """
        Generate scheduling code based on the given sus
        """
        gencode = copy.copy(self.init_code)
        self.vars = VarManager()

        device_nodes = self.execplan.seq(device)

        def refcount(tensor, node) -> int:
            idx = device_nodes.index(node)
            refcnt = 0
            for ref_node in device_nodes[idx+1:]:
                if isinstance(ref_node, IRSegment):
                    if ref_node.forward:
                        if tensor in ref_node.inputs():
                            refcnt += 1
                    else:
                        finputs = ref_node.mirror.inputs()
                        foutputs = ref_node.mirror.outputs()
                        grad_in = [t.grad for t in foutputs]
                        if tensor in finputs + foutputs + grad_in:
                            refcnt += 1
                else:
                    if tensor in ref_node.inputs():
                        refcnt += 1
            return refcnt

        with FunctionBlock(func_name='_train_step', 
                           args=['model', 'dataloader']) as fb:
            fb.insert_body('_ = None')
            # body code
            if len(device_nodes) == 0:
                fb.insert_body('pass')
            elif self.execplan.graph.schedule_plan:
                code = self.emit_schedule_plan(self.execplan.graph.schedule_plan, device)
                fb.insert_body(code)
            else:
                for node in device_nodes:
                    name = self.node_naming(node)
                    code = self.emit_node(node, name=name)
                    fb.insert_body(code)
                    # free unused tensor
                    for tensor in node.inputs() + node.outputs():
                        if isinstance(tensor, IRSubTensor) and not tensor.is_param():
                            refcnt = refcount(tensor, node)
                            if refcnt == 0:
                                self.vars.free(tensor)
            # return code
            outputs = self.return_naming(self.execplan.graph.outputs())
            code = f'return {outputs}'
            fb.insert_body(code)
        gencode += fb.code
        gencode += ['']

        code = '\n'.join(gencode)
        # write to file
        if outfile:
            with open(outfile, 'a' if attach else 'w') as f:
                f.write(code)
        return code

    def emit_schedule_plan(self, schedplan: IRScheduleStrategy, devid: int):
        signature = schedplan.signature
        kwargs: Dict[str, Any] = schedplan.kwargs(devid)
        strkwargs = dict()
        for kwarg, val in kwargs.items():
            name = str(val) if not isinstance(val, IRCell) else 'model.'+self.node_naming(val)
            strkwargs[kwarg] = name
        code = ', '.join(f'{kwarg}={name}' for kwarg, name in strkwargs.items())
        code = f'{signature}({code})'
        return code

    def emit_node(self, node: IRCell, name: str) -> str:
        """
        Emit node / subgraph code
        """
        fsign = 'cube.runtime.executor.fexecute({model}, *{inputs})'
        bsign = 'cube.runtime.executor.backward({input_tensors}, {output_tensors}, {output_grads})'
        
        inputs = [self.tensor_naming(t) for t in node.inputs() if not t.is_param()]
        outputs = [self.tensor_naming(t) for t in node.outputs()]
        inputs = self.tuple_naming(inputs)
        outputs = self.return_naming(outputs)

        if isinstance(node, IRSegment):
            # emit forward
            if node.forward:
                body = fsign.format(model=f'model.{name}', inputs=inputs)
                code = f'{outputs} = {body}'
            # emit backward
            else:
                finputs = [t for t in node.mirror.inputs() if t.requires_grad]
                foutputs = node.mirror.outputs()
                inputs = [t.grad for t in foutputs]
                outputs = [t.grad for t in finputs]
                # remove weight gradient in outputs
                for input in finputs:
                    if input.is_param():
                        outputs.remove(input.grad)
                finputs = self.tuple_naming(finputs)
                foutputs = self.tuple_naming(foutputs)
                inputs = self.tuple_naming(inputs)
                outputs = self.return_naming(outputs)
                body = bsign.format(
                    input_tensors=finputs, output_tensors=foutputs, output_grads=inputs
                )
                code = f'{outputs} = {body}'

        elif isinstance(node, IRDataOperation):
            if len(node.inputs()) != 0:
                raise RuntimeError("Expect Dataloader node has no inputs")
            outputs = [self.tensor_naming(output) for output in node.outputs()]
            outputs = self.return_naming(outputs)
            code = f'{outputs} = next(dataloader)'

        elif isinstance(node, IRAdapter):
            body = fsign.format(model=f'model.{name}', inputs=inputs)
            code = f'{outputs} = {body}'

        elif isinstance(node, IRWeightReducer):
            body = fsign.format(model=f'model.{name}', inputs='()')
            code = f'{outputs} = {body}'

        else:
            raise RuntimeError(f"Unspported node type: {type(node)}")
        return code

    def tuple_naming(self, tensors: List[Any]) -> str:
        tensors = [self.tensor_naming(t) for t in tensors]
        tensors = '(' + ', '.join(tensors + ['']) + ')'
        return tensors

    def return_naming(self, tensors: List[Any]) -> str:
        tensors = [self.tensor_naming(t) for t in tensors]
        if len(tensors) == 0:
            tensors = '_'
        else:
            tensors = ', '.join(tensors)
        return tensors

    def tensor_naming(self, tensor: Union[IRSubTensor, Any]):
        """
        Generate tensor name.

        Will add prefix 'model.' for parameters
        """
        if isinstance(tensor, IRSubTensor) and tensor.is_param():
            return 'model.' + self.vars.allocate(tensor)
        else:
            return self.vars.allocate(tensor)
