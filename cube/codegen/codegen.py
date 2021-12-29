"""
Generate Pytorch code given the model DAG and the transformation config
"""
from typing import List, Any
import torch
import copy

from cube.ir.cten import IRCell, IRTensor
from cube.ir.dtype import IRDType
from cube.graph.tensor import IRSubTensor
from cube.graph.operator.operator import IRBpOperation, IRDataOperation, IRFwOperation
from cube.graph.operator.operator import IROptimOperation
from cube.graph.adapter.adapter import IRAdapter, SelectPrim, MovePrim, MergePrim
from cube.execplan import ExectuionPlan
# from cube.schedule.adapter.collectives import IRCollectives

from cube.graph.graph import IRGraph
from cube.codegen.syntax.symtable import SymbolTable
from cube.codegen.syntax.blocks import ClassBlock, FunctionBlock


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
        # module init code
        self.declare_region: List[str] = list()
        # module forward code
        self.forward_region_units: List[List[str]] = list()
        self.forward_region: List[str] = list()
        # module member name
        self.symbols = SymbolTable()
        # ref module to check shared variables
        self._ref_module = torch.nn.Module()
        # groups
        self._all_comm_groups = list()
        # self.get_all_groups()

    def get_all_groups(self):
        """
        Get all communication groups.

        Creating communication group requires all the devices
        enter the same call.
        """
        raise NotImplementedError
        for devid in self.execplan.devices():
            for su in self.execplan.sequence(devid):
                if su.stype == SUType.Coll:
                    ranks = list(su.nodes(0).ranks)
                    ranks.sort()
                    ranks = tuple(ranks)
                    if ranks not in self._all_comm_groups:
                        self._all_comm_groups.append(ranks)

    def gen(self, device: int, outfile=None, attach=False) -> str:
        """
        Generate model implementation code based on the given graph.
        """
        gencode = copy.copy(self.init_code)
        node_args: List[List[str]] = list()
        gen_nodes: List[IRCell] = list()

        # TODO init group
        # self.emit_comm_group_creation()

        # parse graph body
        for node in self.execplan.sequence(device):
            if isinstance(node, IRGraph):
                # skip backward ir graph
                if all([isinstance(n, IRBpOperation) for n in node.nodes()]):
                    continue
                self.emit_graph_call(node)
            elif isinstance(node, IRFwOperation):
                self.emit_op_call(node)
            elif isinstance(node, IRAdapter):
                node = node.dispatch(rank=device)
                self.emit_adapter_call(node)
            # elif isinstance(node, IRCollectives):
            #     self.emit_collective_call(node)
            elif isinstance(node, IROptimOperation):
                self.emit_optim_init(node)
                self.emit_optim_call(node)
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


    def emit_graph_call(self, graph: IRGraph):
        for node in graph.nodes():
            if isinstance(node, IRBpOperation):
                raise RuntimeError("IRBpOperation is not expected in GenModel")
            self.emit_op_call(node)


    # def emit_comm_group_creation(self):
    #     """
    #     Emit communication group creation code
    #     """
    #     sign = 'self.init_group(ranks={ranks})'
    #     for ranks in self._all_comm_groups:
    #         ranks = list(ranks)
    #         code = sign.format(ranks=ranks)
    #         self.declare_region.append(code)

    def emit_op_call(self, node: IRFwOperation):
        """
        Emit op forward code
        """
        op_code = node.signature
        inputs = [self.tensor_naming(t) for t in node.inputs()]
        kwargs = list()
        for key in node.kwargs:
            code = f'{key}={node.kwargs[key]}'
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
        if len(node.device) != 1:
            raise RuntimeError("Expected IRAdapter to be dispatched")
        rank = node.device[0]
        for prim in node.prims():
            # emit select
            if isinstance(prim, SelectPrim):
                sign = 'cube.runtime.adapter.select({tensor}, {indmap}, {valmap})'
                input = self.tensor_naming(prim.tensor)
                output = self.tensor_naming(prim.output)
                valmap = (prim.valmap.idx, prim.valmap.chunk_num)
                code = f'{output} = {sign.format(tensor=input, indmap=prim.indmap, valmap=valmap)}'
                self.forward_region.append(code)
            # emit move
            elif isinstance(prim, MovePrim):
                send_sign = 'cube.runtime.adapter.send({tensor}, {send_rank})'
                recv_sign = 'cube.runtime.adapter.recv({shape}, {from_rank}, {dtype})'
                tensor = self.tensor_naming(prim.tensor)
                # send
                if rank == prim.from_rank:
                    code = f'{send_sign.format(tensor=tensor, send_rank=prim.to_rank)}'
                    self.forward_region.append(code)
                # recv
                elif rank == prim.to_rank:
                    output = self.tensor_naming(prim.tensor)
                    dtype = self.dtype_map(prim.dtype)
                    code = f'{tensor} = {recv_sign.format(shape=prim.shape, from_rank=prim.from_rank, dtype=dtype)}'
                    self.forward_region.append(code)
            # emit merge
            elif isinstance(prim, MergePrim):
                sign = 'cube.runtime.adapter.merge({tensors}, {concat}, {add})'
                inputs = [self.tensor_naming(t) for t in prim.tensors]
                inputs = '(' + ','.join(inputs + ['']) + ')'
                output = self.tensor_naming(prim.output)
                code = f'{output} = {sign.format(tensors=inputs, concat=prim.concat, add=prim.add)}'
                self.forward_region.append(code)
            else:
                raise TypeError(f"Unkown primitive types {type(prim)} of Adapter")
        # requires grad generation
        sign = '{output} = {output}.contiguous().requires_grad_()'
        for output in node.outputs():
            if isinstance(output, IRSubTensor):
                code = sign.format(output=self.tensor_naming(output))
                self.forward_region.append(code)

    # def emit_comm_call(self, node):
    #     """
    #     Emit communication code
    #     """
    #     comm_code = node.signature
    #     send_tensors = self._forward_region_arg_names(node.inputs())
    #     send_tensors = '(' + ', '.join(send_tensors + ['']) + ')'
    #     send_ranks = node.send_ranks
    #     recv_tensors = self._forward_region_arg_names(node.outputs())
    #     recv_tensors = ', '.join(recv_tensors)
    #     recv_shapes = [tensor.shape for tensor in node.outputs()]
    #     recv_ranks = node.recv_ranks
    #     if node.comm_type == IRCommType.Send:
    #         code = f'{comm_code}({send_tensors}, {send_ranks})'
    #     elif node.comm_type == IRCommType.Recv:
    #         code = f'{recv_tensors} = {comm_code}({recv_shapes}, {recv_ranks})'
    #     elif node.comm_type == IRCommType.SendRecv:
    #         code = f'{recv_tensors} = {comm_code}({send_tensors}, {send_ranks}, {recv_shapes}, {recv_ranks})'
    #     else:
    #         raise TypeError(f"Unsupported IRCommmNode: {node.comm_type}")
    #     self.forward_region.append(code)

    # def emit_collective_call(self, node):
    #     ranks = node.ranks
    #     inputs = self._forward_region_arg_names(node.inputs())
    #     shape = None
    #     if len(inputs) == 0:
    #         assert len(node.outputs()) == 1
    #         shape = node.outputs(0).shape
    #     inputs = '(' + ', '.join(inputs + ['']) + ')'
    #     outputs = self._forward_region_arg_names(node.outputs())
    #     outputs = ', '.join(outputs)
    #     if shape:
    #         code = f'{node.signature}({inputs}, {ranks}, {shape})'
    #     else:
    #         code = f'{node.signature}({inputs}, {ranks})'
    #     if outputs:
    #         code = f'{outputs} = {code}'
    #     self.forward_region.append(code)

    # def emit_transform_call(self, node):
    #     """
    #     Emit in-device tensor select / merge call.
    #     """
    #     for prim in node.trace():
    #         if isinstance(prim, SelectPrim):
    #             signature = 'cube.runtime.transform.select({tensor}, {indmap}, {valmap})'
    #             input = self.tensor_naming(prim.tensor)
    #             indmap = repr(prim.indmap)
    #             valmap = repr(tuple([prim.valmap.idx, prim.valmap.chunk_num]))
    #             output = self.tensor_naming(prim.output)
    #             code = f'{output} = {signature.format(tensor=input, indmap=indmap, valmap=valmap)}'
    #             self.forward_region.append(code)
    #         elif isinstance(prim, MergePrim):
    #             signature = 'cube.runtime.transform.merge({tensors}, {concat}, {add})'
    #             inputs = self._forward_region_arg_names(prim.tensors)
    #             inputs = '(' + ', '.join(inputs) + ')'
    #             output = self.tensor_naming(prim.output)
    #             code = f'{output} = {signature.format(tensors=inputs, concat=prim.concat, add=prim.add)}'
    #             self.forward_region.append(code)
    #         else:
    #             raise RuntimeError(f"Not supported prim: {type(prim)}")
    #     for output in node.outputs():
    #         # contiguous and requires grad
    #         output_name = self.tensor_naming(output)
    #         code = f'{output_name} = {output_name}.contiguous()'
    #         self.forward_region.append(code)
    #         if not output.is_grad():
    #             code = f'{output_name} = {output_name}.requires_grad_()'
    #             self.forward_region.append(code)

    def emit_optim_init(self, node: IROptimOperation):
        # reducer init interface
        reducer_init = '{reducer} = cube.runtime.reducer.Reducer(ranks={ranks})'
        reducer_add = 'self.add_reducer({reducer})'
        add_param = '{reducer}.add_param({grad})'
        # create reducer in declare region
        ranks = list(node.ranks)
        grads = node.inputs()
        reducer_name = f'self.reducer{node._id}'
        self.declare_region.append('')
        init_code = reducer_init.format(reducer=reducer_name, ranks=ranks)
        self.declare_region.append(init_code)
        grads = [self.tensor_naming(t) for t in grads]
        for grad in grads:
            add_param_code = add_param.format(reducer=reducer_name, grad=grad)
            self.declare_region.append(add_param_code)
        add_code = reducer_add.format(reducer=reducer_name)
        self.declare_region.append(add_code)

    def emit_optim_call(self, node: IROptimOperation):
        reducer_name = f'self.reducer{node._id}'
        call_code = f'{reducer_name}.allreduce()'
        self.forward_region.append(call_code)

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

    def gen(self, device: int, outfile=None, attach=False) -> str:
        """
        Generate scheduling code based on the given sus
        """
        gencode = copy.copy(self.init_code)

        device_nodes = self.execplan.sequence(device)
        for idx, node in enumerate(device_nodes):
            if isinstance(node, IRAdapter):
                node = node.dispatch(rank=device)
                device_nodes[idx] = node

        # generate code
        with FunctionBlock(func_name='_train_step', 
                           args=['model', 'dataloader']) as fb:
            if len(device_nodes) == 0:
                fb.insert_body('pass')
            for node in device_nodes:
                name = self.node_naming(node)
                code = self.emit_node(node, name=name)
                fb.insert_body(code)
        gencode += fb.code
        gencode += ['']

        code = '\n'.join(gencode)
        # write to file
        if outfile:
            with open(outfile, 'a' if attach else 'w') as f:
                f.write(code)
        return code

    def emit_node(self, node: IRCell, name: str) -> List[str]:
        """
        Emit node / subgraph code
        """
        fsign = 'cube.runtime.executor.fexecute({model}, *{inputs})'
        bsign = 'cube.runtime.executor.backward({input_tensors}, {output_tensors}, {output_grads})'
        
        inputs = [self.tensor_naming(t) for t in node.inputs() if not t.is_param()]
        outputs = [self.tensor_naming(t) for t in node.outputs()]
        inputs = self.tuple_naming(inputs)
        outputs = self.return_naming(outputs)

        if isinstance(node, IRGraph):
            is_backward = all([isinstance(n, IRBpOperation) for n in node.nodes()])
            # emit forward
            if not is_backward:
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

    def tensor_naming(self, tensor: Any):
        """
        Generate tensor name.

        Will add prefix 'model.' for parameters
        """
        name = super().tensor_naming(tensor)
        if isinstance(tensor, IRSubTensor):
            if tensor.is_param():
                name = 'model.' + name
        return name
