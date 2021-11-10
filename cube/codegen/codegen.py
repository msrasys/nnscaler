"""
Generate Pytorch code given the model DAG and the transformation config
"""
from typing import List, Any
import torch
import copy

from cube.ir.cten import IRTensor
from cube.execplan import ExectuionPlan

from cube.schedule.su import ScheduleUnit, SUType
from cube.schedule.adapter.comm import IRCommType, IRCommunication
from cube.schedule.adapter.select import IRTensorReshape, IRReshapeType
from cube.codegen.syntax.symtable import SymbolTable
from cube.codegen.syntax.blocks import ClassBlock, FunctionBlock


class ModelCodeGen:
    """
    Generate spatial code for the model
    """

    def __init__(self, execplan: ExectuionPlan):
        if not isinstance(execplan, ExectuionPlan):
            raise TypeError("execplan should be ExecutionPlan")
        self.execplan = execplan
        # model full code
        self.init_code: List[str] = [
            '\n\n########## Generated Model Code ###########',
            'import torch', 'import cube', '', '']
        # module init code
        self.declare_region: List[str] = list()
        # module forward code
        self.all_su_forward_region: List[List[str]] = list()
        self.forward_region: List[str] = list()
        # module member name
        self.symbols = SymbolTable()
        # ref module to check shared variables
        self._ref_module = torch.nn.Module()

    def gen(self, device: int, outfile=None, attach=False) -> str:
        """
        Generate model implementation code based on the given graph.
        """
        device_sus = self.execplan.sequence(device)
        device_sus = [su for su in device_sus \
                      if su.stype != SUType.Backward \
                      and su.stype != SUType.Dataloader]

        gencode = copy.copy(self.init_code)

        # register forward input
        su_args: List[List[str]] = list()
        for su in device_sus:
            fargs = list()
            for input in su.inputs():
                if isinstance(input, IRTensor) and input.is_param():
                    continue
                fargs.append(self.naming(input))
            for name in fargs:
                self.symbols.create(name)
            su_args.append(fargs)

        # parse graph body
        for su in device_sus:
            for node in su.nodes():
                if isinstance(node, IRTensorReshape):
                    self.emit_reshape_call(node)
                if isinstance(node, IRCommunication):
                    self.emit_comm_call(node)
                else:
                    self.emit_op_call(node)
                # emit input declaration
                for arg in node.inputs():
                    self.emit_var_declare(arg)
                # record output tensor name
                for out in node.outputs():
                    if isinstance(out, IRTensor) or isinstance(out, str):
                        self.symbols.create(self.naming(out))
            self.all_su_forward_region.append(self.forward_region)
            self.forward_region = list()

        # generate full code
        with ClassBlock(class_name='GenModel', derived=['torch.nn.Module']) as cb:
            with FunctionBlock(func_name='__init__', args=['self']) as ib:
                ib.insert_body(self.declare_region)
            cb.insert_body('')
            cb.insert_body(ib.code)
            for idx, su in enumerate(device_sus):
                name = f'su{su._id}'
                input_args = ['self'] + su_args[idx]
                forward_code = self.all_su_forward_region[idx]
                with FunctionBlock(func_name=name, args=input_args) as fb:
                    fb.insert_body(forward_code)
                    # generate output
                    out_names = self._forward_region_arg_names(su.outputs())
                    return_code = f"return {', '.join(out_names)}"
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

    def emit_var_declare(self, var: Any):
        """
        Emit tensor declaration code
        """
        if isinstance(var, IRTensor):
            name = self.naming(var)
            # emit parameter code
            if var.is_param() and not self.symbols.exist(name):
                self.symbols.create(name)
                code = f'self.{name} = torch.nn.Parameter(torch.empty({tuple(var.shape)}))'
                self.declare_region.append(code)
        elif isinstance(var, str):
            name = self.naming(var)
            if name.startswith('self.'):
                if not hasattr(self._ref_module, var[5:]):
                    if self.symbols.create(name):
                        #TODO: add default value
                        code = f'{name} = None'
                        self.declare_region.append(code)
        return

    def emit_op_call(self, node):
        """
        Emit op forward code
        """
        op_code = node.signature
        arg_names = self._forward_region_arg_names(node.inputs())
        arg_region = '(' + ', '.join(arg_names) + ')'
        if len(node.outputs()) == 0:
            code = f'{op_code}{arg_region}'
        else:
            out_names = self._forward_region_arg_names(node.outputs())
            out_names = ', '.join(out_names)
            code = f'{out_names} = {op_code}{arg_region}'
        self.forward_region.append(code)

    def emit_comm_call(self, node):
        """
        Emit communication code
        """
        comm_code = node.signature
        send_tensors = self._forward_region_arg_names(node.inputs())
        send_tensors = '(' + ', '.join(send_tensors + ['']) + ')'
        send_ranks = node.send_ranks
        recv_tensors = self._forward_region_arg_names(node.outputs())
        recv_tensors = ', '.join(recv_tensors)
        recv_shapes = [tensor.shape for tensor in node.outputs()]
        recv_ranks = node.recv_ranks
        if node.comm_type == IRCommType.Send:
            code = f'{comm_code}({send_tensors}, {send_ranks})'
        elif node.comm_type == IRCommType.Recv:
            code = f'{recv_tensors} = {comm_code}({recv_shapes}, {recv_ranks})'
        elif node.comm_type == IRCommType.SendRecv:
            code = f'{recv_tensors} = {comm_code}({send_tensors}, {send_ranks}, {recv_shapes}, {recv_ranks})'
        else:
            raise TypeError(f"Unsupported IRCommmNode: {node.comm_type}")
        self.forward_region.append(code)

    def emit_reshape_call(self, node):
        """
        Emit in-device tensor select / merge call.
        """
        src_tensors = self._forward_region_arg_names(node.inputs())
        dst_tensors = self._forward_region_arg_names(node.outputs())
        # emit select
        if node.ttype == IRReshapeType.Select:
            src_tensor = src_tensors[0]
            #TODO: relative indices
            indices = node.select_indices
            indices = [slicer.get() for slicer in indices]
            dst_tensors = ', '.join(dst_tensors)
            code = f'{dst_tensors} = {node.signature}({src_tensor}, {indices})'
            self.forward_region.append(code)
        elif node.ttype == IRReshapeType.Merge:
            axis = node.merge_axis
            src_tensor = '(' + ', '.join(src_tensors + ['']) + ')'
            dst_tensor = dst_tensors[0]
            code = f'{dst_tensor} = {node.signature}({src_tensor}, {axis})'
            self.forward_region.append(code)
        else:
            raise TypeError(f"Unknown Reshape Type: {node.ttype}")

    def _forward_region_arg_names(self, tensors: List[Any]):
        """
        Generate arg name list for forward region.

        Will add prefix 'self.' for var defined in declare region
        """
        named_args : List[str] = list()
        for tensor in tensors:
            name = self.naming(tensor)
            if isinstance(tensor, IRTensor) and tensor.is_param():
                named_args.append('self.' + name)
            else:
                named_args.append(self.naming(name))
        return named_args

    def naming(self, tensor: Any) -> str:
        """
        Return the var name (unique for different variable)
        """
        if isinstance(tensor, IRTensor):
            tensor_name = 'tensor' if tensor.name is None else tensor.name
            if '.' in tensor_name:
                tensor_name = tensor_name.split('.')[0]
            name = '_'.join([tensor_name, str(tensor._id)])
        else:
            name = str(tensor)
        return name

    def clear(self):
        """
        Clear buffer that used for generating code
        """
        # module init code
        self.declare_region: List[str] = list()
        # module forward code
        self.all_su_forward_region: List[List[str]] = list()
        self.forward_region: List[str] = list()
        # module member name
        self.symbols = SymbolTable()


class ScheduleCodeGen:

    def __init__(self, execplan: ExectuionPlan):
        if not isinstance(execplan, ExectuionPlan):
            raise TypeError("execplan should be ExecutionPlan")
        self.execplan = execplan
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
        device_sus = self.execplan.sequence(device)

        # generate code
        with FunctionBlock(func_name='_train_step', 
                           args=['model', 'dataloader']) as fb:
            for su in device_sus:
                name = f'su{su._id}'
                code = self.emit_su(su, name=name)
                fb.insert_body(code)
        gencode += fb.code
        gencode += ['']

        code = '\n'.join(gencode)
        # write to file
        if outfile:
            with open(outfile, 'a' if attach else 'w') as f:
                f.write(code)
        return code

    def emit_su(self, su: ScheduleUnit, name: str) -> List[str]:
        """
        Emit su code
        """
        fsign = 'cube.runtime.executor.fexecute({model}, *{inputs})'
        bsign = 'cube.runtime.executor.backward({input_tensors}, {output_tensors}, {output_grads})'
        
        if su.stype == SUType.Dataloader:
            if len(su.inputs()) != 0:
                raise RuntimeError("Dataloader su has no inputs")
            outputs = [self.naming(output, su) for output in su.outputs()]
            return_val = ','.join(outputs)
            code = f'{return_val} = next(dataloader)'
            return code

        elif su.stype == SUType.Forward or su.stype == SUType.Comm:
            inputs = list()
            for tensor in su.inputs():
                if isinstance(tensor, IRTensor):
                    if tensor.is_param():
                        continue
                inputs.append(self.naming(tensor, su))
            inputs = '(' + ', '.join(inputs + ['']) + ')'
            body = fsign.format(
                model = f'model.{name}',
                inputs = inputs
            )
            outputs = [self.naming(output, su) for output in su.outputs()]
            return_val = ','.join(outputs)
            if len(su.outputs()) == 0:
                code = body
            else:
                code = f'{return_val} = {body}'
            return code

        elif su.stype == SUType.Backward:
            # 1). input_tensors are forward inputs (happened before su inputs)
            #       => backward graph output tensor (share tensor in forward / backward graph)
            # 2). output_tensors are forward outputs (su.inputs())
            #       => backward graph input tensor (share tensor in forward / backward)
            # 3). output_grads are recved tesnors of this graph (graph.recv_tensors)
            #       => backward graph input tensor (graph.recv_tensors)
            fsu = su.mirror
            finputs = list()
            for tensor in fsu.inputs():
                if isinstance(tensor, IRTensor):
                    if tensor.is_param():
                        continue
                finputs.append(self.naming(tensor, fsu))
            fargs = '(' + ', '.join(finputs + ['']) + ')'

            fouts = list()
            for tensor in fsu.outputs():
                fouts.append(self.naming(tensor, fsu))
            fouts = '(' + ', '.join(fouts + ['']) + ')'

            fout_grads = list()
            for fout in fsu.outputs():
                fout_grads.append(self.naming(fout.grad, fsu))
            fout_grads = '(' + ', '.join(fout_grads + ['']) + ')'

            body = bsign.format(
                input_tensors = fargs,
                output_tensors = fouts,
                output_grads = fout_grads
            )

            # returned value are graph.outputs
            return_val = [self.naming(tensor, su) for tensor in su.outputs()]
            # TODO: fix this by using grad attributed
            return_val = return_val[:len(finputs)]
            if len(return_val) > 0:
                return_code = ', '.join(return_val) + ' = '
            else:
                return_code = ''
            code = f'{return_code}{body}'
            return code
        else:
            raise RuntimeError(f"Unsupported su tag: {su.tag}")

    def naming(self, tensor: Any, su) -> str:
        """
        Return the var name (unique for different variable)

        If the var is a leaf tensor, will add prefix `self.` to its name
        """
        if isinstance(tensor, IRTensor):
            # note in su there is no parameters
            # if len(tensor.src(su.nodes())) == 0:
            #     name = '*next(dataloader)'
            # else:
            tensor_name = 'tensor' if tensor.name is None else tensor.name
            if '.' in tensor_name:
                tensor_name = tensor_name.split('.')[0]
            name = '_'.join([tensor_name, str(tensor._id)])
        else:
            name = str(tensor)
        return name
