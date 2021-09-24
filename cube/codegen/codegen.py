"""
Generate Pytorch code given the model DAG and the transformation config
"""
from typing import List, Any, Dict

from cube.graph import IRAction, IRTensor, IROperation
from cube.graph.ir_seq import IRSequence
from cube.codegen.syntax.symtable import SymbolTable
from cube.codegen.syntax.blocks import ClassBlock, FunctionBlock

import torch
import copy


class SScheduleCodeGen:
    """
    Generate spatial code for the model
    """

    def __init__(self, action: IRAction):
        if not isinstance(action, IRAction):
            raise TypeError("graph should be IRGraph")
        self.graph = action.graph
        # model full code
        self.init_code: List[str] = [
            '\n\n########## Generated Code ###########',
            'import torch', '', '']
        # module init code
        self.declare_region: List[str] = list()
        # module forward code
        self.forward_region: List[str] = list()
        # module member name
        self.symbols = SymbolTable()
        # ref module to check shared variables
        self._ref_module = torch.nn.Module()

    def gen(self, device: int, outfile=None, attach=False) -> str:
        """
        Generate model implementation code based on the given graph.
        """
        gencode = copy.copy(self.init_code)
        # register forward input
        fargs = [self.naming(input) for input in self.graph.inputs()]
        for name in fargs:
            self.symbols.create(name)

        # parse graph body
        for node in self.graph.nodes():
            self.emit_op_call(node)
            # emit input declaration
            for arg in node.inputs():
                self.emit_var_declare(arg)
            # record output tensor name
            for out in node.outputs():
                if isinstance(out, IRTensor) or isinstance(out, str):
                    self.symbols.create(self.naming(out))

        # generate full code
        with ClassBlock(class_name='GenModel', derived=['torch.nn.Module']) as cb:
            with FunctionBlock(func_name='__init__', args=['self']) as ib:
                ib.insert_body(self.declare_region)
            cb.insert_body('')
            cb.insert_body(ib.code)
            with FunctionBlock(func_name='forward', args=['self']+fargs) as fb:
                fb.insert_body(self.forward_region)
                # generate output
                out_names = self._forward_region_arg_names(self.graph.outputs())
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
        return code

    def emit_var_declare(self, var: Any):
        """
        Emit tensor declaration code
        """
        if isinstance(var, IRTensor):
            name = self.naming(var)
            # indicate this is a leaf tensor, should be parameter
            if self.symbols.create(name):
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

    def emit_op_call(self, node: IROperation):
        """
        Emit op forward code
        """
        op_code = node.signature
        out_names = self._forward_region_arg_names(node.outputs())
        out_names = ', '.join(out_names)
        arg_names = self._forward_region_arg_names(node.inputs())
        arg_region = '(' + ', '.join(arg_names) + ')'
        code = f'{out_names} = {op_code}{arg_region}'
        self.forward_region.append(code)

    def _forward_region_arg_names(self, args: List[Any]):
        """
        Generate arg name list for forward region.

        Will add prefix 'self.' for var defined in declare region
        """
        named_args : List[str] = list()
        input_name = [self.naming(input) for input in self.graph.inputs()]
        for arg in args:
            name = self.naming(arg)
            if isinstance(arg, IRTensor) and \
               arg.is_leaf() and (name not in input_name):
                named_args.append('self.' + self.naming(arg))
            else:
                named_args.append(self.naming(arg))
        return named_args

    def naming(self, tensor: Any) -> str:
        """
        Return the var name (unique for different variable)

        If the var is a leaf tensor, will add prefix `self.` to its name
        """
        if isinstance(tensor, IRTensor):
            tensor_name = 'tensor' if tensor.name is None else tensor.name
            if '.' in tensor_name:
                tensor_name = tensor_name.split('.')[0]
            name = '_'.join([tensor_name, str(tensor._id)])
        else:
            name = str(tensor)
        return name


class TScheduleCodeGen:

    def __init__(self, seq: IRSequence):
        if not isinstance(seq, IRSequence):
            raise TypeError("seq should be IRSequence")
        self.seq = seq
        # model full code
        self.init_code: List[str] = [
            '\n\n########## Generated Code ###########',
            'import torch', 'import cube', '']
        # module member name
        self.symbols = SymbolTable()

    def gen(self, device: int, outfile=None, attach=False) -> str:
        """
        Generate scheduling code based on the given actions
        """
        gencode = copy.copy(self.init_code)
        actions = list()
        for action in self.seq:
            if device in action.device:
                actions.append(action)

        # {send: xxx, recv: xxx} action1 {send:xxx, recv:xxx} action2 ....
        action_with_comms = [dict()]
        for action in actions:
            # send info
            send_tensors, send_devices = action.get_send_tensors()
            send_shapes = tuple([tensor.shape for tensor in send_tensors])
            send_tensors = [self.naming(tensor) for tensor in send_tensors]

            # recv info
            recv_tensors, recv_devices = action.get_recv_tensors()
            recv_shapes = tuple([tensor.shape for tensor in recv_tensors])
            recv_tensors = [self.naming(tensor) for tensor in recv_tensors]
            
            comm = action_with_comms[-1]

            # recv before the action
            if len(recv_tensors) != 0:
                comm.update({
                    'recv_tensors' : recv_tensors,
                    'recv_devices' : recv_devices,
                    'recv_shapes'  : recv_shapes
                })

            # action
            action_with_comms.append(action)

            # send after the action
            comm = dict()
            if len(send_tensors) != 0:
                comm.update({
                    'send_tensors' : send_tensors,
                    'send_devices' : send_devices,
                    'send_shapes'  : send_shapes
                })
            action_with_comms.append(comm)

        # generate code
        with FunctionBlock(func_name='_train_step', 
                           args=['model', 'dataloader']) as fb:
            for action_or_comm in action_with_comms:
                if isinstance(action_or_comm, dict):
                    code = self.emit_comm(action_or_comm)
                    fb.insert_body(code)
                else:
                    code = self.emit_action(action_or_comm)
                    fb.insert_body(code)
        gencode += fb.code
        gencode += ['']

        code = '\n'.join(gencode)
        # write to file
        if outfile:
            with open(outfile, 'a' if attach else 'w') as f:
                f.write(code)
        return code

    def emit_comm(self, comm: Dict) -> List[str]:
        """
        Emit send / recv code
        """
        ssign = 'cube.runtime.collectives.send({send_tensors}, {to_devices})'
        rsign = 'cube.runtime.collectives.recv({shapes}, {from_devices})'
        srsign = 'cube.runtime.collectives.send_and_recv({send_tensors}, {to_devices}, {recv_shapes}, {from_devices})'

        # generate for send
        if ('send_tensors') in comm and ('recv_tensors' not in comm):
            send_tensors = '(' + ', '.join(comm['send_tensors'] + ['']) + ')'
            code = ssign.format(
                send_tensors = send_tensors,
                to_devices   = comm['send_devices']
            )
            return code + f"  # send: {comm['send_shapes']}"
        # generate for recv
        elif ('send_tensors' not in comm) and ('recv_tensors' in comm):
            body = rsign.format(
                shapes       = comm['recv_shapes'],
                from_devices = comm['recv_devices']
            )
            return_val = ', '.join(comm['recv_tensors'])
            code = f'{return_val} = {body}'
            return code
        # generate for send + recv
        elif ('send_tensors' in comm) and ('recv_tensors' in comm):
            send_tensors = '(' + ', '.join(comm['send_tensors'] + ['']) + ')'
            body = srsign.format(
                send_tensors = send_tensors,
                to_devices   = comm['send_devices'],
                recv_shapes  = comm['recv_shapes'],
                from_devices = comm['recv_devices']
            )
            return_val = ', '.join(comm['recv_tensors'])
            code = f"{return_val} = {body}  # send: {comm['send_shapes']}"
            return code
        else:
            return []

    def emit_action(self, action: IRAction) -> List[str]:
        """
        Emit action code
        """
        fsign = 'cube.runtime.temporal.forward({model}, *{inputs})'
        bsign = 'cube.runtime.temporal.backward({input_tensors}, {output_tensors}, {output_grads})'
        
        if action.name == 'forward':
            inputs = [self.naming(tensor) for tensor in action.inputs()]
            inputs = '(' + ', '.join(inputs + ['']) + ')'
            body = fsign.format(
                model = 'model',
                inputs = inputs
            )
            outputs = [self.naming(output) for output in action.outputs()]
            return_val = ','.join(outputs)
            code = f'{return_val} = {body}'
            return code

        elif action.name == 'backward':
            # 1). input_tensors are forward inputs (happened before action inputs)
            #       => backward graph output tensor (share tensor in forward / backward graph)
            # 2). output_tensors are forward outputs (action.inputs())
            #       => backward graph input tensor (share tensor in forward / backward)
            # 3). output_grads are recved tesnors of this graph (graph.recv_tensors)
            #       => backward graph input tensor (graph.recv_tensors)
            forward_inputs = self.seq.get_forward_inputs(action)
            forward_inputs = [self.naming(tensor) for tensor in forward_inputs]
            forward_inputs = '(' + ', '.join(forward_inputs + ['']) + ')'
            forward_outputs = self.seq.get_forward_outputs(action)
            forward_outputs = [self.naming(tensor) for tensor in forward_outputs]
            forward_outputs = '(' + ', '.join(forward_outputs + ['']) + ')'
            num_recv_tensors = len(action.recv_tensors)
            if num_recv_tensors == 0:
                recv_grads = list()
            else:
                recv_grads = action.inputs()[-num_recv_tensors:]
            recv_grads = [self.naming(tensor) for tensor in recv_grads]
            recv_grads = '(' + ', '.join(recv_grads + ['']) + ')'

            body = bsign.format(
                input_tensors = forward_inputs,
                output_tensors = forward_outputs,
                output_grads = recv_grads
            )

            # returned value are graph.outputs
            return_val = [self.naming(tensor) for tensor in action.outputs()]
            if len(return_val) > 0:
                return_code = ', '.join(return_val) + ' = '
            else:
                return_code = ''
            code = f'{return_code}{body}'
            return code
        else:
            raise RuntimeError(f"Unsupported action tag: {action.tag}")

    def naming(self, tensor: Any) -> str:
        """
        Return the var name (unique for different variable)

        If the var is a leaf tensor, will add prefix `self.` to its name
        """
        if isinstance(tensor, IRTensor):
            if len(tensor.src()) == 0:
                name = '*next(dataloader)'
            else:
                tensor_name = 'tensor' if tensor.name is None else tensor.name
                if '.' in tensor_name:
                    tensor_name = tensor_name.split('.')[0]
                name = '_'.join([tensor_name, str(tensor._id)])
        else:
            name = str(tensor)
        return name
