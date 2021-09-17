"""
Generate Pytorch code given the model DAG and the transformation config
"""
from typing import List, Any

from cube.graph import IRGraph, IRTensor, IROperation
from cube.codegen.syntax.symtable import SymbolTable
from cube.codegen.syntax.blocks import ClassBlock, FunctionBlock

import torch


class SScheduleCodeGen:
    """
    Generate spatial code for the model
    """

    def __init__(self, graph: IRGraph):
        if not isinstance(graph, IRGraph):
            raise TypeError("graph should be IRGraph")
        self.graph = graph
        # model full code
        self.code: List[str] = ['import torch', '', '']
        # module init code
        self.declare_region: List[str] = list()
        # module forward code
        self.forward_region: List[str] = list()
        # module member name
        self.symbols = SymbolTable()
        # ref module to check shared variables
        self._ref_module = torch.nn.Module()

    def gen(self, outfile=None) -> List[str]:
        """
        Generate model implementation code based on the given graph.
        """
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
        self.code += cb.code
        self.code += ['']

        code = '\n'.join(self.code)
        # write to file
        if outfile:
            with open(outfile, 'w') as f:
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
