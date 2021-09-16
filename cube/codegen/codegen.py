"""
Generate Pytorch code given the model DAG and the transformation config
"""

from typing import List, Any

from cube.graph import IRGraph, IRTensor, IROperation
from cube.codegen.syntax.symtable import SymbolTable
from cube.codegen.syntax.blocks import ClassBlock, FunctionBlock


class SScheduleCodeGen:
    """
    Generate spatial code for the model
    """

    def __init__(self, graph: IRGraph):
        if not isinstance(graph, IRGraph):
            raise TypeError("graph should be IRGraph")
        self.graph = graph
        # model full code
        self.code: List[str] = list()
        # module init code
        self.declare_region: List[str] = list()
        # module forward code
        self.forward_region: List[str] = list()
        # module member name
        self.symbols = SymbolTable()

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
        with ClassBlock(class_name='GenModel', derived='torch.nn.Module') as cb:
            with FunctionBlock(func_name='__init__', args=['self']) as ib:
                ib.insert_body(self.declare_region)
            cb.insert_body(ib.code)
            with FunctionBlock(func_name='forward', args=['self']+fargs) as fb:
                fb.insert_body(self.emit_op_call)
            cb.insert_body(fb.code)
        self.code = cb.code

        # write to file
        if outfile:
            with open(outfile, 'w'):
                for line in self.code:
                    outfile.write(line)

        return self.code

    def emit_var_declare(self, var: Any):
        """
        Emit tensor declaration code
        """
        if isinstance(var, IRTensor):
            name = self.naming(var)
            # indicate this is a leaf tensor, should be parameter
            if self.symbols.create(name):
                code = f'self.{name} = torch.nn.Parameter(torch.empty({tuple(IRTensor.shape)}))'
                self.declare_region.append(code)
        elif isinstance(var, str):
            name = self.naming(var)
            if self.symbols.create(name):
                #TODO: add type info
                code = f'self.{name} = None'
                self.declare_region.append(code)
        return

    def emit_op_call(self, node: IROperation):
        """
        Emit op forward code
        """
        op_code = node.signature
        out_region = ', '.join([self.naming(out) for out in node.outputs()])
        arg_region = '(' + ', '.join([self.naming(arg) for arg in node.inputs()]) + ')'
        code = f'{out_region} = {op_code}{arg_region}'
        self.forward_region.append(code)

    def naming(self, tensor: Any) -> str:
        """
        Return the var name (unique for different variable)
        """
        if isinstance(tensor, IRTensor):
            tensor_name = 'tensor' if tensor.name is None else tensor.name
            name = '_'.join([tensor_name, tensor._id])
        else:
            name = str(tensor)
        return name
