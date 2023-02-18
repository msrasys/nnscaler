from typing import List
from cube.codegen.emit import FuncEmission

from cube.ir.tensor import IRSubTensor
from cube.ir.adapter import IRAdapter
from cube.ir.adapter.prim import IRAdapterPrim

from cube.codegen.syntax.blocks import ClassBlock, FunctionBlock

from cube.codegen.emit import FuncEmission


class AutogradAdapterCodeGen(FuncEmission):
    """
    Generate autograd adapter code (PyTorch)
    """
    def __init__(self):

        self.fw_ins: List[IRSubTensor] = list()
        self.fw_body: List[str] = list()
        self.fw_ous: List[IRSubTensor] = list()

        self.bw_ins: List[IRSubTensor] = list()
        self.bw_body: List[str] = list()
        self.bw_ous: List[IRSubTensor] = list()

    def emit_prim(self, prim: IRAdapterPrim) -> str:
        if len(prim.inputs()) == 1:
            itensors = FuncEmission.tensor_name(prim.inputs()[0])
        else:
            itensors = FuncEmission.tuple_name(prim.inputs())
        kwargs = list()
        for name, val in prim.kwargs.items():
            kwargs.append(f'{name}={val}')
        kwargs = ', '.join(kwargs)
        outputs = FuncEmission.return_name(prim.outputs())
        code = f'{outputs} = {prim.signature}({itensors}, {kwargs})'
        return code

    def gen(self, fadapter: IRAdapter) -> List[str]:
        assert fadapter.isfw() and fadapter.differentiable and fadapter.custom, "generate autograd for a non-differentiable adapter"
        assert fadapter.mirror is not None
        name = AutogradAdapterCodeGen.name(fadapter)
        with ClassBlock(class_name=name, derived=['torch.autograd.Function']) as cb:
            # forward
            cb.insert_body('@staticmethod')
            finputs = [FuncEmission.tensor_name(t) for t in fadapter.inputs()]
            with FunctionBlock(func_name='forward', args=['ctx']+finputs) as fw:
                for prim in fadapter.prims:
                    fw.insert_body(self.emit_prim(prim))
                outputs = FuncEmission.return_name(fadapter.outputs())
                fw.insert_body(f'return {outputs}')
            cb.insert_body(fw.code)
            # backward
            cb.insert_body('@staticmethod')
            badapter: IRAdapter = fadapter.mirror
            binputs = [FuncEmission.tensor_name(t) for t in badapter.inputs()]
            with FunctionBlock(func_name='backward', args=['ctx']+binputs) as bw:
                for prim in badapter.prims:
                    bw.insert_body(self.emit_prim(prim))
                outputs = FuncEmission.return_name(badapter.outputs())
                bw.insert_body(f'return {outputs}')
            cb.insert_body(bw.code)
        return cb.code
    
    @staticmethod
    def name(adapter: IRAdapter) -> str:
        return f'Adapter{adapter.cid}'
