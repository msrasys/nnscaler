#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Optional

from nnscaler.ir.tensor import IRSubTensor
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import IRAdapterPrim

from nnscaler.codegen.syntax.blocks import ClassBlock, FunctionBlock

from nnscaler.codegen.emit import FuncEmission


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
            itensors = self.tensor_name(prim.inputs()[0])
        else:
            itensors = self.tuple_name(prim.inputs())
        kwargs = list()
        for name, val in prim.kwargs.items():
            kwargs.append(f'{name}={val}')
        kwargs = ', '.join(kwargs)
        outputs = self.return_name(prim.outputs())
        code = f'{outputs} = {prim.signature}({itensors}, {kwargs})'
        return code

    def gen(self, fadapter: IRAdapter, extra_args: Optional[List[str]] = None) -> List[str]:
        """
        Generate a `torch.autograd.Function` class for a custom differentiable adapter.

        Args:
            fadapter (IRAdapter): the forward adapter
            extra_args (Optional[List[str]]): names of non-tensor variables referenced by the
                prims (e.g. the scale unit index when the code is shared by all scale units).
                The autograd function has no `self`, so they are passed as the leading arguments
                of `forward` and saved on `ctx` for `backward`.
        """
        assert fadapter.isfw() and fadapter.is_custom_autograd, "generate autograd for a non-differentiable adapter"
        assert fadapter.mirror is not None
        name = self.name(fadapter)
        extra_args = list(extra_args or [])
        with ClassBlock(class_name=name, derived=['torch.autograd.Function']) as cb:
            # forward
            cb.insert_body('@staticmethod')
            finputs = [self.tensor_name(t) for t in fadapter.inputs()]
            with FunctionBlock(func_name='forward', args=['ctx'] + extra_args + finputs) as fw:
                for arg in extra_args:
                    fw.insert_body(f'ctx.{arg} = {arg}')
                for prim in fadapter.prims:
                    fw.insert_body(self.emit_prim(prim))
                outputs = self.return_name(fadapter.outputs())
                fw.insert_body(f'return {outputs}')
            cb.insert_body(fw.code)
            # backward
            cb.insert_body('@staticmethod')
            badapter: IRAdapter = fadapter.mirror
            binputs = [self.tensor_name(t) for t in badapter.inputs()]
            with FunctionBlock(func_name='backward', args=['ctx']+binputs) as bw:
                for arg in extra_args:
                    bw.insert_body(f'{arg} = ctx.{arg}')
                for prim in badapter.prims:
                    bw.insert_body(self.emit_prim(prim))
                outputs = self.return_name(badapter.outputs())
                # no gradient for the extra non-tensor arguments of forward
                bw.insert_body(f'return {"None, " * len(extra_args)}{outputs}')
            cb.insert_body(bw.code)
        return cb.code

    def name(self, adapter: IRAdapter) -> str:
        return f'Adapter{adapter.cid}'
