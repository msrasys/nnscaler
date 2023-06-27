
from typing import List, Dict, Any, Optional, Tuple
import copy
import warnings

from cube.ir.cten import IRCell, IRTensor
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.ir.tensor import IRSubTensor
from cube.ir.adapter import IRWeightReducer, IRAdapter
from cube.graph.graph import IRSegment

from cube.graph.schedule import IRScheduleStrategy

from cube.execplan.execplan import ExecutionPlan, ExeRepetend, ExeReuseCell

from cube.codegen.emit import FuncEmission
from cube.codegen.syntax.symtable import SymbolTable
from cube.codegen.lifecycle import LifeCycle
from cube.codegen.syntax.blocks import FunctionBlock, ForBlock


class ScheduleCodeGen(FuncEmission):

    def __init__(self, execplan: ExecutionPlan, scale_ndevs: Optional[int] = None):
        """
        Create Module code generator

        @param execplan ExecutionPlan
        @param scale_ndevs Optional[int]: scale to number of devices
        """
        self.execplan = execplan
        self.devices: Tuple[int] = tuple(sorted(execplan.graph.device))
        # model full code
        self.init_code: List[str] = [
            '\n\n########## Generated Schedule Code ###########',
            'import torch', 'import cube', '']
        # module member name
        self.symbols = SymbolTable()
        self._scale_to_ndevs = scale_ndevs

    def gen(self, device: int, outfile=None, attach=None) -> str:
        """
        Generate scheduling code on device
        """
        gencode = copy.copy(self.init_code)
        device_map = device if self._scale_to_ndevs is None else \
            device % len(self.devices)
        device_nodes = self.execplan.seq(device_map)

        assert all(not isinstance(n, IRFwOperation) for n in device_nodes), \
            "Expected all forward operators have been grouped into IRSegment"

        lifetime = LifeCycle(device_nodes, [], self.execplan.graph.outputs())

        args = ['model'] + [ScheduleCodeGen.tensor_name(t) for t in self.execplan.graph.inputs()]

        with FunctionBlock(func_name='_train_step', 
                           args=args) as fb:
            fb.insert_body('_ = None')
            fb.insert_body('model.zero_grad()')
            # body code
            if len(device_nodes) == 0:
                fb.insert_body('pass')
            # legacy hardcode strategy
            elif isinstance(self.execplan.graph.sched, IRScheduleStrategy):
                code = self.emit_legacy_schedplan(self.execplan.graph.sched, device)
                fb.insert_body(code)
            else:
                for line, node in enumerate(device_nodes):
                    # execute
                    codes = ScheduleCodeGen.emit_node(node)
                    fb.insert_body(codes)
                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        fb.insert_body(ScheduleCodeGen.emit_release(tensors))
            # return code
            outputs = ScheduleCodeGen.return_name_complex(self.execplan.graph.outputs())
            code = f'return {outputs}'
            fb.insert_body(code)
        gencode += fb.code

        gencode += ['', '']

        # infer code
        if not any(not node.isfw() for node in device_nodes):
            gencode += ['_infer_step = _train_step']
        else:
            # legacy hardcode strategy
            if isinstance(self.execplan.graph.sched, IRScheduleStrategy):
                warnings.warn('using legacy IRScheduleStrategy cannot generate inference code. '
                              'Switch to use scheduling without strategy')
            with FunctionBlock(func_name='_infer_step', 
                               args=args) as fb:
                fb.insert_body('_ = None')
                # body code
                if len(device_nodes) == 0:
                    fb.insert_body('pass')
                for line, node in enumerate(device_nodes):
                    if not node.isfw(): continue  # skip backward segments and adapters
                    # execute
                    codes = ScheduleCodeGen.emit_node(node, force_no_grad=True)
                    fb.insert_body(codes)
                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    tensors = [t for t in tensors if isinstance(t, IRTensor) and not t.is_grad()]
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        fb.insert_body(ScheduleCodeGen.emit_release(tensors))
                # return code
                outputs = ScheduleCodeGen.return_name_complex(self.execplan.graph.outputs())
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

    @staticmethod
    def emit_node(node: IRCell, force_no_grad: bool = False) -> List[str]:
        """
        Emit node / subgraph code
        """
        fsign = '{outputs} = cube.runtime.executor.fexecute({name}, {model}, *{inputs}, requires_grad={req_grad})'
        asign = '{outputs} = cube.runtime.executor.aexecute({model}, *{inputs}, requires_grad={req_grad})'
        bsign = '{input_grads} = cube.runtime.executor.backward({name}, {input_tensors}, {output_tensors}, {output_grads})'

        node_inputs, node_outputs = node.inputs(), node.outputs()
        req_grad = any(t.requires_grad for t in node.outputs() if isinstance(t, IRTensor))
        req_grad = False if force_no_grad else req_grad

        # handle for forward
        inputs = ScheduleCodeGen.tuple_name(node_inputs, skip_attr=True, prefix_attr='model.')
        outputs = ScheduleCodeGen.return_name(node_outputs, skip_attr=True, prefix_attr='model.')
        
        unwrap_node = node.cell if isinstance(node, ExeReuseCell) else node
        name = ScheduleCodeGen.node_name(unwrap_node)
        
        if isinstance(unwrap_node, IRSegment):
            # emit forward segment
            if node.isfw():
                code = fsign.format(
                    outputs = outputs,
                    name = f"'{name}'",
                    model = f'model.{name}',
                    inputs = inputs,
                    req_grad = req_grad
                )
            else:
                # get gradient computation arguments
                input_tensors, output_tensors, output_grads, input_grads = \
                        ScheduleCodeGen.get_backward_callsite_io_tensors(node)
                # special handle for loss
                for idx, tensor in enumerate(output_grads):
                    if isinstance(tensor, IRSubTensor) and tensor.is_loss():
                        output_grads[idx] = None
                code = bsign.format(
                    name = f"'{ScheduleCodeGen.node_name(unwrap_node.mirror)}'",
                    input_grads = ScheduleCodeGen.return_name(input_grads),
                    input_tensors = ScheduleCodeGen.tuple_name(input_tensors, skip_attr=True, prefix_attr='model.'),
                    output_tensors = ScheduleCodeGen.tuple_name(output_tensors, skip_attr=True, prefix_attr='model.'),
                    output_grads = ScheduleCodeGen.tuple_name(output_grads, skip_attr=True, prefix_attr='model.')
                )

        elif isinstance(unwrap_node, IRDataOperation):
            code = f'{outputs} = next(dataloader)'

        elif isinstance(unwrap_node, IRAdapter):
            code = asign.format(
                outputs = outputs,
                model = f'model.{name}',
                inputs = inputs,
                req_grad = req_grad
            )

        elif isinstance(unwrap_node, IRWeightReducer):
            code = asign.format(
                outputs = outputs,
                model=f'model.{name}',
                inputs='()',
                req_grad=req_grad
            )

        else:
            raise RuntimeError(f"Unspported node type: {type(unwrap_node)}")
        
        return [code]
    
    @staticmethod
    def emit_repetend(repetend: ExeRepetend) -> List[str]:
        """
        Emit code for executing a repetend
        """
        with ForBlock(var=None, iters=f'range({repetend.repeat})') as fb:
            for node in repetend.nodes():
                ncode = ScheduleCodeGen.emit_node(node)
                fb.insert_body(ncode)
        return fb.code

    @staticmethod
    def emit_legacy_schedplan(schedplan: IRScheduleStrategy, devid: int) -> List[str]:
        """
        Lagecy code
        """
        signature = schedplan.signature
        kwargs: Dict[str, Any] = schedplan.kwargs(devid)
        strkwargs = dict()
        for kwarg, val in kwargs.items():
            if isinstance(val, IRCell):
                name = 'model.' + ScheduleCodeGen.node_name(val)
            elif isinstance(val, (tuple, list)):
                brackets = ')' if len(val) != 1 else ',)'
                name = '(' + ', '.join('model.' + ScheduleCodeGen.node_name(n) \
                    if isinstance(n, IRCell) else str(n) for n in val) + brackets
            else:
                name = str(val)        
            strkwargs[kwarg] = name
        code = ', '.join(f'{kwarg}={name}' for kwarg, name in strkwargs.items())
        code = f'{signature}({code})'
        return [code]
