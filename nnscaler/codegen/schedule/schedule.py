#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Optional, Tuple, Union
import copy
import logging
import re as _re

from nnscaler.ir.cten import IRCell, IRObject, IRTensor, IR
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.ir.adapter import IRWeightReducer, IRAdapter
from nnscaler.graph.graph import IRSegment

from nnscaler.execplan.execplan import ExecutionPlan, ExeReuseCell

from nnscaler.codegen.emit import FuncEmission
from nnscaler.codegen.syntax.symtable import SymbolTable
from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.codegen.syntax.blocks import FunctionBlock, Block
from nnscaler.flags import CompileFlag


_logger = logging.getLogger(__name__)


class ScheduleCodeGen(FuncEmission):

    def __init__(
        self,
        execplan: ExecutionPlan,
        runtime_ndevs: Optional[int] = None,
        *,
        scale_ndevs: Optional[int] = None,
    ):
        """
        Create a schedule code generator

        Args:
            execplan (ExecutionPlan): execution plan
            runtime_ndevs (Optional[int]): the number of devices in runtime
            scale_ndevs (Optional[int]): Deprecated. Use `runtime_ndevs` instead
        """
        self.execplan = execplan
        self.devices: Tuple[int] = tuple(sorted(execplan.graph.device))
        if self.devices != tuple(range(len(self.devices))):
            raise ValueError(f'device must be consecutive')

        if scale_ndevs is not None:
            _logger.warning("scale_ndevs is deprecated, please use runtime_ndevs instead")
            if runtime_ndevs is not None:
                raise ValueError("You cannot use runtime_ndevs and scale_ndevs at the same time")
        self.runtime_ndevs: int = runtime_ndevs or scale_ndevs or len(self.devices)
        # we will scale the graph as data parallelism
        # when we have more devices than the number of devices used in the graph
        # here we don't need to do anything as things are already done in ModuleCodeGen.
        if self.runtime_ndevs % len(self.devices) != 0:
            raise ValueError(f'runtime_ndevs must be a multiple of {len(self.devices)}')
        self.enable_dp = self.runtime_ndevs > len(self.devices)

        # model full code
        self.init_code: List[str] = [
            '\n\n########## Generated Schedule Code ###########',
            'import torch', 'import nnscaler', '']
        # module member name
        self.symbols = SymbolTable()

    def gen(self, device: int, outfile=None, attach=None) -> str:
        """
        Generate scheduling code on device
        """
        gencode = copy.copy(self.init_code)
        device_map = device % len(self.devices)
        device_nodes = self.execplan.seq(device_map)
        # We will manually set the `skip_reducer` flag and `skip_zero_grad`
        # when we use scheduler (i.e. pipeline parallelism)
        # Otherwise, the caller of `_train_step` will set these flags to support gradient accumulation
        use_scheduler = self.execplan.graph.sched is not None

        assert all(not isinstance(n, IRFwOperation) for n in device_nodes), \
            "Expected all forward operators have been grouped into IRSegment"
        self._validate_events(device_nodes)

        lifetime = LifeCycle(device_nodes, [], self.execplan.outputs())

        args = ['model'] + [self.tensor_name(t) for t in self.execplan.graph.inputs()]

        last_stream = None
        buffered_codes = []

        def _get_codes_with_stream_context(codes: List[str], stream: Optional[str]) -> List[str]:
            if stream is None:
                return codes + ['']
            else:
                with Block(f'with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream({repr(stream)})):' ) as stream_block:
                    stream_block.insert_body(codes)
                return stream_block.code + ['']

        def _append_code(fb: FunctionBlock, code: Union[List[str], str], stream: Optional[str] = None, force_flush=False):
            codes = code if isinstance(code, list) else [code]

            nonlocal last_stream, buffered_codes

            if last_stream == stream:
                buffered_codes.extend(codes)
            else:
                if buffered_codes:
                    fb.insert_body(_get_codes_with_stream_context(buffered_codes, last_stream))

                last_stream = stream
                buffered_codes.clear()
                buffered_codes.extend(codes)

            if force_flush:
                if buffered_codes:
                    fb.insert_body(_get_codes_with_stream_context(buffered_codes, last_stream))
                buffered_codes.clear()
                last_stream = None

        with FunctionBlock(func_name='_train_step',
                           args=args) as fb:
            _append_code(fb, '_ = None')

            if use_scheduler:
                _append_code(fb, 'nnscaler.flags.RuntimeFlag.skip_zero_grad = False')
            _append_code(
                fb,
                self._emit_stream_context(self.execplan.zero_grad_stream_context, ['model.zero_grad()']),
                self.execplan.zero_grad_stream_context.stream
                    if self.execplan.zero_grad_stream_context else None
            )

            # body code
            if len(device_nodes) == 0:
                _append_code(fb, 'pass')
            else:
                def _is_backward_segment(node: IRCell) -> bool:
                    node = node.cell if isinstance(node, ExeReuseCell) else node
                    return isinstance(node, IRSegment) and not node.isfw()

                # collect backward segments that needs to reduce gradients
                # which are the last backward segments of every stage.
                # (Every segment will be used multiple times via `ExeReuseCell`)
                last_backward_node_oids = []
                if use_scheduler:
                    # Key: segment id
                    # Value: the last backward ExeReuseCell of the segment
                    last_backwards = {}
                    for node in device_nodes[::-1]:
                        if not _is_backward_segment(node):
                            continue
                        assert isinstance(node, ExeReuseCell), 'Expected ExeReuseCell for backward segment when using scheduler'
                        if node.cell.cid not in last_backwards:
                            last_backwards[node.cell.cid] = node
                    last_backward_node_oids = [id(node) for node in last_backwards.values()]

                for line, node in enumerate(device_nodes):
                    # when use scheduler, skip reducer if it is not the last backward of same segments
                    if use_scheduler and _is_backward_segment(node):
                        _append_code(fb,
                            f'nnscaler.flags.RuntimeFlag.skip_reducer = '
                            f'{id(node) not in last_backward_node_oids !r}'
                        )
                    codes = self.emit_node(node)
                    _append_code(fb, codes, self._get_node_stream(node))
                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        _append_code(fb, self.emit_release(tensors))
            # return code
            outputs = self.return_name_complex(self.execplan.outputs())
            code = f'return {outputs}'
            _append_code(fb, code, force_flush=True)

        gencode += fb.code

        gencode += ['', '']

        last_stream = None
        buffered_codes.clear()
        # infer code
        if not any(not node.isfw() for node in device_nodes):
            gencode += ['_infer_step = _train_step']
        else:
            with FunctionBlock(func_name='_infer_step', args=args) as fb:
                _append_code(fb, '_ = None')
                # body code
                if len(device_nodes) == 0:
                    _append_code(fb, 'pass')
                for line, node in enumerate(device_nodes):
                    if not node.isfw(): continue  # skip backward segments and adapters
                    # execute
                    codes = self.emit_node(node, force_no_grad=True)
                    _append_code(fb, codes, self._get_node_stream(node))
                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    tensors = [t for t in tensors if isinstance(t, IRTensor) and not t.is_grad()]
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        _append_code(fb, self.emit_release(tensors))
                # return code
                outputs = self.return_name_complex(self.execplan.outputs())
                code = f'return {outputs}'
                _append_code(fb, code, force_flush=True)
            gencode += fb.code
        gencode += ['']

        code = '\n'.join(gencode)
        # Post-process: fix undefined non-tensor refs in pipeline mode.
        # Collect getitem info from the IR graph (not from code text) so we
        # can see cross-stage extractions.
        getitem_info = self._collect_getitem_info()
        if getitem_info:
            code = self._fix_undefined_nontensor_refs(code, getitem_info)
        code = self._fix_undefined_backward_grads(code)
        # write to file
        if outfile:
            with open(outfile, 'a' if attach else 'w') as f:
                f.write(code)
        return code

    def _get_node_stream_context(self, node: IRCell):
        unwrap_node = node.cell if isinstance(node, ExeReuseCell) else node
        stream_context = node.get_op_context('stream_context')

        if not stream_context and isinstance(node, ExeReuseCell):
            stream_context = node.cell.get_op_context('stream_context')

        if not stream_context and isinstance(unwrap_node, IRWeightReducer):
            stream_context = self.execplan.weight_reducer_stream_context

        return stream_context

    def _get_node_stream(self, node: IRCell) -> Optional[str]:
        stream_context = self._get_node_stream_context(node)
        if stream_context:
            return stream_context.stream
        else:
            return None

    def _emit_stream_context(self, stream_context, codes: List[str]) -> List[str]:
        wait_stream_codes = []
        if stream_context and stream_context.wait_streams:
            for wait_stream in stream_context.wait_streams:
                wait_stream_codes.append(f'torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream({repr(wait_stream)}))')

        wait_event_codes = []
        if stream_context and stream_context.wait_events:
            for wait_event in stream_context.wait_events:
                wait_event_codes.append(f'nnscaler.runtime.device.DeviceGroup().get_event({repr(wait_event)}).wait()')

        record_event_codes = []
        if stream_context and stream_context.record_events:
            for record_event in stream_context.record_events:
                record_event_codes.append(f'nnscaler.runtime.device.DeviceGroup().get_event({repr(record_event)}).record()')

        return wait_stream_codes + wait_event_codes + codes + record_event_codes

    def _validate_events(self, nodes):
        """
        Validate that all events waited by nodes are recorded by some previous nodes in the graph
        """
        events = set()
        for node in nodes:
            stream_context = self._get_node_stream_context(node)
            if stream_context:
                if stream_context.wait_events:
                    for wait_event in stream_context.wait_events:
                        if wait_event not in events:
                            raise ValueError(f'Event `{wait_event}` is waited but has not been recorded')
                if stream_context.record_events:
                    for record_event in stream_context.record_events:
                        events.add(record_event)

    def _collect_getitem_info(self):
        """Scan IR graph for non-tensor getitem ops: ``_operator.getitem(samples, key)``.

        Returns a dict mapping a variable-name prefix (derived from the IRObject
        name, e.g. ``'getitem_1'``) to the extraction key (e.g. ``'context'``).
        """
        info = {}
        for node in self.execplan.graph.nodes(flatten=True):
            if not isinstance(node, IRFwOperation):
                continue
            if node.signature != '_operator.getitem':
                continue
            outputs = node.outputs()
            if not outputs:
                continue
            out = outputs[0]
            if isinstance(out, IRTensor):
                continue
            if not isinstance(out, IRObject):
                continue
            inputs = node.inputs()
            if len(inputs) < 2:
                continue
            key_obj = inputs[1]
            if isinstance(key_obj, IRObject) and isinstance(key_obj.value, str):
                key = key_obj.value
            elif isinstance(key_obj, str):
                key = key_obj
            else:
                continue
            # Get the codegen variable name for this IRObject and extract its
            # prefix (everything before the last _<tid> suffix).
            full_var = self.tensor_name(out)
            # full_var is e.g. "getitem_1_1080" where "getitem_1" is from out.name
            # and "1080" is out.tid.  The prefix is "getitem_1".
            prefix = full_var.rsplit('_', 1)[0] if '_' in full_var else full_var
            info[prefix] = key
        return info

    @staticmethod
    def _fix_undefined_nontensor_refs(code: str, getitem_info: dict) -> str:
        """Fix undefined non-tensor variable references in pipeline schedule code.

        ``getitem_info`` maps a variable-name prefix (e.g. ``'getitem_1'``) to
        the extraction key (e.g. ``'context'``).  For every fexecute() call that
        references a ``getitem_<prefix>_<tid>`` variable which has not been
        assigned, we insert ``var = _operator.getitem(samples_<tid2>, key)``
        using the co-occurring ``samples_*`` on the same line.
        """
        lines = code.split('\n')
        new_lines = []
        defined_vars = set()
        in_function = False

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('def _train_step(') or stripped.startswith('def _infer_step('):
                in_function = True
                defined_vars = set()
            elif stripped.startswith('def ') and in_function:
                in_function = False

            if in_function:
                assign_match = _re.match(r'^(\s*)(.+?)\s*=\s', line)
                if assign_match:
                    lhs = assign_match.group(2)
                    for var in _re.findall(r'\b(\w+)\b', lhs):
                        defined_vars.add(var)

                if 'nnscaler.runtime.executor.fexecute(' in line:
                    refs = _re.findall(r'\b(getitem_(\w+?)_(\d+))\b', line)
                    samples_match = _re.search(r'\b(samples_\d+)\b', line)
                    if samples_match:
                        samples_var = samples_match.group(1)
                        indent = len(line) - len(line.lstrip())
                        for full_name, prefix_part, tid in refs:
                            if full_name in defined_vars:
                                continue
                            prefix = f'getitem_{prefix_part}'
                            if prefix not in getitem_info:
                                continue
                            key = getitem_info[prefix]
                            fix = ' ' * indent + f"{full_name} = _operator.getitem({samples_var}, '{key}')"
                            new_lines.append(fix)
                            defined_vars.add(full_name)

            new_lines.append(line)

        return '\n'.join(new_lines)

    @staticmethod
    def _fix_undefined_backward_grads(code: str) -> str:
        """Replace undefined gradient variables with zero-grad in backward() calls.

        In pipeline mode, non-loss output tensors (e.g. z_loss returned for
        logging) may have gradient variables that are never assigned.  These
        should receive zero gradients so they don't inject spurious gradient
        flow.  Loss tensors are already handled by the ``is_loss()`` check
        which emits ``None`` (PyTorch defaults to ``ones_like`` for scalars).
        """
        lines = code.split('\n')
        new_lines = []
        defined_vars = set()
        in_function = False
        replaced_vars = set()

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('def _train_step(') or stripped.startswith('def _infer_step('):
                in_function = True
                defined_vars = set()
                replaced_vars = set()
                func_match = _re.search(r'def \w+\(([^)]*)\)', stripped)
                if func_match:
                    for arg in func_match.group(1).split(','):
                        defined_vars.add(arg.strip())
            elif stripped.startswith('def ') and in_function:
                in_function = False

            if in_function:
                assign_match = _re.match(r'^(\s*)(.+?)\s*=\s', line)
                if assign_match:
                    lhs = assign_match.group(2)
                    for var in _re.findall(r'\b(\w+)\b', lhs):
                        defined_vars.add(var)

                if 'nnscaler.runtime.executor.backward(' in line:
                    all_tuples = list(_re.finditer(r'\(([^()]*)\)', line))
                    if len(all_tuples) >= 2:
                        output_tensors_match = all_tuples[-2]
                        grads_match = all_tuples[-1]
                        ot_names = [v.strip() for v in output_tensors_match.group(1).split(',') if v.strip()]
                        grads_str = grads_match.group(1)
                        grad_names = [v.strip() for v in grads_str.split(',') if v.strip()]
                        new_grad_names = list(grad_names)
                        for i, gvar in enumerate(grad_names):
                            if gvar == 'None':
                                continue
                            if gvar not in defined_vars:
                                if i < len(ot_names):
                                    new_grad_names[i] = f'torch.zeros_like({ot_names[i]})'
                                else:
                                    new_grad_names[i] = 'None'
                                replaced_vars.add(gvar)
                        new_grads_str = ', '.join(new_grad_names)
                        if new_grads_str != grads_str:
                            line = line[:grads_match.start(1)] + new_grads_str + line[grads_match.end(1):]

                if stripped.startswith('del ') and replaced_vars:
                    del_vars = [v.strip() for v in stripped[4:].split(',')]
                    remaining = [v for v in del_vars if v not in replaced_vars]
                    if not remaining:
                        new_lines.append(line[:len(line) - len(stripped)] + 'pass')
                        continue
                    elif len(remaining) < len(del_vars):
                        line = line[:len(line) - len(stripped)] + 'del ' + ', '.join(remaining)

            new_lines.append(line)

        return '\n'.join(new_lines)

    def emit_detach(self, tensor: IRTensor) -> str:
        """
        Emit detach code
        """
        return f'{self.tensor_name(tensor)} = {self.tensor_name(tensor)}.detach()'

    def emit_node(self, node: IRCell, force_no_grad: bool = False) -> List[str]:
        """
        Emit node / subgraph code
        """
        fsign = '{outputs} = nnscaler.runtime.executor.fexecute({name}, {model}, *{inputs}, requires_grad={req_grad})'
        asign = '{outputs} = nnscaler.runtime.executor.aexecute({model}, *{inputs}, requires_grad={req_grad})'
        bsign = '{input_grads} = nnscaler.runtime.executor.backward({name}, {input_tensors}, {output_tensors}, {output_grads})'

        node_inputs, node_outputs = node.inputs(), node.outputs()
        # the real inputs in gencode
        gen_inputs = node_inputs
        req_grad = any(t.requires_grad for t in node.outputs() if isinstance(t, IRTensor))
        req_grad = False if force_no_grad else req_grad

        # handle for forward
        inputs = self.tuple_name(node_inputs, skip_attr=True, prefix_attr='model.')
        outputs = self.return_name(node_outputs, skip_attr=True, prefix_attr='model.')

        unwrap_node = node.cell if isinstance(node, ExeReuseCell) else node
        name = self.node_name(unwrap_node)
        stream_context = self._get_node_stream_context(node)

        if isinstance(unwrap_node, IRSegment):
            # emit forward segment
            if node.isfw():
                codes = [fsign.format(
                    outputs = outputs,
                    name = f"'{name}'",
                    model = f'model.{name}',
                    inputs = inputs,
                    req_grad = req_grad
                )]
            else:
                # get gradient computation arguments
                input_tensors, output_tensors, output_grads, input_grads = \
                        self.get_backward_callsite_io_tensors(node)
                gen_inputs = (input_tensors, output_tensors, output_grads)
                # special handle for loss
                for idx, tensor in enumerate(output_grads):
                    if isinstance(tensor, IRSubTensor) and tensor.is_loss():
                        output_grads[idx] = None
                codes = [bsign.format(
                    name = f"'{self.node_name(unwrap_node.mirror)}'",
                    input_grads = self.return_name(input_grads),
                    input_tensors = self.tuple_name(input_tensors, skip_attr=True, prefix_attr='model.'),
                    output_tensors = self.tuple_name(output_tensors, skip_attr=True, prefix_attr='model.'),
                    output_grads = self.tuple_name(output_grads, skip_attr=True, prefix_attr='model.')
                )]
                """
                In the end2end mode, although the graph's output may contain tensors that requires grad,
                like the loss tensor, the backward pass has been done by the nnscaler runtime by calling
                `nnscaler.runtime.executor.backward` in the generated code. In other words, the returned
                loss cannot be used by `loss.backward()`.
                In pipeline parallelism, loss tensors of micro-batches are generated at the last stage and
                transferred to remaining stages at the end for `_train_step` function. We add the detach
                operation here so that the backward graph's tensors can be deallocated right after the
                backward pass.
                """
                plan_outputs = IRCell.get_objects_from_complex(self.execplan.outputs())
                for tensor in output_tensors:
                    if not isinstance(tensor, IRTensor):
                        continue
                    if tensor in plan_outputs:
                        codes.append(self.emit_detach(tensor))

        elif isinstance(unwrap_node, IRDataOperation):
            codes = [f'{outputs} = {unwrap_node.signature}(*{inputs})']

        elif isinstance(unwrap_node, IRAdapter):
            codes = [asign.format(
                outputs = outputs,
                model = f'model.{name}',
                inputs = inputs,
                req_grad = req_grad
            )]

        elif isinstance(unwrap_node, IRWeightReducer):
            codes = [asign.format(
                outputs = outputs,
                model=f'model.{name}',
                inputs='()',
                req_grad=req_grad
            )]

        else:
            raise RuntimeError(f"Unspported node type: {type(unwrap_node)}")

        if stream_context and stream_context.stream:
            # register all input tensors to the current stream
            # to avoid cross-stream premature deallocation issue
            # see `Tensor.record_stream` in PyTorch for more details

            # NOTE: The dataloader output is not considered here. Two reasons:
            # 1) Dataloader output is wrapped in an IRObject, and internal IRTensors cannot be accessed here easily.
            # 2) Dataloader output is used as input of the first forward segment immediately after data loading
            # 3) Dataloader output is never del'ed. (its lifetime is not managed here due to #1)
            input_tensor_name = [
                self.tensor_name(t) for t in IR.get_objects(gen_inputs)
                if isinstance(t, IRTensor) and not t.is_attr()
            ]
            record_stream_codes = []
            for t in input_tensor_name:
                record_stream_codes.append(f'{t}.record_stream(torch.cuda.current_stream())')

            codes = record_stream_codes + codes

        if CompileFlag.line_timer:
            type_str = type(unwrap_node).__name__
            codes = [f'nnscaler.runtime.function.print_time({repr(type_str)})'] + codes

        return self._emit_stream_context(stream_context, codes)
