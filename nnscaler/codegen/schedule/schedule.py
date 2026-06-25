#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Optional, Tuple, Union
import copy
import inspect
import logging

from nnscaler.ir.cten import IRCell, IRTensor, IR, IRObject
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

                # tid -> (adapter, output) for async-recv adapters whose wait is deferred
                pending_async_recvs = {}
                for line, node in enumerate(device_nodes):
                    # when use scheduler, skip reducer if it is not the last backward of same segments
                    if use_scheduler and _is_backward_segment(node):
                        _append_code(fb,
                            f'nnscaler.flags.RuntimeFlag.skip_reducer = '
                            f'{id(node) not in last_backward_node_oids !r}'
                        )
                    if CompileFlag.async_recv:
                        wait_codes = self._emit_async_recv_waits(node, pending_async_recvs)
                        if wait_codes:
                            _append_code(fb, wait_codes, self._get_node_stream(node))
                    codes = self.emit_node(node)
                    _append_code(fb, codes, self._get_node_stream(node))
                    if CompileFlag.async_recv:
                        self._track_async_recv(node, pending_async_recvs)
                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        _append_code(fb, self.emit_release(tensors))
            # return code
            outputs = self.return_name_complex(self.execplan.outputs())
            if CompileFlag.async_recv:
                # ensure any pending async-recv whose output is a step output
                # (e.g. the gathered loss) has completed before returning
                _append_code(fb, 'nnscaler.runtime.executor.AsyncCommHandler().drain()', force_flush=True)
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
                pending_async_recvs = {}
                for line, node in enumerate(device_nodes):
                    if not node.isfw(): continue  # skip backward segments and adapters
                    # execute
                    if CompileFlag.async_recv:
                        wait_codes = self._emit_async_recv_waits(node, pending_async_recvs, force_no_grad=True)
                        if wait_codes:
                            _append_code(fb, wait_codes, self._get_node_stream(node))
                    codes = self.emit_node(node, force_no_grad=True)
                    _append_code(fb, codes, self._get_node_stream(node))
                    if CompileFlag.async_recv:
                        self._track_async_recv(node, pending_async_recvs)
                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    tensors = [t for t in tensors if isinstance(t, IRTensor) and not t.is_grad()]
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        _append_code(fb, self.emit_release(tensors))
                # return code
                outputs = self.return_name_complex(self.execplan.outputs())
                if CompileFlag.async_recv:
                    _append_code(fb, 'nnscaler.runtime.executor.AsyncCommHandler().drain()', force_flush=True)
                code = f'return {outputs}'
                _append_code(fb, code, force_flush=True)
            gencode += fb.code
        gencode += ['']

        code = '\n'.join(gencode)
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

    def _emit_segment_hook_code(
            self, hook, hook_meta,
            inputs_str: str,
            is_pre: bool,
            outputs_str: str = None
        ) -> List[str]:
        """Generate hook call code for a segment.

        Args:
            hook: The hook function.
            hook_meta: Meta data passed to the hook.
            inputs_str: Tuple string of segment inputs.
            is_pre: True for pre-hook, False for post-hook.
            outputs_str: Tuple string of segment outputs (for post-hook).

        Returns:
            List[str]: lines of hook call code.
        """
        module_path = inspect.getmodule(hook).__name__
        fsig = f'{module_path}.{hook.__name__}'
        kwargs = '{}' # always pass empty kwargs for now, can be extended in the future if needed
        if is_pre:
            return [f'{fsig}(model, {repr(hook_meta)}, {inputs_str}, {kwargs})']
        else:
            return [f'{fsig}(model, {repr(hook_meta)}, {inputs_str}, {kwargs}, {outputs_str})']

    def emit_detach(self, tensor: IRTensor) -> str:
        """
        Emit detach code
        """
        return f'{self.tensor_name(tensor)} = {self.tensor_name(tensor)}.detach()'

    def _unwrap_node(self, node: IRCell) -> IRCell:
        return node.cell if isinstance(node, ExeReuseCell) else node

    def _track_async_recv(self, node: IRCell, pending_async_recvs: dict) -> None:
        """
        If `node` is an async-recv adapter (issued via `async_op=True`), record its
        output tensors as pending so a later consumer can emit the deferred wait.
        """
        unwrap_node = self._unwrap_node(node)
        if not isinstance(unwrap_node, IRAdapter) or not self.is_async_recv_adapter(unwrap_node):
            return
        for out in IR.get_objects(node.outputs()):
            if isinstance(out, IRObject):
                pending_async_recvs[out.tid] = (unwrap_node, out)

    def _node_wait_inputs(self, node: IRCell) -> List:
        """Return the io objects of `node` that may need a pending async-recv wait."""
        unwrap_node = self._unwrap_node(node)
        if isinstance(unwrap_node, IRSegment) and not node.isfw():
            input_tensors, output_tensors, output_grads, _ = self.get_backward_callsite_io_tensors(node)
            return IR.get_objects((input_tensors, output_tensors, output_grads))
        return IR.get_objects(node.inputs())

    def _emit_async_recv_waits(self, node: IRCell, pending_async_recvs: dict,
                               force_no_grad: bool = False) -> List[str]:
        """
        Emit deferred-wait calls for any pending async-recv tensors that `node`
        consumes. Each waited tensor is removed from `pending_async_recvs`.
        """
        codes = []
        seen_tids = set()
        for inp in self._node_wait_inputs(node):
            if not isinstance(inp, IRObject) or inp.tid not in pending_async_recvs or inp.tid in seen_tids:
                continue
            seen_tids.add(inp.tid)
            adapter, pending_out = pending_async_recvs.pop(inp.tid)
            adapter_name = self.node_name(adapter)
            tensor_name = self.tensor_name(pending_out)
            req_grad = (not force_no_grad) and isinstance(pending_out, IRTensor) and pending_out.requires_grad
            codes.append(
                f'{tensor_name} = nnscaler.runtime.executor.aexecute('
                f'model.{adapter_name}_wait, *({tensor_name}, ), requires_grad={req_grad})'
            )
        return codes

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
            # segment hooks
            pre_hook, post_hook, hook_meta = unwrap_node.pre_hook, unwrap_node.post_hook, unwrap_node.hook_meta
            # emit forward segment
            if node.isfw():
                codes = [fsign.format(
                    outputs = outputs,
                    name = f"'{name}'",
                    model = f'model.{name}',
                    inputs = inputs,
                    req_grad = req_grad
                )]
                if pre_hook:
                    codes = self._emit_segment_hook_code(pre_hook, hook_meta, inputs, is_pre=True) + codes
                if post_hook:
                    codes = codes + self._emit_segment_hook_code(
                        post_hook, hook_meta, inputs, is_pre=False,
                        outputs_str=outputs if len(node_outputs) <= 1 else f'({outputs})'
                    )
            else:
                # get gradient computation arguments
                input_tensors, output_tensors, output_grads, input_grads = \
                        self.get_backward_callsite_io_tensors(node)
                gen_inputs = (input_tensors, output_tensors, output_grads)
                # special handle for loss
                for idx, tensor in enumerate(output_grads):
                    if isinstance(tensor, IRSubTensor) and tensor.is_loss():
                        output_grads[idx] = None

                input_grads_str = self.return_name(input_grads)
                input_tensors_str = self.tuple_name(input_tensors, skip_attr=True, prefix_attr='model.')
                output_tensors_str = self.tuple_name(output_tensors, skip_attr=True, prefix_attr='model.')
                output_grads_str = self.tuple_name(output_grads, skip_attr=True, prefix_attr='model.')
                codes = [bsign.format(
                    name = f"'{self.node_name(unwrap_node.mirror)}'", # always use name of fw segment
                    input_grads = input_grads_str,
                    input_tensors = input_tensors_str,
                    output_tensors = output_tensors_str,
                    output_grads = output_grads_str
                )]

                bwd_input_str = f'({input_tensors_str}, {output_tensors_str}, {output_grads_str})'
                bwd_output_str = input_grads_str if len(input_grads) <= 1 else f'({input_grads_str})'
                if pre_hook:
                    codes = self._emit_segment_hook_code(pre_hook, hook_meta, bwd_input_str, is_pre=True) + codes
                if post_hook:
                    codes = codes + self._emit_segment_hook_code(post_hook, hook_meta, bwd_input_str, is_pre=False, outputs_str=bwd_output_str)

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
