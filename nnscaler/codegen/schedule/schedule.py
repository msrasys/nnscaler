#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Optional, Tuple, Union
import copy
import inspect
import logging

from nnscaler.ir.cten import IRCell, IRTensor, IR
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


fsign = '{outputs} = nnscaler.runtime.executor.fexecute({name}, {model}, *{inputs}, requires_grad={req_grad})'
asign = '{outputs} = nnscaler.runtime.executor.aexecute({model}, *{inputs}, requires_grad={req_grad})'
bsign = '{input_grads} = nnscaler.runtime.executor.backward({name}, {input_tensors}, {output_tensors}, {output_grads})'
bi_sign = '{input_grads} = nnscaler.runtime.executor.backward_input({name}, {input_tensors}, {output_tensors}, {output_grads}, {weights})'
bw_sign = 'nnscaler.runtime.executor.backward_weight({name}, {weights})'
bw_fn_name = 'nnscaler.runtime.executor.backward_weight'
ssign = '{inputs} = nnscaler.runtime.executor.sync_tensors({inputs})'


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


        def to_tensor_names(val) -> str:
            """
            Return the tensor names in a complex data type.
            Currently support complex data type of Dict, List, Tuple
            """
            objs = IR.get_objects(val)
            return self.tuple_name([obj for obj in objs if isinstance(obj, IRTensor)])

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

        def _split_backward_codes(codes: List[str]) -> Tuple[List[str], List[str]]:
            """
            Split backward codes into two parts: backward_input and backward_weight
            """
            idx_weight = 0
            for idx, code in enumerate(codes):
                if code.strip().startswith(bw_fn_name):
                    idx_weight = idx
                    break
            else:
                raise RuntimeError(f"Expected backward_weight call in backward segment when using FBW schedule")
            return codes[:idx_weight], codes[idx_weight:]

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

                def _is_adapter(node: IRCell) -> bool:
                    node = node.cell if isinstance(node, ExeReuseCell) else node
                    return isinstance(node, IRAdapter)

                def _depends_totally_on(adapter: IRAdapter, node: IRCell) -> bool:
                    """
                    Check if the adapter depends on the node
                    """
                    node_outputs = set(node.outputs())
                    adapter_inputs = set(adapter.inputs())
                    return adapter_inputs.issubset(node_outputs)

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

                last_backward_node = None
                last_backward_weight_codes = []
                for line, node in enumerate(device_nodes):
                    # when use scheduler, skip reducer if it is not the last backward of same segments
                    if use_scheduler and _is_backward_segment(node):
                        _append_code(fb,
                            f'nnscaler.flags.RuntimeFlag.skip_reducer = '
                            f'{id(node) not in last_backward_node_oids !r}'
                        )

                    codes = self.emit_node(node)

                    if use_scheduler and _is_backward_segment(node) and CompileFlag.use_fbw:
                        if last_backward_node is not None:
                            _append_code(fb, last_backward_weight_codes, self._get_node_stream(last_backward_node))
                        last_backward_node = node
                        codes_input, codes_weight = _split_backward_codes(codes)
                        last_backward_weight_codes = codes_weight
                        _append_code(fb, codes_input, self._get_node_stream(node))
                    else:
                        if last_backward_node is not None:
                            if _is_adapter(node) and _depends_totally_on(node, last_backward_node):
                                # if the next node is an adapter that depends on the last backward,
                                # we need to emit the adapter before the last backward_weight codes
                                _append_code(fb, codes, self._get_node_stream(node))
                            else:
                                _append_code(fb, last_backward_weight_codes, self._get_node_stream(last_backward_node))
                                last_backward_node = None
                                last_backward_weight_codes = []
                                _append_code(fb, codes, self._get_node_stream(node))
                        else:
                            _append_code(fb, codes, self._get_node_stream(node))

                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        _append_code(fb, self.emit_release(tensors))

                if last_backward_node is not None:
                    _append_code(fb, last_backward_weight_codes, self._get_node_stream(last_backward_node))

            # return code
            if CompileFlag.async_comm:
                _append_code(fb, 'nnscaler.runtime.executor.AsyncCommHandler().drain_sends()')
                _append_code(fb, ssign.format(inputs=to_tensor_names(self.execplan.outputs())))
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
                if CompileFlag.async_comm:
                    _append_code(fb, 'nnscaler.runtime.executor.AsyncCommHandler().drain_sends()')
                    _append_code(fb, ssign.format(inputs=to_tensor_names(self.execplan.outputs())))
                outputs = self.return_name_complex(self.execplan.outputs())
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

    def emit_node(self, node: IRCell, force_no_grad: bool = False) -> List[str]:
        """
        Emit node / subgraph code
        """
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
                if CompileFlag.use_fbw:
                    codes = [bi_sign.format(
                        input_grads = input_grads_str,
                        name = f"'{self.node_name(unwrap_node.mirror)}'", # always use name of fw segment
                        input_tensors = input_tensors_str,
                        output_tensors = output_tensors_str,
                        output_grads = output_grads_str,
                        weights = 'model.parameters()'
                    )]
                    codes.append(bw_sign.format(
                        name = f"'{self.node_name(unwrap_node.mirror)}'", # always use name of fw segment
                        weights = 'model.parameters()'
                    ))
                else:
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
