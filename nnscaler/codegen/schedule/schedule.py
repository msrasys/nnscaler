#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Optional, Tuple, Union
import copy
import inspect
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
            'import torch', 'import nnscaler', 'import chronotrigger.trace as ct', 'import _operator', '']
        # module member name
        self.symbols = SymbolTable()
        self._segment_stage_ids = self._build_segment_stage_ids()

    def _build_segment_stage_ids(self) -> dict[int, int]:
        segment_stage_ids = {}
        fsegments = [
            seg for seg in self.execplan.graph.select(ntype=IRSegment, flatten=False)
            if seg.isfw()
        ]
        for stage_id, segment in enumerate(fsegments):
            segment_stage_ids[segment.cid] = stage_id
            if isinstance(segment.mirror, IRSegment):
                segment_stage_ids[segment.mirror.cid] = stage_id
        return segment_stage_ids

    def _segment_trace_fields(self, node: IRSegment) -> Tuple[str, Optional[int]]:
        fw_segment = node if node.isfw() else node.mirror
        stage_id = self._segment_stage_ids.get(node.cid)
        if stage_id is None and isinstance(fw_segment, IRSegment):
            stage_id = self._segment_stage_ids.get(fw_segment.cid)
        segment_name = self.node_name(fw_segment) if isinstance(fw_segment, IRSegment) else self.node_name(node)
        return segment_name, stage_id

    @staticmethod
    def _wrap_trace(
        kind: str,
        entity: str,
        codes: List[str],
        *,
        virtual_stage: Optional[int] = None,
        peer: Optional[int] = None,
        micro_batch_id: Optional[int] = None,
        process_scope: bool = True,
    ) -> List[str]:
        if not codes:
            return codes
        fields = []
        if virtual_stage is not None:
            fields.append(f'vs={virtual_stage}')
        if peer is not None:
            fields.append(f'peer={peer}')
        if micro_batch_id is not None:
            fields.append(f'mb={micro_batch_id}')
        if not process_scope:
            fields.append('process_scope=False')
        kwargs = f", {', '.join(fields)}" if fields else ''
        with Block(f'with ct.range(ct.Kind.{kind}, {entity!r}{kwargs}):') as trace_block:
            trace_block.insert_body(codes)
        return trace_block.code

    @staticmethod
    def _wrap_scope(
        entity: str,
        codes: List[str],
        *,
        micro_batch_id: Optional[int] = None,
    ) -> List[str]:
        if not codes:
            return codes
        fields = f', mb={micro_batch_id}' if micro_batch_id is not None else ''
        with Block(f'with ct.scope({entity!r}{fields}):') as scope_block:
            scope_block.insert_body(codes)
        return scope_block.code

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

        getitem_info = self._collect_getitem_info()

        dataloader_var = None
        for t in self.execplan.graph.inputs():
            name = self.tensor_name(t)
            if name.startswith('dataloader_'):
                dataloader_var = name
                break

        # Segment inputs can include non-tensor sample objects whose producing
        # getitem op lives outside the local VPP schedule. Track the sample
        # object produced by each micro-batch dataloader read so fallbacks can
        # reuse it after it exists, or use random access without advancing the
        # dataloader cursor.
        sample_reads_by_mid = {}
        for node in device_nodes:
            unwrap = node.cell if isinstance(node, ExeReuseCell) else node
            if isinstance(unwrap, IRDataOperation):
                micro_batch_id = node.micro_batch_id if isinstance(node, ExeReuseCell) else None
                if micro_batch_id is None:
                    continue
                for out in node.outputs():
                    if isinstance(out, IRObject):
                        out_name = self.tensor_name(out)
                        if 'samples_' in out_name:
                            sample_reads_by_mid.setdefault(
                                micro_batch_id, (out_name, out.tid)
                            )
                            break

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

                produced_tids = set()
                for inp in IRSegment.get_objects_from_complex(self.execplan.graph.inputs()):
                    if isinstance(inp, IRObject):
                        produced_tids.add(inp.tid)

                fallback_state = {
                    'sample_reads_by_mid': sample_reads_by_mid,
                    'fallback_vars_by_mid': {},
                }
                for line, node in enumerate(device_nodes):
                    # when use scheduler, skip reducer if it is not the last backward of same segments
                    if use_scheduler and _is_backward_segment(node):
                        _append_code(fb,
                            f'nnscaler.flags.RuntimeFlag.skip_reducer = '
                            f'{id(node) not in last_backward_node_oids !r}'
                        )
                    codes = self.emit_node(
                        node,
                        produced_tids=produced_tids,
                        getitem_info=getitem_info,
                        dataloader_var=dataloader_var,
                        fallback_state=fallback_state,
                        rank=device,
                    )
                    _append_code(fb, codes, self._get_node_stream(node))
                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        _append_code(fb, self.emit_release(tensors))
                    for out in IRSegment.get_objects_from_complex(node.outputs()):
                        if isinstance(out, IRObject):
                            produced_tids.add(out.tid)
            # return code
            outputs = self.return_name_complex(self.execplan.outputs())
            _append_code(fb, 'nnscaler.runtime.executor.AsyncCommHandler().drain()')
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
                infer_produced_tids = set()
                for inp in IRSegment.get_objects_from_complex(self.execplan.graph.inputs()):
                    if isinstance(inp, IRObject):
                        infer_produced_tids.add(inp.tid)
                infer_fallback_state = {
                    'sample_reads_by_mid': sample_reads_by_mid,
                    'fallback_vars_by_mid': {},
                }
                for line, node in enumerate(device_nodes):
                    if not node.isfw(): continue  # skip backward segments and adapters
                    # execute
                    codes = self.emit_node(
                        node,
                        force_no_grad=True,
                        produced_tids=infer_produced_tids,
                        getitem_info=getitem_info,
                        dataloader_var=dataloader_var,
                        fallback_state=infer_fallback_state,
                        rank=device,
                    )
                    _append_code(fb, codes, self._get_node_stream(node))
                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    tensors = [t for t in tensors if isinstance(t, IRTensor) and not t.is_grad()]
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        _append_code(fb, self.emit_release(tensors))
                    for out in IRSegment.get_objects_from_complex(node.outputs()):
                        if isinstance(out, IRObject):
                            infer_produced_tids.add(out.tid)
                # return code
                outputs = self.return_name_complex(self.execplan.outputs())
                _append_code(fb, 'nnscaler.runtime.executor.AsyncCommHandler().drain()')
                code = f'return {outputs}'
                _append_code(fb, code, force_flush=True)
            gencode += fb.code
        gencode += ['']

        code = '\n'.join(gencode)
        dead_segs = self._find_dead_samples_segments(device_nodes)
        code = self._remove_dead_dataloader_reads(code, dead_segs)
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

    @staticmethod
    def _var_prefix(var_name: str) -> str:
        return var_name.rsplit('_', 1)[0] if '_' in var_name else var_name

    def _collect_getitem_info(self):
        """Scan IR graph for non-tensor extraction ops from sample dicts."""
        extract_signatures = {'_operator.getitem', 'builtins.dict.get'}
        info = {}
        for node in self.execplan.graph.nodes(flatten=True):
            if not isinstance(node, IRFwOperation):
                continue
            if node.signature not in extract_signatures:
                continue
            outputs = node.outputs()
            if not outputs:
                continue
            out = outputs[0]
            if isinstance(out, IRTensor) or not isinstance(out, IRObject):
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
            full_var = self.tensor_name(out)
            prefix = self._var_prefix(full_var)
            src = inputs[0]
            src_prefix = None
            if isinstance(src, IRObject):
                src_full = self.tensor_name(src)
                src_prefix = self._var_prefix(src_full)
            info[prefix] = (key, src_prefix)
        return info

    def _gen_nontensor_getitem_codes(self, node_inputs, produced_tids, getitem_info,
                                      dataloader_var=None, fallback_state=None,
                                      micro_batch_id=None):
        """Generate getitem calls for non-tensor inputs not produced locally."""
        if produced_tids is None or not getitem_info:
            return []
        codes = []
        local_fallback_var = None
        for inp in node_inputs:
            if isinstance(inp, IRTensor) or not isinstance(inp, IRObject):
                continue
            if inp.tid in produced_tids:
                continue
            var_name = self.tensor_name(inp)
            prefix = self._var_prefix(var_name)
            if prefix not in getitem_info:
                continue

            chain = []
            cur = prefix
            root_src_prefix = None
            while cur in getitem_info:
                key, src = getitem_info[cur]
                chain.append(key)
                root_src_prefix = src
                if src is None or src not in getitem_info:
                    break
                cur = src
            chain.reverse()
            if root_src_prefix is None:
                continue

            source_var = None
            for other_inp in node_inputs:
                if not isinstance(other_inp, IRObject):
                    continue
                other_name = self.tensor_name(other_inp)
                other_prefix = self._var_prefix(other_name)
                if other_prefix == root_src_prefix:
                    source_var = other_name
                    break

            if source_var is None:
                if dataloader_var and fallback_state is not None:
                    if local_fallback_var is None:
                        sample_var = None
                        if micro_batch_id is not None:
                            sample_reads = fallback_state.get('sample_reads_by_mid') or {}
                            read_info = sample_reads.get(micro_batch_id)
                            if read_info is not None:
                                read_var, read_tid = read_info
                                if produced_tids is not None and read_tid in produced_tids:
                                    sample_var = read_var

                        if sample_var is None and micro_batch_id is not None:
                            fallback_vars = fallback_state.setdefault(
                                'fallback_vars_by_mid', {}
                            )
                            sample_var = fallback_vars.get(micro_batch_id)
                            if sample_var is None:
                                sample_var = f'_nontensor_fallback_samples_{micro_batch_id}'
                                fallback_vars[micro_batch_id] = sample_var
                                codes.append(
                                    f"{sample_var} = {dataloader_var}.get_micro_batch({micro_batch_id})"
                                )
                        local_fallback_var = sample_var
                    source_var = local_fallback_var
                if source_var is None:
                    continue

            current_var = source_var
            for i, key in enumerate(chain[:-1]):
                inter_var = f'_nontensor_{inp.tid}_{i}'
                codes.append(f"{inter_var} = _operator.getitem({current_var}, '{key}')")
                current_var = inter_var
            leaf_key = chain[-1] if chain else getitem_info[prefix][0]
            codes.append(f"{var_name} = _operator.getitem({current_var}, '{leaf_key}')")
            produced_tids.add(inp.tid)
        return codes

    def _find_dead_samples_segments(self, device_nodes) -> set:
        """Find forward segments whose samples_* inputs are not consumed."""
        dead = set()
        for node in device_nodes:
            unwrap_node = node.cell if isinstance(node, ExeReuseCell) else node
            if not isinstance(unwrap_node, IRSegment) or not unwrap_node.isfw():
                continue
            samples_tids = set()
            for inp in unwrap_node.inputs():
                if isinstance(inp, IRObject) and not isinstance(inp, IRTensor):
                    name = self.tensor_name(inp)
                    if 'samples_' in name:
                        samples_tids.add(inp.tid)
            if not samples_tids:
                continue
            used = False
            for internal_node in unwrap_node.nodes(flatten=True):
                for inp in internal_node.inputs():
                    if isinstance(inp, IRObject) and inp.tid in samples_tids:
                        used = True
                        break
                if used:
                    break
            if not used:
                dead.add(self.node_name(unwrap_node))
        return dead

    @staticmethod
    def _remove_dead_dataloader_reads(code: str, dead_samples_segments: set = None) -> str:
        """Remove samples_* = next(dataloader) lines that are never used."""
        if dead_samples_segments is None:
            dead_samples_segments = set()
        lines = code.split('\n')

        next_assignments = []
        for i, line in enumerate(lines):
            match = _re.match(r'^\s+(samples_\w+)\s*=\s*next\(', line)
            if match:
                next_assignments.append((i, match.group(1)))
        if not next_assignments:
            return code

        var_assignment_lines = {}
        for i, var in next_assignments:
            var_assignment_lines.setdefault(var, set()).add(i)

        dead_vars = set()
        for var, assign_lines in var_assignment_lines.items():
            pattern = _re.compile(r'\b' + _re.escape(var) + r'\b')
            used_meaningfully = False
            for i, line in enumerate(lines):
                if i in assign_lines:
                    continue
                if not pattern.search(line):
                    continue
                fexecute_match = _re.search(r"fexecute\('(segment\d+)'", line)
                if fexecute_match and fexecute_match.group(1) in dead_samples_segments:
                    continue
                used_meaningfully = True
                break
            if not used_meaningfully:
                dead_vars.add(var)

        if not dead_vars:
            return code

        dead_lines = set()
        for i, var in next_assignments:
            if var in dead_vars:
                dead_lines.add(i)
        lines = [line for i, line in enumerate(lines) if i not in dead_lines]
        return '\n'.join(lines)

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

    def emit_node(self, node: IRCell, force_no_grad: bool = False,
                  produced_tids: set = None, getitem_info: dict = None,
                  dataloader_var: str = None, fallback_state: dict = None,
                  rank: Optional[int] = None) -> List[str]:
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
        micro_batch_id = node.micro_batch_id if isinstance(node, ExeReuseCell) else None

        if isinstance(unwrap_node, IRSegment):
            # segment hooks
            pre_hook, post_hook, hook_meta = unwrap_node.pre_hook, unwrap_node.post_hook, unwrap_node.hook_meta
            # emit forward segment
            if node.isfw():
                segment_name, virtual_stage = self._segment_trace_fields(unwrap_node)
                getitem_codes = self._gen_nontensor_getitem_codes(
                    node_inputs,
                    produced_tids,
                    getitem_info,
                    dataloader_var=dataloader_var,
                    fallback_state=fallback_state,
                    micro_batch_id=micro_batch_id,
                )
                codes = getitem_codes
                if pre_hook:
                    codes += self._emit_segment_hook_code(pre_hook, hook_meta, inputs, is_pre=True)
                operation_codes = [fsign.format(
                    outputs = outputs,
                    name = f"'{name}'",
                    model = f'model.{name}',
                    inputs = inputs,
                    req_grad = req_grad,
                )]
                codes += self._wrap_trace(
                    'FWD',
                    segment_name,
                    operation_codes,
                    virtual_stage=virtual_stage,
                    micro_batch_id=micro_batch_id,
                    process_scope=False,
                )
                if post_hook:
                    codes = codes + self._emit_segment_hook_code(
                        post_hook, hook_meta, inputs, is_pre=False,
                        outputs_str=outputs if len(node_outputs) <= 1 else f'({outputs})'
                    )
            else:
                segment_name, virtual_stage = self._segment_trace_fields(unwrap_node)
                # get gradient computation arguments
                input_tensors, output_tensors, output_grads, input_grads = \
                        self.get_backward_callsite_io_tensors(node)
                gen_inputs = (input_tensors, output_tensors, output_grads)
                # special handle for loss
                for idx, tensor in enumerate(output_grads):
                    if isinstance(tensor, IRSubTensor) and tensor.is_loss():
                        output_grads[idx] = None

                if produced_tids is not None:
                    filtered_ot, filtered_og = [], []
                    for output_tensor, output_grad in zip(output_tensors, output_grads):
                        if output_grad is None or (
                            isinstance(output_grad, IRSubTensor)
                            and output_grad.tid in produced_tids
                        ):
                            filtered_ot.append(output_tensor)
                            filtered_og.append(output_grad)
                    output_tensors = filtered_ot
                    output_grads = filtered_og

                gen_inputs = (input_tensors, output_tensors, output_grads)
                input_grads_str = self.return_name(input_grads)
                input_tensors_str = self.tuple_name(input_tensors, skip_attr=True, prefix_attr='model.')
                output_tensors_str = self.tuple_name(output_tensors, skip_attr=True, prefix_attr='model.')
                output_grads_str = self.tuple_name(output_grads, skip_attr=True, prefix_attr='model.')
                codes = [bsign.format(
                    name = f"'{self.node_name(unwrap_node.mirror)}'", # always use name of fw segment
                    input_grads = input_grads_str,
                    input_tensors = input_tensors_str,
                    output_tensors = output_tensors_str,
                    output_grads = output_grads_str,
                )]
                codes = self._wrap_trace(
                    'BWD',
                    segment_name,
                    codes,
                    virtual_stage=virtual_stage,
                    micro_batch_id=micro_batch_id,
                )

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
            already_produced = False
            if produced_tids is not None:
                for out in node_outputs:
                    if isinstance(out, IRObject) and out.tid in produced_tids:
                        already_produced = True
                        break
            if already_produced:
                codes = []
            else:
                codes = [f'{outputs} = {unwrap_node.signature}(*{inputs})']

        elif isinstance(unwrap_node, IRAdapter):
            codes = [asign.format(
                outputs = outputs,
                model = f'model.{name}',
                inputs = inputs,
                req_grad = req_grad
            )]
            codes = self._wrap_scope(
                self.node_name(unwrap_node),
                codes,
                micro_batch_id=micro_batch_id,
            )

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
