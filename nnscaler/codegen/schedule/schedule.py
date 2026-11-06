#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Dict, List, Optional, Tuple, Union, NamedTuple
import copy
import inspect
import logging
import re as _re

from nnscaler.ir.cten import IRCell, IRObject, IRTensor, IR
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.ir.adapter import IRWeightReducer, IRAdapter
from nnscaler.ir.adapter.prim import MovePrim
from nnscaler.graph.graph import IRSegment

from nnscaler.execplan.execplan import ExecutionPlan, ExeReuseCell

from nnscaler.codegen.emit import FuncEmission, IRValue
from nnscaler.codegen.syntax.symtable import SymbolTable
from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.codegen.syntax.blocks import FunctionBlock, Block
from nnscaler.flags import CompileFlag


_logger = logging.getLogger(__name__)


fsign = '{outputs} = nnscaler.runtime.executor.fexecute({name}, {model}, *{inputs}, requires_grad={req_grad}{stream_kwarg})'
asign = '{outputs} = nnscaler.runtime.executor.aexecute({model}, *{inputs}, requires_grad={req_grad}{stream_kwarg})'
bsign = '{input_grads} = nnscaler.runtime.executor.backward({name}, {input_tensors}, {output_tensors}, {output_grads}{stream_kwarg})'
bi_sign = '{input_grads} = nnscaler.runtime.executor.backward_input({name}, {input_tensors}, {output_tensors}, {output_grads}, {weights}{stream_kwarg})'
bw_sign = 'nnscaler.runtime.executor.backward_weight({name}, {weights}{stream_kwarg})'
ssign = '{inputs} = nnscaler.runtime.executor.sync_tensors({inputs})'


class _SampleAccess(NamedTuple):
    """A constant-key lookup path rooted at an original dataloader output."""
    sample_tid: int
    lookups: Tuple[Tuple[str, Tuple[str, ...]], ...]


class _SampleReadPlan(NamedTuple):
    """One node's replay prefix and the exact objects it reads and defines."""
    codes: Tuple[str, ...]
    inputs: Tuple[IRObject, ...]
    outputs: Tuple[IRObject, ...]


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
            'import torch', 'import nnscaler', 'import _operator', '']
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
        send_bundle_ranges = self._collect_send_bundle_ranges(device_nodes, device_map) \
            if use_scheduler else {}

        assert all(not isinstance(n, IRFwOperation) for n in device_nodes), \
            "Expected all forward operators have been grouped into IRSegment"
        self._validate_events(device_nodes)

        args = ['model'] + [self.tensor_name(t) for t in self.execplan.graph.inputs()]

        # Plan replay once. Lifetime analysis and both emitted schedules consume
        # the same dependencies/code, including synthetic random-access samples.
        sample_reads = self._plan_sample_reads(device_nodes)
        lifetime = LifeCycle(device_nodes, [], self.execplan.outputs(),
                             extra_inputs={node: plan.inputs for node, plan in sample_reads.items()})

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

        def _append_send_bundle_events(fb: FunctionBlock, events, before_node: bool):
            methods = {
                'reserve': 'reserve_send_bundle',
                'begin': 'begin_send_bundle',
            } if before_node else {'end': 'end_send_bundle'}
            for event_name, bundle_key in events:
                if method_name := methods.get(event_name):
                    args = repr(bundle_key) if event_name != 'end' else ''
                    _append_code(
                        fb,
                        'nnscaler.runtime.executor.AsyncCommHandler()'
                        f'.{method_name}({args})',
                    )

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
            if use_scheduler and CompileFlag.async_reducer:
                for code in self._emit_async_reducer_expectations(device_nodes):
                    _append_code(fb, code)

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

                pending_async_recvs = {}
                for line, node in enumerate(device_nodes):
                    unwrap_node = node.cell if isinstance(node, ExeReuseCell) else node
                    bundle_events = send_bundle_ranges.get(line, ())
                    _append_send_bundle_events(fb, bundle_events, before_node=True)
                    wait_codes = self._emit_async_recv_waits(node, pending_async_recvs, produced_tids)
                    if wait_codes:
                        _append_code(fb, wait_codes, self._get_node_stream(node))

                    # when use scheduler, skip reducer if it is not the last backward of same segments
                    if use_scheduler and _is_backward_segment(node):
                        skip_reducer = id(node) not in last_backward_node_oids
                        if CompileFlag.async_reducer:
                            # Scheduled pipeline async reducer tracks all local
                            # gradient contributions and launches each bucket
                            # when its expected contributions are complete.
                            skip_reducer = False
                        _append_code(fb,
                            f'nnscaler.flags.RuntimeFlag.skip_reducer = '
                            f'{skip_reducer !r}'
                        )
                    elif use_scheduler and isinstance(unwrap_node, IRWeightReducer):
                        _append_code(fb, 'nnscaler.flags.RuntimeFlag.skip_reducer = False')
                    codes = self.emit_node(
                        node,
                        produced_tids=produced_tids,
                        sample_reads=sample_reads.get(node),
                    )
                    _append_code(fb, codes, self._get_node_stream(node))
                    self._track_async_recv(node, pending_async_recvs)
                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        _append_code(fb, self.emit_release(tensors))
                    for out in IRSegment.get_objects_from_complex(node.outputs()):
                        if isinstance(out, IRObject):
                            produced_tids.add(out.tid)
                    _append_send_bundle_events(fb, bundle_events, before_node=False)
            # return code
            if CompileFlag.async_comm:
                _append_code(fb, 'nnscaler.runtime.executor.AsyncCommHandler().drain_sends()')
                _append_code(fb, ssign.format(inputs=to_tensor_names(self.execplan.outputs())))
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
                infer_pending_async_recvs = {}
                for line, node in enumerate(device_nodes):
                    if not node.isfw(): continue  # skip backward segments and adapters
                    bundle_events = send_bundle_ranges.get(line, ())
                    _append_send_bundle_events(fb, bundle_events, before_node=True)
                    wait_codes = self._emit_async_recv_waits(
                        node, infer_pending_async_recvs, infer_produced_tids
                    )
                    if wait_codes:
                        _append_code(fb, wait_codes, self._get_node_stream(node))

                    # execute
                    codes = self.emit_node(
                        node,
                        force_no_grad=True,
                        produced_tids=infer_produced_tids,
                        sample_reads=sample_reads.get(node),
                    )
                    _append_code(fb, codes, self._get_node_stream(node))
                    self._track_async_recv(node, infer_pending_async_recvs)
                    # release
                    tensors = lifetime.release_tensors_after_line(line)
                    tensors = [t for t in tensors if not isinstance(t, IRTensor) or not t.is_grad()]
                    if len(tensors) > 0 : # not necessarily to have one after each line
                        _append_code(fb, self.emit_release(tensors))
                    for out in IRSegment.get_objects_from_complex(node.outputs()):
                        if isinstance(out, IRObject):
                            infer_produced_tids.add(out.tid)
                    _append_send_bundle_events(fb, bundle_events, before_node=False)
                # return code
                if CompileFlag.async_comm:
                    _append_code(fb, 'nnscaler.runtime.executor.AsyncCommHandler().drain_sends()')
                    _append_code(fb, ssign.format(inputs=to_tensor_names(self.execplan.outputs())))
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

    @staticmethod
    def _pipeline_send_key(node: IRCell, device: int):
        unwrap_node = node.cell if isinstance(node, ExeReuseCell) else node
        if not isinstance(unwrap_node, IRAdapter):
            return None

        links = set()
        for prim in unwrap_node.prims:
            if not isinstance(prim, MovePrim) or len(prim.inputs()) == 0:
                continue
            src, dst = prim.kwargs['src'], prim.kwargs['dst']
            if src == device:
                links.add((src, dst))
        return tuple(sorted(links)) or None

    @classmethod
    def _collect_send_bundle_ranges(cls, nodes: List[IRCell], device: int):
        """Group adjacent pipeline sends to the same physical link into bundles."""
        events = {}
        line = 0
        while line < len(nodes):
            key = cls._pipeline_send_key(nodes[line], device)
            if key is None:
                line += 1
                continue

            end = line
            is_forward = nodes[line].isfw()
            while (
                end + 1 < len(nodes)
                and nodes[end + 1].isfw() == is_forward
                and cls._pipeline_send_key(nodes[end + 1], device) == key
            ):
                end += 1
            input_tids = {
                inp.tid
                for send_node in nodes[line:end + 1]
                for inp in IRSegment.get_objects_from_complex(send_node.inputs())
                if isinstance(inp, IRObject)
            }
            reserve_line = line
            for candidate in range(line - 1, -1, -1):
                output_tids = {
                    out.tid
                    for out in IRSegment.get_objects_from_complex(nodes[candidate].outputs())
                    if isinstance(out, IRObject)
                }
                if input_tids.intersection(output_tids):
                    reserve_line = candidate
                    break
            events.setdefault(reserve_line, []).append(('reserve', key))
            events.setdefault(line, []).append(('begin', key))
            events.setdefault(end, []).append(('end', key))
            line = end + 1
        return events

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

    def _emit_async_reducer_expectations(self, device_nodes: List[IRCell]) -> List[str]:
        reducer_params: Dict[int, List[str]] = {}
        for node in device_nodes:
            unwrap_node = self._unwrap_node(node)
            if not isinstance(unwrap_node, IRWeightReducer):
                continue
            reducer_params[unwrap_node._id] = [
                self.tensor_name(t, prefix_attr='model.')
                for t in unwrap_node.inputs()
            ]

        if not reducer_params:
            return []

        param_counts: Dict[str, int] = {}
        for node in device_nodes:
            unwrap_node = self._unwrap_node(node)
            if not isinstance(unwrap_node, IRSegment) or unwrap_node.isfw():
                continue
            for pname in self._segment_parameter_names(unwrap_node):
                param_counts[pname] = param_counts.get(pname, 0) + 1

        codes = []
        for reducer_id, param_names in sorted(reducer_params.items()):
            entries = []
            for pname in sorted(param_names):
                count = param_counts.get(pname, 1)
                entries.append(f'{pname}: {count}')
            codes.append(
                f'model.wreducer{reducer_id}.set_async_grad_expected_counts({{{", ".join(entries)}}})'
            )
        return codes

    def _segment_parameter_names(self, node: IRSegment) -> List[str]:
        if node.isfw():
            fw_node = node
        else:
            fw_node = node.mirror
        if not isinstance(fw_node, IRSegment):
            return []

        param_names = set()
        for inner_node in fw_node.nodes(flatten=True):
            for tensor in IRSegment.get_objects_from_complex([inner_node.inputs(), inner_node.kwargs]):
                if not isinstance(tensor, IRSubTensor) or not tensor.is_param():
                    continue
                param_names.add(self.tensor_name(tensor, prefix_attr='model.'))
        return sorted(param_names)

    def _unwrap_node(self, node: IRCell) -> IRCell:
        return node.cell if isinstance(node, ExeReuseCell) else node

    def _track_async_recv(self, node: IRCell, pending_async_recvs: dict):
        unwrap_node = self._unwrap_node(node)
        if not isinstance(unwrap_node, IRAdapter) or not self.is_async_recv_adapter(unwrap_node):
            return
        for out in IR.get_objects(node.outputs()):
            if isinstance(out, IRObject):
                pending_async_recvs[out.tid] = (unwrap_node, out)

    def _node_wait_inputs(self, node: IRCell, produced_tids: Optional[set]) -> List[IRObject]:
        unwrap_node = self._unwrap_node(node)
        if isinstance(unwrap_node, IRSegment) and not node.isfw():
            input_tensors, output_tensors, output_grads, _ = self.get_backward_callsite_io_tensors(node)
            return IR.get_objects((input_tensors, output_tensors, output_grads))
        return IR.get_objects(node.inputs())

    def _emit_async_recv_waits(
        self,
        node: IRCell,
        pending_async_recvs: dict,
        produced_tids: Optional[set],
    ) -> List[str]:
        codes = []
        seen_tids = set()
        for inp in self._node_wait_inputs(node, produced_tids):
            if not isinstance(inp, IRObject) or inp.tid not in pending_async_recvs or inp.tid in seen_tids:
                continue
            seen_tids.add(inp.tid)
            adapter, pending_out = pending_async_recvs.pop(inp.tid)
            adapter_name = self.node_name(adapter)
            tensor_name = self.tensor_name(pending_out)
            req_grad = isinstance(pending_out, IRTensor) and pending_out.requires_grad
            context = self._get_node_stream_context(node)
            track_stream = self.execplan.use_multi_streams or bool(context and context.stream)
            codes.append(
                f'{tensor_name} = nnscaler.runtime.executor.aexecute('
                f'model.{adapter_name}_wait, *({tensor_name}, ), requires_grad={req_grad}, '
                f'track_stream={track_stream})'
            )
        return codes

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

    def _collect_getitem_info(self) -> Dict[int, _SampleAccess]:
        """Record replayable sample accesses using original IR IDs, not names.

        An interleaved rank may need ``data['context'].get('scale', 1)``
        without scheduling the original lookup ops locally. Record that path
        once, including dict.get defaults, so each microbatch can replay it.
        Only constant arguments and paths rooted at the dataloader are safe.
        """
        nodes = self.execplan.graph.nodes(flatten=True)
        sample_tids = {
            output.tid for node in nodes if isinstance(node, IRDataOperation)
            for output in node.outputs() if isinstance(output, IRObject)
        }
        lookups = {}
        for node in nodes:
            if not isinstance(node, IRFwOperation) or node.signature not in {
                '_operator.getitem', 'builtins.dict.get',
            }:
                continue
            output = node.output(0)
            if isinstance(output, IRTensor) or not isinstance(output, IRObject):
                continue
            source = node.input(0)
            if not isinstance(source, IRObject) or isinstance(source, IRTensor):
                continue
            args = node.inputs()[1:]
            if any(not obj.is_constant for obj in IR.get_objects(args)):
                continue
            args = IR.try_unwrap(args)
            if not args or not isinstance(args[0], str):
                continue
            function = 'dict.get' if node.signature == 'builtins.dict.get' else node.signature
            lookups[output.tid] = source.tid, function, tuple(self.tensor_name(arg) for arg in args)

        info = {}
        for output_tid in lookups:
            source_tid = output_tid
            path = []
            while source_tid in lookups:
                source_tid, function, args = lookups[source_tid]
                path.append((function, args))
            if source_tid in sample_tids:
                info[output_tid] = _SampleAccess(source_tid, tuple(reversed(path)))
        return info

    def _plan_sample_reads(self, nodes) -> Dict[IRCell, _SampleReadPlan]:
        """Resolve metadata replay before emission, retaining object dependencies.

        Prefer an available input, then this microbatch's scheduled sample, and
        finally random access. The latter must not advance the dataloader cursor.
        Training and inference share these forward-only replay decisions.
        """
        accesses = self._collect_getitem_info()
        graph_inputs = IR.get_objects(self.execplan.graph.inputs())
        dataloader = next((obj for obj in graph_inputs
                           if self.tensor_name(obj).startswith('dataloader_')), None)
        data_ops = [node for node in self.execplan.graph.nodes(flatten=True)
                    if isinstance(node, IRDataOperation)]
        # get_micro_batch(mid) returns the single sample from this loader. It
        # cannot stand in for an arbitrary output of another data operation.
        # Replication preserves loader/output IDs. Count logical outputs, not
        # device copies; retain the full tuple to reject multi-output loaders.
        loader_outputs = {
            tuple(output.tid for output in node.outputs())
            for node in data_ops if dataloader is not None
            and node.input(0).tid == dataloader.tid
        }
        sole_outputs = next(iter(loader_outputs)) if len(loader_outputs) == 1 else ()
        fallback_tid = sole_outputs[0] if len(sole_outputs) == 1 else None
        scheduled_samples = {}
        for node in nodes:
            if isinstance(node, ExeReuseCell) and isinstance(node.cell, IRDataOperation):
                for original, sample in zip(node.cell.outputs(), node.outputs()):
                    if isinstance(sample, IRObject) and not isinstance(sample, IRTensor):
                        scheduled_samples[original.tid, node.micro_batch_id] = sample

        produced = {obj.tid for obj in graph_inputs}
        fallback_samples, plans = {}, {}
        for node in nodes:
            if not node.isfw():
                continue
            original = self._unwrap_node(node)
            mid = node.micro_batch_id if isinstance(node, ExeReuseCell) else None
            pairs = list(zip(original.inputs(), node.inputs()))
            available = {old.tid: actual for old, actual in pairs
                         if isinstance(old, IRObject) and isinstance(actual, IRObject)
                         and actual.tid in produced}
            codes, inputs, outputs = [], [], []
            for old, obj in pairs:
                if not isinstance(obj, IRObject) or isinstance(obj, IRTensor) or obj.tid in produced:
                    continue
                access = accesses.get(old.tid)
                if access is None:
                    continue
                source = available.get(access.sample_tid)
                if source is None and dataloader is not None and mid is not None:
                    sample = scheduled_samples.get((access.sample_tid, mid))
                    if sample is not None and sample.tid in produced:
                        source = sample
                    else:
                        if access.sample_tid != fallback_tid:
                            raise ValueError(
                                f'Cannot replay sample {access.sample_tid} for microbatch {mid}: '
                                'no matching sample is available and the dataloader root is ambiguous')
                        source = fallback_samples.get(mid)
                        if source is None:
                            source = fallback_samples[mid] = IRObject(f'_samples_{mid}')
                            codes.append(f'{self.tensor_name(source)} = '
                                         f'{self.tensor_name(dataloader)}.get_micro_batch({mid})')
                            inputs.append(dataloader)
                            outputs.append(source)
                if source is None:
                    continue
                inputs.append(source)
                expression = self.tensor_name(source)
                for function, args in access.lookups:
                    expression = f'{function}({expression}, {", ".join(args)})'
                codes.append(f'{self.tensor_name(obj)} = {expression}')
                outputs.append(obj)
                produced.add(obj.tid)
            if codes:
                plans[node] = _SampleReadPlan(tuple(codes), tuple(dict.fromkeys(inputs)), tuple(outputs))
            produced.update(obj.tid for obj in IR.get_objects(node.outputs()))
        return plans


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
            used = any(obj.tid in samples_tids for obj in IR.get_objects(unwrap_node.outputs()))
            for internal_node in unwrap_node.nodes(flatten=True):
                for inp in IR.get_objects((internal_node.inputs(), internal_node.kwargs)):
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
                if line.lstrip().startswith('del '):
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
        kept = []
        for i, line in enumerate(lines):
            if i in dead_lines:
                continue
            call = _re.search(r"fexecute\('(segment\d+)'", line)
            if call and call.group(1) in dead_samples_segments:
                # Preserve the segment's positional signature without reading
                # a sample it never consumes (or leaving an undefined name).
                for name in dead_vars:
                    line = _re.sub(r'\b' + _re.escape(name) + r'\b', 'None', line)
            if line.lstrip().startswith('del '):
                indent = line[:len(line) - len(line.lstrip())]
                names = [name.strip() for name in line.lstrip()[4:].split(',')]
                names = [name for name in names if name not in dead_vars]
                if not names:
                    continue
                line = indent + 'del ' + ', '.join(names)
            kept.append(line)
        return '\n'.join(kept)

    def _emit_segment_hook_code(
            self, hook, hook_meta,
            inputs_str: str,
            is_pre: bool,
            outputs_str: str = None,
            track_stream: bool = False,
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
            code = f'{fsig}(model, {repr(hook_meta)}, {inputs_str}, {kwargs})'
        else:
            code = f'{fsig}(model, {repr(hook_meta)}, {inputs_str}, {kwargs}, {outputs_str})'
        if track_stream:
            # Hooks enqueue CUDA work too, after/before the executor's own
            # lifetime scope. Include opaque inputs and returned containers.
            owners = inputs_str if is_pre else f'({inputs_str}, {outputs_str})'
            return [f'with nnscaler.runtime.stream.input_stream_lifetime({owners}, True):',
                    '    ' + code]
        return [code]

    def emit_detach(self, tensor: IRTensor) -> str:
        """
        Emit detach code
        """
        return f'{self.tensor_name(tensor)} = {self.tensor_name(tensor)}.detach()'

    def emit_node(self, node: IRCell, force_no_grad: bool = False,
                  produced_tids: set = None, sample_reads: Optional[_SampleReadPlan] = None) -> List[str]:
        """
        Emit node / subgraph code
        """
        node_inputs, node_outputs = node.inputs(), node.outputs()
        req_grad = any(t.requires_grad for t in node.outputs() if isinstance(t, IRTensor))
        req_grad = False if force_no_grad else req_grad

        # handle for forward
        inputs = self.tuple_name(node_inputs, skip_attr=True, prefix_attr='model.')
        outputs = self.return_name(node_outputs, skip_attr=True, prefix_attr='model.')

        unwrap_node = node.cell if isinstance(node, ExeReuseCell) else node
        name = self.node_name(unwrap_node)
        stream_context = self._get_node_stream_context(node)
        # A default-stream consumer can also receive data from a named stream.
        # Record actual resolved inputs inside the executor, after metadata
        # replay and any asynchronous communication callbacks.
        track_stream = bool(stream_context and stream_context.stream) or self.execplan.use_multi_streams
        # Omitted flags in direct/older callers default to safe tracking.
        # Only an explicitly single-stream plan may skip it.
        stream_kwarg = f', track_stream={track_stream}'
        prefix_codes = list(sample_reads.codes) if sample_reads else []
        if sample_reads and produced_tids is not None:
            produced_tids.update(obj.tid for obj in sample_reads.outputs)

        if isinstance(unwrap_node, IRSegment):
            # segment hooks
            pre_hook, post_hook, hook_meta = unwrap_node.pre_hook, unwrap_node.post_hook, unwrap_node.hook_meta
            # emit forward segment
            if node.isfw():
                codes = []
                if pre_hook:
                    codes += self._emit_segment_hook_code(
                        pre_hook, hook_meta, inputs, is_pre=True, track_stream=track_stream)
                codes += [fsign.format(
                    outputs = outputs,
                    name = f"'{name}'",
                    model = f'model.{name}',
                    inputs = inputs,
                    req_grad = req_grad,
                    stream_kwarg = stream_kwarg,
                )]
                if post_hook:
                    codes = codes + self._emit_segment_hook_code(
                        post_hook, hook_meta, inputs, is_pre=False,
                        outputs_str=outputs if len(node_outputs) <= 1 else f'({outputs})',
                        track_stream=track_stream,
                    )
            else:
                # get gradient computation arguments
                input_tensors, output_tensors, output_grads, input_grads = \
                        self.get_backward_callsite_io_tensors(node)
                # special handle for loss
                for idx, tensor in enumerate(output_grads):
                    if isinstance(tensor, IRSubTensor) and tensor.is_loss():
                        output_grads[idx] = None

                if produced_tids is not None:
                    terminal = {tensor.parent for tensor in IR.get_objects(self.execplan.outputs())
                                if isinstance(tensor, IRSubTensor)}
                    for idx, (output, grad) in enumerate(zip(output_tensors, output_grads)):
                        if (isinstance(grad, IRSubTensor) and grad.tid not in produced_tids
                                and output.parent in terminal):
                            # Logging-only terminal outputs need zero seeds,
                            # including paths arriving from an earlier stage.
                            output_grads[idx] = IRValue(f'torch.zeros_like({self.tensor_name(output)})')

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
                        weights = 'model.parameters()',
                        stream_kwarg = stream_kwarg,
                    )]
                    codes.append(bw_sign.format(
                        name = f"'{self.node_name(unwrap_node.mirror)}'", # always use name of fw segment
                        weights = 'model.parameters()',
                        stream_kwarg = stream_kwarg,
                    ))
                else:
                    codes = [bsign.format(
                        name = f"'{self.node_name(unwrap_node.mirror)}'", # always use name of fw segment
                        input_grads = input_grads_str,
                        input_tensors = input_tensors_str,
                        output_tensors = output_tensors_str,
                        output_grads = output_grads_str,
                        stream_kwarg = stream_kwarg,
                    )]

                bwd_input_str = f'({input_tensors_str}, {output_tensors_str}, {output_grads_str})'
                bwd_output_str = input_grads_str if len(input_grads) <= 1 else f'({input_grads_str})'
                if pre_hook:
                    codes = self._emit_segment_hook_code(
                        pre_hook, hook_meta, bwd_input_str, is_pre=True, track_stream=track_stream) + codes
                if post_hook:
                    codes += self._emit_segment_hook_code(
                        post_hook, hook_meta, bwd_input_str, is_pre=False,
                        outputs_str=bwd_output_str, track_stream=track_stream)

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
                req_grad = req_grad,
                stream_kwarg = stream_kwarg,
            )]

        elif isinstance(unwrap_node, IRWeightReducer):
            codes = [asign.format(
                outputs = outputs,
                model=f'model.{name}',
                inputs='()',
                req_grad=req_grad,
                stream_kwarg=stream_kwarg,
            )]

        else:
            raise RuntimeError(f"Unspported node type: {type(unwrap_node)}")

        codes = prefix_codes + codes

        if CompileFlag.line_timer:
            type_str = type(unwrap_node).__name__
            codes = [f'nnscaler.runtime.function.print_time({repr(type_str)})'] + codes

        return self._emit_stream_context(stream_context, codes)
