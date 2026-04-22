#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""In pipeline parallelism, we want to execute micro-batches of samples on multiple devices
simultaneously. To achieve this, the computation dataflow graph is split into
several sub-graph (`IRSegment` in nnScaler), each of which has been assigned to related
devices (by default, we assumed a forward segment and its corresponding backward segment
share a same device placement).
In a pipeline schedule, each segment is replicated and annotated with different micro-batch
index (`Block` in nnScaler). Therefore, the schedule plan is a list of `Block` on each device.
To be valid, the blocks should satisfy the data dependency constraints, i.e., if block A
and block B share the same micro-batch index, and segment in B is dependent on segment in A,
then block A should be executed before block B. Note that there is no data dependency
between blocks with different micro-batch index.
Given the schedule (execution orders of blocks in each device), we can estimate the execution
time of the whole pipeline by:
- the execution time (span) of each block
- the data dependency between blocks belonging to different device groups
- the communication time between devices

In nnScaler's current implementation, we use a global view to define and validate the schedule
plan. To be more specific
- the execution time is discreted into integer steps
- each block takes an integer span to finish execution
- the communication time between blocks is omitted
- a valid schedule plan should satisfy that if block A depends on block B, then the start step
  of block A should not be earlier than end time of block B.

This implementation may be improved in the future, since
- it is hard to estimate the span of each block before the actual execution
- the communication time between blocks cannot be ignored in a real system, especially when
  the network bandwidth is limited
- the real start time of each block may be different from that defined in the schedule plan.
  Dependencies are materialized by inserting send and recv adapters between blocks. As a result,
  a block may start as soon as its input data is ready, rather than strictly following the start
  time in the schedule plan.


Multiple Stream Support:
In the current implementation, we also support users to specify stream configuration
for different types of communication
(e.g., inter-segment communication, result broadcasting, gradient reduction)
by setting its `stream_config` property.
The stream configuration will be attached to the corresponding adapters,
and can be used in codegen to generate stream context manager for the adapters.
This can help to achieve better performance in pipeline parallelism
by separating compute and communication to different streams.

Some implementation details:
- Stream information is attached to Segments/Adapters via `op_context` field,
    which is a dict that can be used to store any information for codegen.
    The key for stream context is 'stream_context', and the value is a `StreamContext` dataclass that contains the stream name and wait stream names.
- Stream information for other operations including WeightReducers, dataloaders, and zero_grad will be passed to `ExecutionPlan`
    and finally attached to those operations in scheduler codegen.

How to create a SchedulePlan for users:
1) Create a SchedulePlan with the graph and number of micro-batches.
2) Add segments to the plan by `add_segment` or `insert_step` interface, and specify stream context for each segment if needed.
3) Set `stream_config` property to set stream configuration for other operations.
4) Call `finish` to mark done.

"""

from typing import Dict, List,  Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import IntEnum

from nnscaler.ir.cten import IRCell
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter import IRWeightReducer
from nnscaler.ir.operator import IRDataOperation

from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.flags import CompileFlag


@dataclass
class StreamContext:
    """
    StreamContext is a dataclass that holds the stream context of an operation.
    Currently, all stream context information is ignored when tracing.
    But we can set it in codegen to improve the performance of the generated code.
    One example is we can seperate compute and communication to different streams
    to achieve better performance in pipeline parallelism.

    Args:
        stream: The stream name that the operation is executed on.
        wait_streams: The stream names that the operation needs to wait for before execution.
        record_events: The event names that the operation needs to record after execution.
        wait_events: The event names that the operation needs to wait for before execution.
    """
    stream: Optional[str] = None
    wait_streams: Optional[List[str]] = None
    record_events: Optional[List[str]] = None
    wait_events: Optional[List[str]] = None

    def __post_init__(self):
        if isinstance(self.wait_streams, str):
            self.wait_streams = [self.wait_streams]
        if isinstance(self.record_events, str):
            self.record_events = [self.record_events]
        if isinstance(self.wait_events, str):
            self.wait_events = [self.wait_events]


class Block:
    """
    A block is a node in SchedulePlan, representing an IRCell
    that is executed with input data of a given micro-batch index.
    """

    def __init__(self, cell: IRCell, micro_batch_id: int, span: int, stream_context: Optional[StreamContext] = None) -> None:
        """Create an execution block with IRCell on microbatch index. The
        block will take `span` steps to finish execution.
        """
        assert isinstance(cell, IRCell), f"Expected IRCell, but got {type(cell)}: {cell}"
        self._content: IRCell = cell
        self._micro_batch_id: int = micro_batch_id
        self._span = span
        self._stream_context: Optional[StreamContext] = stream_context

    def __eq__(self, other):
        if isinstance(other, Block):
            return other.content == self.content and other.mid == self.mid
        return False

    def __hash__(self) -> int:
        return hash((self._content, self._micro_batch_id))

    @property
    def device(self) -> Tuple[int]:
        return tuple(self._content.device)

    @property
    def mid(self) -> int:
        return self._micro_batch_id

    @property
    def content(self) -> IRCell:
        return self._content

    @property
    def span(self) -> int:
        return self._span

    @property
    def stream_context(self) -> Optional['StreamContext']:
        return self._stream_context

    def __repr__(self) -> str:
        return f"{self._content.cid}{'f' if self.content.isfw() else 'b'}{self._micro_batch_id}"


class ScheduleDependency:

    def __init__(self, graph: IRGraph) -> None:
        # adapter info
        self.graph: IRGraph = graph
        self.dataloaders : List[IRDataOperation] = []
        self.segments: List[IRSegment] = []
        self.adapters: List[IRAdapter] = []
        # the IRSegment that consumes the output of IRAdapter
        self.recvers: Dict[IRAdapter, IRSegment] = {}
        # the IRSegment that produces the input of IRAdapter
        self.senders: Dict[IRAdapter, IRSegment] = {}
        self.reducers: List[IRWeightReducer] = []

    def build(self):
        """
        Cluster operations and build dependency to identify the connected
        segments for each adapter.
        """
        # get all dataloaders
        self.dataloaders = list(self.graph.select(ntype=IRDataOperation, flatten=False))
        # get all segment
        segments: List[IRSegment] = self.graph.select(ntype=IRSegment, flatten=False)
        self.segments = segments
        # get all adapters
        for adapter in self.graph.select(ntype=IRAdapter, flatten=False):
            self.adapters.append(adapter)
            for segment in segments:
                if self.graph.depends(adapter, segment):
                    assert adapter not in self.recvers, \
                        f"Detected one adapter receives data from more than one segments"
                    self.recvers[adapter] = segment
                elif self.graph.depends(segment, adapter):
                    assert adapter not in self.senders, \
                        f"Detected one adapter {adapter} sends data to more than one segments"
                    self.senders[adapter] = segment
        # get all weight reducers
        self.reducers = self.graph.select(ntype=IRWeightReducer, flatten=False)

    def depends(self, prev: Block, next: Block) -> bool:
        return prev.mid == next.mid and self.graph.depends(prev.content, next.content)


@dataclass
class StreamConfig:
    """
    StreamConfig is a dataclass that holds the stream configuration
    for different types of operations.

    Args:
        dataloader: The stream context for dataloader operations.
        zero_grad: The stream context for zero_grad operations.
        inter_segment_move: The stream context for inter-segment communication adapters, which
            are used to transfer data between consecutive segment.
        result_broadcast: The stream context for result broadcast adapters,
            which are used to broadcast results to all ranks in the same scale unit after forward pass.
        grad_reduce: The stream context for gradient reduction adapters,
            which are used to reduce gradients across ranks.
        cuda_sync_required: Whether to call `torch.cuda.synchronize()` before and after train_step
    """

    dataloader: Optional[StreamContext] = None
    zero_grad: Optional[StreamContext] = None
    inter_segment_move: Optional[StreamContext] = None
    result_broadcast: Optional[StreamContext] = None
    grad_reduce: Optional[StreamContext] = None
    cuda_sync_required: bool = True


class PlanBase:

    def __init__(self, graph: IRGraph, _dependency: Optional[ScheduleDependency] = None):
        self._graph: IRGraph = graph
        self._blocks: List[Block] = []

        # execution time table
        # 1) the blocks in execution at each step
        self._step_blocks: List[List[Block]] = []
        # 2) the devices in execution at each step
        self._step_devices: List[Set[int]] = []
        # 3) the adapters executed *after* the segments on their steps
        self._step_adapters: List[List[Block]] = []

        # the start time step of block
        self._block_start_step: Dict[Block, int] = {}
        # dependency table
        self._dependency = _dependency if _dependency is not None \
            else ScheduleDependency(graph)

        # stream context for different adapters,
        # which can be used in codegen to generate stream context manager for the adapters.
        self._stream_config: StreamConfig = StreamConfig(cuda_sync_required=False)

        # topological sequence
        self._seqs: List[IRCell] = []

    @property
    def nsteps(self) -> int:
        return len(self._step_blocks)

    @property
    def graph(self) -> IRGraph:
        return self._graph

    @property
    def device(self) -> Tuple[int]:
        device = set()
        for devs in self._step_devices:
            device.update(devs)
        return tuple(device)

    @property
    def stream_config(self) -> StreamConfig:
        return self._stream_config

    @stream_config.setter
    def stream_config(self, stream_config: StreamConfig):
        self._stream_config = stream_config

    def _set_node_stream(self, node: IRCell, stream_context: StreamContext):
        if stream_context is not None:
            node.set_op_context('stream_context', stream_context)

    def nodes(self) -> Tuple[Block]:
        return tuple(self._seqs)

    def add_block(self, block: Block, step: int):
        """Add a block to start executing from step"""
        self._extend_step(step + block.span - 1)
        # check
        for t in range(block.span):
            for devid in block.device:
                if devid in self._step_devices[step+t]:
                    raise RuntimeError(
                        f"inserting confict at device {devid} of time step {step+t}: "
                        f"cannot execute multiple blocks at a same time step")
        for t in range(block.span):
            self._step_blocks[step+t].append(block)
            self._step_devices[step+t].update(block.device)
        self._block_start_step[block] = step
        self._blocks.append(block)
        return block

    def add_segment(
        self,
        seg: IRSegment, micro_batch_id: int,
        step: int, span: Optional[int] = 1,
        stream_context: Optional[StreamContext] = None,
    ) -> Block:
        """Add a segment to be executed with micro_batch_id data at step.

        The segments after `step` will keep unchanged.

        Args:
            seg (IRSegment): the segment to add to the plan
            micro_batch_id (int): the micro-batch id to execute the segment
            step (int): the step to execute the segment
            span (int): the time step costs to execute the segment

        Returns:
            block (Block): the block representing the segment
        """
        block = Block(seg, micro_batch_id, span, stream_context)
        self.add_block(block, step)
        return block

    def insert_step(
        self,
        step: int, seg: IRSegment,
        micro_batch_id: int, span: Optional[int] = 1,
        stream_context: Optional[StreamContext] = None,
    ) -> Block:
        """Insert `span` steps at current `step`.

        The segments after `step` will be pushed `span` time step for execution。

        Args:
            step (int): the step to insert
            seg (IRSegment): the segment to insert
            micro_batch_id (int): the micro-batch id to execute the segment
            span (int): the time step costs to execute the segment

        Returns:
            block (Block): the block representing the segment
        """
        # shift
        assert all(len(adapters) == 0 for adapters in self._step_adapters)
        for block in self._blocks:
            start = self.start(block)
            if start >= step:
                self._block_start_step[block] += span
            elif start + block.span > step:
                raise NotImplementedError(
                    f"Cannot shift the block {block} that is in execution on step {step}")
        # insert
        block = Block(seg, micro_batch_id, span, stream_context)
        for _ in range(span):
            self._step_blocks.insert(step, [block])
            self._step_devices.insert(step, set(seg.device))
            self._step_adapters.insert(step, [])
        self._block_start_step[block] = step
        self._blocks.append(block)
        return block

    def remove_step(self, step: int):
        """Remove the step if there are no blocks in execution.

        All the blocks after the `step` will be shifted earlier.
        This can only apply when no adapters are placed.

        Args:
            step (int): the step to remove

        Returns:
            None
        """
        if len(self._step_blocks[step]) > 0:
            raise RuntimeError(f"Cannot remove step {step} with blocks in execution")
        if len(self._step_adapters[step]) > 0:
            raise RuntimeError(f"Cannot remove step {step} with adapters in execution")
        # shift
        for block in self._blocks:
            if self.start(block) > step:
                self._block_start_step[block] -= 1
        self._step_blocks.pop(step)
        self._step_devices.pop(step)
        self._step_adapters.pop(step)

    def shrink(self):
        """Remove steps that have no blocks in execution

        Note the implementation is costly. Users should avoid
        calling it many times.
        """
        for step in range(self.nsteps-1, -1, -1):
            if len(self._step_blocks[step]) == 0:
                self.remove_step(step)

    def blocks(self, step: int) -> Tuple[Block]:
        """Get blocks in execution at the step"""
        if step >= self.nsteps:
            return ()
        blocks = self._step_blocks[step]
        return tuple(blocks)

    def start_blocks(self, step: int) -> Tuple[Block]:
        """Get blocks starting at the step"""
        if step >= self.nsteps:
            return ()
        blocks = self._step_blocks[step]
        blocks = tuple(blk for blk in blocks if self.start(blk) == step)
        return blocks

    def start(self, block: Block) -> int:
        """Get the start step of the block"""
        return self._block_start_step[block]

    def all_blocks(self) -> Tuple[Block]:
        """
        Get all segment blocks
        """
        return tuple(self._blocks)

    def depends(self, prev: Block, succ: Block) -> bool:
        """Check whether prev block directly depends on succ block"""
        return self._dependency.depends(prev, succ)

    def _extend_step(self, step: int):
        """Extend the maximal accessible steps of plan to `step` index"""
        if len(self._step_blocks) <= step:
            nextend = step - len(self._step_blocks) + 1
            self._step_blocks += [[] for _ in range(nextend)]
            self._step_devices += [set() for _ in range(nextend)]
            self._step_adapters += [[] for _ in range(nextend)]

    def _place_dataloader(self):
        """
        Place dataloaders together with segments
        """
        def insert_block(dl, mid, step):
            dl_block = Block(dl, mid, 1)
            # print(f'inserting microbatch {mid} at step {step} before {segment.name}{segment.cid}')
            self._blocks.append(dl_block)
            self._step_blocks[step+block.span-1].insert(0, dl_block)
            self._block_start_step[dl_block] = step+block.span-1

        # insert dataloaders to its devices before the first required segment
        for dl in self._dependency.dataloaders:
            self._set_node_stream(dl, self._stream_config.dataloader)
            inserted_mids = set()
            for step in range(self.nsteps):
                blocks = self.start_blocks(step)
                for block in blocks:
                    segment, mid = block.content, block.mid
                    if mid in inserted_mids: continue
                    if dl.device[0] not in segment.device: continue
                    if self.graph.depends(dl, segment):
                        insert_block(dl, mid, step)
                        inserted_mids.add(mid)
                        break
            # we guarantee each dataloader is inserted into the schedule plan,
            # in case that graph output requires the data from dataloader.
            for mid in range(self._num_microbatches):
                if mid not in inserted_mids:
                    insert_block(dl, mid, self.nsteps - 1)

    def topo_sort(self):
        """
        Sort the step-based execution plan and generates an execution sequence
        followed topological order.
        """
        self._seqs = []
        for step in range(self.nsteps):
            self._seqs += self.start_blocks(step)
            self._seqs += self._step_adapters[step]


class SchedulePlan(PlanBase):
    """
    A schedule plan leverages the fact no data dependency across different
    micro-batches. The schedule plan takes a step-based description to describe
    the scheduling of different micro-batch data.

    The step-based description describes every segment to be executed on which
    micro-batch data and executed at which step. The dependency requires segments
    inside one micro-batch should follow happen-before relationship:

      If segment A depends on segment B, then step of segment A must be smaller
      after segment B for a same micro-batch index.

    For each device, only up to one segment can be executed on a step.
    """

    def __init__(self, graph: IRGraph, num_microbatches: int):
        super().__init__(graph)
        # execution sequence
        self._num_microbatches = num_microbatches
        # bind to the graph
        graph._bind_schedule(self)

    @property
    def nmicros(self) -> int:
        """
        Get number of micro-batches
        """
        return self._num_microbatches

    @property
    def graph(self) -> IRGraph:
        return self._graph

    def finish(self):
        """
        Check whether the description contains full micro-batches
        """
        assert self.validate(), f"The schedule plan is not valid."

    def apply(self):
        """
        Insert generated adapters, dataloaders and reducers, and generat
        an execution sequence in topological order.
        This can only be called by system after adapter generation..
        """
        # step 1: build dependency for scheduling
        self._dependency.build()
        # step 2: insert adapters and dataloaders to the plan
        self._place_adapters()
        self._place_dataloader()
        # step 3: generate topological sequence, append reducers
        self.topo_sort()

    def validate(self) -> bool:
        """
        Validate the plan to check if it satisfies data dependency

        Returns:
            valid (bool): whether the plan is valid
        """
        for block1 in self._blocks:
            for block2 in self._blocks:
                if self._dependency.depends(block1, block2):
                    if self.start(block1) + block1.span > self.start(block2):
                        return False
        return True

    def _place_adapters(self):
        """
        Place adapters to make sure the communication happens
        correctly and efficiently.
        """
        for adapter in self._dependency.adapters:
            assert adapter in self._dependency.senders, (
                f"Detected an adapter\n\t{adapter}\ndoesn't have a sender segment. "
                f"This usually happens when its sender is dataloader or graph inputs."
                f"Please replicate dataloader to remove this adapter.")
            sender: IRSegment = self._dependency.senders[adapter]

            # since the schedule should return the same graph outputs on every device,
            # there will be adapters created to broadcast outputs of each microbatch
            # from the last-stage devices to all the devices.
            # These adapters don't have any dependent recver segment,
            # and will be placed at the end of the plan to not block the schedule execution.
            if adapter not in self._dependency.recvers:
                self._set_node_stream(adapter, self._stream_config.result_broadcast)
                for mid in range(self._num_microbatches):
                    self._step_adapters[self.nsteps-1].append(Block(adapter, mid, 1))
                continue

            self._set_node_stream(adapter, self._stream_config.inter_segment_move)

            # find sender step and insert adapter
            for step in range(self.nsteps):
                blocks = self.start_blocks(step)
                assert all(isinstance(blk, Block) for blk in blocks)
                segments = [block.content for block in blocks]
                mids = [block.mid for block in blocks]
                if sender in segments:
                    span = blocks[segments.index(sender)].span
                    mid = mids[segments.index(sender)]
                    self._step_adapters[step+span-1].append(Block(adapter, mid, 1))

    def topo_sort(self):
        super().topo_sort()
        for reducer in self._dependency.reducers:
            self._seqs.append(reducer)

    def str(self, show_max_steps: Optional[int] = None) -> str:
        if show_max_steps is None:
            show_max_steps = self.nsteps

        dscp = f"SchedulePlan:\n"
        if show_max_steps < self.nsteps:
            dscp += f"only show the first {show_max_steps} steps\n"

        sids: Dict[IRCell, int] = {}
        for block in self._blocks:
            if block.content not in sids:
                sids[block.content] = len(sids)

        for idx, (cell, sid) in enumerate(sids.items()):
            dscp += f'{cell.name}{cell.cid:<3} = {sid}; '
            if (idx + 1) % 3 == 0:
                dscp += '\n'

        dscp += '\nAnnotation: i(f/b)j = segment i on executing (forward/backward) microbatch j'
        for devid in sorted(self.device):
            timeline = '\n'
            step = 0
            while step < min(self.nsteps, show_max_steps):
                # segment
                have_block = False
                for block in self._step_blocks[step]:
                    if devid in block.device:
                        have_block = True
                        break
                if have_block:
                    blk_repr = f"{sids[block.content]}{'f' if block.content.isfw() else 'b'}{block.mid}"
                    timeline += f" {'-'.join([blk_repr] * block.span)}"
                    step += block.span
                else:
                    timeline += f" ---"
                    step += 1
                # adapter
                # have_block = False
                # for block in self._step_adapters[step]:
                #     if devid in block.device:
                #         have_block = True
                #         break
                # if have_block:
                #     timeline += ' {0: <5}'.format('adapt')
                # else:
                #     timeline += ' {0: <5}'.format('')
            if show_max_steps < self.nsteps:
                timeline += f" ... (remaining {self.nsteps-show_max_steps} steps)"
            dscp += timeline
        return dscp

    def __repr__(self):
        return self.str(show_max_steps=20)
