from typing import Dict, List,  Optional, Tuple, Set

from cube.ir.cten import IRCell
from cube.ir.adapter import IRAdapter
from cube.ir.adapter import IRWeightReducer
from cube.ir.operator import IRDataOperation

from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment



class Block:
    """
    A block is a node in SchedulePlan, representing an IRCell
    that is executed with input data of a given micro-batch index.
    """

    def __init__(self, cell: IRCell, micro_batch_id: int, span: int) -> None:
        """Create an execution block with IRCell on microbatch index. The 
        block will take `span` steps to finish execution.
        """
        assert isinstance(cell, IRCell), f"Expected IRCell, but got {type(cell)}: {cell}"
        self._content: IRCell = cell
        self._micro_batch_id: int = micro_batch_id
        self._span = span

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
    
    def dispatch(self, devid: int):
        return Block(self._content.dispatch(devid), self._micro_batch_id)

    def __repr__(self) -> str:
        return f"{self._content.cid}{'f' if self.content.isfw() else 'b'}{self._micro_batch_id}"


class ScheduleDependency:

    def __init__(self, graph: IRGraph) -> None:
        # adapter info
        self.graph: IRGraph = graph
        self.dataloaders : List[IRDataOperation] = []
        self.segments: List[IRSegment] = []
        self.adapters: List[IRAdapter] = []
        self.recvers: Dict[IRAdapter, IRSegment] = {}
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
    
    def depend(self, prev: Block, next: Block) -> bool:
        return prev.mid == next.mid and self.graph.depends(prev.content, next.content)


class PlanBase:

    def __init__(self, graph: IRGraph, _dependency: Optional[ScheduleDependency] = None):
        self._graph: IRGraph = graph
        self._segments: List[Block] = []
        self._step_segments: List[List[Block]] = []
        self._step_devices: List[Set[int]] = []
        # adapters executed *after* the segments on that step
        self._step_adapters: List[List[Block]] = []
        self._block_step: Dict[Block, int] = {}

        self._dependency = _dependency if _dependency is not None \
            else ScheduleDependency(graph)
        
        # topological sequence
        self._seqs: List[IRCell] = []

    @property
    def nsteps(self) -> int:
        return len(self._step_segments)
    
    @property
    def graph(self) -> IRGraph:
        return self._graph
    
    @property
    def device(self) -> Tuple[int]:
        device = set()
        for devs in self._step_devices:
            device.update(devs)
        return tuple(device)

    def nodes(self) -> Tuple[Block]:
        return tuple(self._seqs)

    def add_segment(self, seg: IRSegment, micro_batch_id: int, step: int, span: Optional[int] = 1) -> Block:
        """
        Add a segment `seg` to be executed with `micro-batch-id` data at step `step`.
        """
        self._extend_step(step + span - 1)
        if len(self._step_segments[step]) == 1 and isinstance(self._step_segments[0], PlanBase):
            assert False, "Cannot add an IRSegment into a step that already has Repetend."
        assert all(devid not in self._step_devices for devid in seg.device), \
            f"A device cannot execute multiple segments on a same step"
        block = Block(seg, micro_batch_id, span)
        for t in range(span):
            self._step_segments[step+t].append(block)
            self._step_devices[step+t].update(seg.device)
        self._block_step[block] = step
        self._segments.append(block)
        return block
    
    def segments(self, step: int) -> Tuple[Block]:
        """
        Get segment blocks at step
        """
        assert step < self.nsteps
        blocks = self._step_segments[step]
        blocks = tuple(blk for blk in blocks if self.step(blk) == step)
        return blocks
    
    def step(self, block: Block) -> int:
        """Get the step of the block
        """
        return self._block_step[block]
    
    def all_segments(self) -> Tuple[Block]:
        """
        Get all segment blocks
        """
        return tuple(self._segments)
    
    def _extend_step(self, step: int):
        """
        Extend the maximize plan with `step`.
        """
        if len(self._step_segments) <= step:
            nextend = step - len(self._step_segments) + 1
            self._step_segments += [[] for _ in range(nextend)]
            self._step_devices += [set() for _ in range(nextend)]
            self._step_adapters += [[] for _ in range(nextend)]

    def _place_dataloader(self):
        """
        Place dataloaders together with segments
        """
        # insert dataloaders to its devices before the first required segment
        for dl in self._dependency.dataloaders:
            inserted_mids = set()
            for step in range(self.nsteps):
                blocks = self.segments(step)
                for block in blocks:
                    segment, mid = block.content, block.mid
                    if mid in inserted_mids: continue
                    if dl.device[0] not in segment.device: continue
                    if self.graph.depends(dl, segment):
                        dl_block = Block(dl, mid, 1)
                        # print(f'inserting microbatch {mid} at step {step} before {segment.name}{segment.cid}')
                        self._step_segments[step+block.span-1].insert(0, dl_block)
                        self._block_step[dl_block] = step+block.span-1
                        inserted_mids.add(mid)
                        break

    def topo_sort(self):
        """
        Sort the step-based execution plan and generates an execution sequence
        followed topological order.
        """
        self._seqs = []
        for step in range(self.nsteps):
            self._seqs += self.segments(step)
            self._seqs += self._step_adapters[step]


class Repetend(PlanBase):
    """
    A repetend is a node in SchedulePlan, representing its nodes 
    will be repeatedly executed by `span` times witn growing
    micro-batch index.
    """

    def __init__(self, graph: IRGraph, dependency: ScheduleDependency,
                 span: int, step_segments: List[List[Block]], ):
        """
        @param graph IRGraph
        @param dependency: ScheduleDependency
        @param span int: the repeated execution time
        @param step_segments List[List[Block]]
        """
        super().__init__(graph, dependency)
        self._span = span
        self._extend_step(len(step_segments))
        self._step_segments = step_segments
        for step, blocks in enumerate(step_segments):
            devices = set()
            for block in blocks:
                devices.update(block.device)
            self._step_devices[step] = devices
        # the adapters that will be performed outside the repetend
        self._post_adapters: List[Block] = []

    @property
    def span(self) -> int:
        return self._span

    def nodes(self) -> Tuple[Block]:
        return tuple(self._seqs)
    
    def apply(self):
        self._place_adapters()
        self._place_dataloader()
        self.topo_sort()
    
    def _place_adapters(self):
        """
        Place adapters
        """
        # step1: unrolling repetend for one step
        cnts: Dict[IRSegment, int] = {}
        for step in range(self.nsteps):
            for blk in self.segments(step):
                cnts.setdefault(blk.content, 0)
                cnts[blk.content] += 1
        extended_blocks = []
        for step in range(self.nsteps):
            for blk in self.segments(step):
                extend_blk = Block(blk.content, blk.mid + cnts[blk.content], blk.span)
                extended_blocks.append(extend_blk)
        # step2: generate adapters for each step
        all_blocks = self.all_segments()
        for adapter, sender in self._dependency.senders.items():
            for step in range(self.nsteps):
                for block in self.segments(step):
                    if block.content != sender: continue
                    # sender adapter can be classified into three categories
                    # 1) its recver are in the same repetend
                    # 2) its recver are in neighbored repetend
                    #       - we don't allow send and recver in un-neighbored repetend
                    # 3) its recver are outside the repetend
                    recver = self._dependency.recvers[adapter]
                    rblock = Block(recver, block.mid, block.span)
                    ablock = Block(adapter, block.mid, 1)
                    # case 1)
                    if rblock in all_blocks:
                        self._step_adapters[step+block.span-1].append(ablock)
                    # case 2)
                    elif rblock in extended_blocks:
                        self._step_adapters[self.nsteps-1].append(Block(adapter, block.mid - cnts[blk.content], 1))
                        self._post_adapters.append(ablock)
                    # case 3)
                    else:
                        self._post_adapters.append(ablock)

    def get_post_adapters(self) -> List[Block]:
        return tuple(self._post_adapters)

    def __repr__(self):
        dscp = f'Repetend-{self.device}(span={self._span}\n'
        for step, blks in enumerate(self._step_segments):
            dscp += f'\n  Substep {step}:\n'
            for blk in blks:
                dscp += '  ' + repr(blk) + '\n'
        dscp += ')'
        return dscp


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

    def repeat(self, from_step: int, to_step: int, span: int) -> Repetend:
        """
        Create a repetend where the nodes inside the step ranges will
        be repeatedly executed by `span` time, with the increasing micro-batch
        index. The microbatch index among same segment must be
        consecutive.

        Note: calling this will shrink self.nsteps and the blocks begin from
        to_step will be shifted to the front of total steps by `to_step - from_step

        @param from_step int: starting (included) step
        @param to_step int: stopping (excluded) step
        @param span int: repeat time, i.e., number of increasing micro-batch index
        
        @return repetend Repetend
        """
        raise NotImplementedError("repeat is not supported.")
        assert 0 < from_step and from_step < self.nsteps
        assert 0 < to_step and to_step <= self.nsteps
        segment_blocks: List[List[Block]] = self._step_segments[from_step:to_step]
        repetend = Repetend(self._graph, self._dependency, span, segment_blocks)
        self._step_segments = self._step_segments[:from_step] + [[repetend]] + self._step_segments[to_step:]
        self._step_adapters = self._step_adapters[:from_step] + [[]] + self._step_adapters[to_step:]
        self._step_devs = self._step_devs[:from_step] + [set(repetend.device)] + self._step_devs[to_step:]
        return repetend

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
        # step 2: apply repetends
        for blocks in self._step_segments:
            if len(blocks) == 1 and isinstance(blocks[0], Repetend):
                blocks[0].apply()
        # step 3: apply this scheduling
        self._place_adapters()
        self._place_dataloader()
        # step 4: generate topological sequence
        self.topo_sort()

    def validate(self) -> bool:
        """
        Validate the plan to check if it satisfies data dependency

        @return valid bool
        """
        for block1 in self._segments:
            for block2 in self._segments:
                if self._dependency.depend(block1, block2):
                    if self.step(block1) >= self.step(block2):
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
            # find sender step and insert adapter
            for step in range(self.nsteps):
                blocks = self.segments(step)
                if len(blocks) == 1 and isinstance(blocks[0], Repetend):
                    self._step_adapters[step] += list(blocks[0].get_post_adapters())
                else:
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

    def __repr__(self) -> str:
        dscp = f"SchedulePlan:\n"

        sids: Dict[IRCell, int] = {}
        for block in self._segments:
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
            while step < self.nsteps:
                # segment
                have_block = False
                for block in self._step_segments[step]:
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
            dscp += timeline
        return dscp
