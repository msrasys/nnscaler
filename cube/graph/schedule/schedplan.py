from typing import Dict, List, Union, Callable, Optional, Tuple, Set
import itertools

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

    def __init__(self, cell: IRCell, micro_batch_id: int) -> None:
        """
        """
        assert isinstance(cell, IRCell), f"Expected IRCell, but got {type(cell)}: {cell}"
        self._block: IRCell = cell
        self._micro_batch_id: int = micro_batch_id

    @property
    def device(self) -> Tuple[int]:
        return tuple(self._block.device)

    @property
    def mid(self) -> int:
        return self._micro_batch_id
    
    @property
    def blk(self) -> IRCell:
        return self._block
    
    def dispatch(self, devid: int):
        return Block(self._block.dispatch(devid), self._micro_batch_id)

    def __repr__(self) -> str:
        return f'Block({self._micro_batch_id})-{self.device} : {self._block}'


class PlanBase:

    def __init__(self):
        self._step_devs: List[Set[int]] = []
        self._step_segments: List[List[Block]] = []
        # adapters executed after the segments on that step
        self._step_adapters: List[List[Block]] = []

        # topological sequence
        self._seqs: List[IRCell] = []

    @property
    def nsteps(self) -> int:
        return len(self._step_segments)
    
    def nodes(self) -> Tuple[Block]:
        return tuple(self._seqs)

    def add_segment(self, seg: IRSegment, micro_batch_id: int, step: int) -> Block:
        """
        Add a segment `seg` to be executed with `micro-batch-id` data at step `step`.
        """
        self._extend_step(step)
        assert all(devid not in self._step_devs[step] for devid in seg.device), \
            f"A step cannot execute multiple segments on a same device"
        block = Block(seg, micro_batch_id)
        self._step_segments[step].append(block)
        self._step_devs[step].update(seg.device)
        return block
    
    def _extend_step(self, step: int):
        if len(self._step_segments) <= step:
            nextend = step - len(self._step_segments) + 1
            self._step_segments += [[] for _ in range(nextend)]
            self._step_devs += [set() for _ in range(nextend)]
            self._step_adapters += [[] for _ in range(nextend)]

    def topo_sort(self):
        self._seqs = []
        for step in range(self.nsteps):
            self._seqs += self._step_segments[step]
            self._seqs += self._step_adapters[step]


class Repetend(PlanBase):
    """
    A repetend is a node in SchedulePlan, representing its nodes 
    will be repeatedly executed by `span` times witn growing
    micro-batch index.
    """

    def __init__(self, span: int, 
                 step_nodes: List[List[Block]],
                 step_adapters: List[List[IRAdapter]],
                 step_devs: List[Set[int]]):
        """
        @param span int: the repeated execution time
        """
        super().__init__()
        self._span = span
        self._step_segments = step_nodes
        self._step_adapters = step_adapters
        self._step_devs = step_devs
        # adapters out of for loop
        self._post_adapters: List[IRAdapter] = []

    @property
    def device(self) -> Tuple[int]:
        device = set()
        for devs in self._step_devs:
            device.update(devs)
        return tuple(device)

    @property
    def span(self) -> int:
        return self._span

    def nodes(self) -> Tuple[Block]:
        return tuple(self._seqs)

    def __repr__(self):
        dscp = f'Repetend-{self.device}(span={self._span}\n'
        for blk in self._seqs:
            dscp += '  ' + repr(blk) + '\n'
        dscp += ')'
        return dscp


class SchedulePlan(PlanBase):

    def __init__(self, graph: IRGraph, num_microbatches: int):
        super().__init__()
        self._graph: IRGraph = graph

        # adapter info
        self._dataloaders : List[IRDataOperation] = []
        self._segments: List[IRSegment] = []
        self._adapters: List[IRAdapter] = []
        self._recvers: Dict[IRAdapter, IRSegment] = {}
        self._senders: Dict[IRAdapter, IRSegment] = {}
        self._reducers: List[IRWeightReducer] = []
        # execution sequence
        self._device_seqs: Dict[int, Union[Repetend, IRSegment]] = {}
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
    def device(self) -> Tuple[int]:
        devs = set()
        for node in self._seqs:
            devs.update(node.device)
        return tuple(devs)
    
    @property
    def graph(self) -> IRGraph:
        return self._graph

    def steady_repeat(self, from_step: int, to_step: int, repeat: int):
        raise NotImplementedError("Not supported for steady representation")

    def finish(self) -> bool:
        """
        Check whether the description contains full micro-batches
        """
        pass

    def apply(self):
        """
        Insert generated adapters in the emitted sequence.
        This can be called by system only after generating adapters.
        """
        # step 1: identify connected segements for each generated adapter
        self._build_dependency()
        # step 2: place adapters, dataloaders
        self._place_adapters()
        self._place_dataloader()
        # step 3: generate topological sequence
        self.topo_sort()

    def _build_dependency(self):
        """
        Cluster operations and build dependency to identify the connected
        segments for each adapter.
        """
        # get all dataloaders
        self._dataloaders = list(self._graph.select(ntype=IRDataOperation, flatten=False))
        # get all segment
        segments: List[IRSegment] = self._graph.select(ntype=IRSegment, flatten=False)
        self._segments = segments
        # get all adapters
        for adapter in self._graph.select(ntype=IRAdapter, flatten=False):
            self._adapters.append(adapter)
            for segment in segments:
                if self._graph.depends(adapter, segment):
                    assert adapter not in self._recvers, \
                        f"Detected more than one segments to recv data from a same adapter"
                    self._recvers[adapter] = segment
                elif self._graph.depends(segment, adapter):
                    assert adapter not in self._senders, \
                        f"Detected more than one segments to send data from a same adapter"
                    self._senders[adapter] = segment
        # get all weight reducers
        self._reducers = self._graph.select(ntype=IRWeightReducer, flatten=False)

    def _place_adapters(self, cost_fn: Optional[Callable] = None):
        """
        Place adapters to make sure the communication happens
        correctly and efficiently.

        @param cost_fn Optional[Callable]: takes a segment and outputs
            the execution cost in float.By default (None), this assumes 
            each segement has the same execution cost of 1.0.
        """
        cost_fn = lambda x: 1.0 if cost_fn is None else cost_fn
        for adapter in self._adapters:
            assert adapter in self._senders
            sender: IRSegment = self._senders[adapter]
            # find sender step and insert adapter
            for step, blocks in enumerate(self._step_segments):
                segments = [block.blk for block in blocks]
                mids = [block.mid for block in blocks]
                if sender in segments:
                    mid = mids[segments.index(sender)]
                    self._step_adapters[step].append(Block(adapter, mid))

    def _place_dataloader(self):
        """
        Place dataloaders together with segments
        """
        for dl in self._dataloaders:
            for step, blocks in enumerate(self._step_segments):
                for block in blocks:
                    segment, mid = block.blk, block.mid
                    if self.graph.depends(dl, segment):
                        self._step_segments[step].insert(0, Block(dl, mid))
                        break

    def topo_sort(self):
        super().topo_sort()
        for reducer in self._reducers:
            self._seqs.append(reducer)

    def depends(self, prev: Block, next: Block) -> bool:
        return prev.mid == next.mid and self._graph.depends(prev.blk, next.blk)

    def __repr__(self) -> str:
        dscp = f"SchedulePlan:\n"
        for step in range(self.nsteps):
            dscp += f'\nStep {step}:\n'
            for segment in self._step_segments[step]:
                dscp += repr(segment) + '\n'
            for adapter in self._step_adapters[step]:
                dscp += repr(adapter) + '\n'
        return dscp
