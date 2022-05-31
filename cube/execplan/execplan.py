from typing import Callable, Dict, List, Optional
import copy
import numpy as np

from cube.ir.cten import IRCell
from cube.ir.adapter import IRAdapter
from cube.ir.operator import IRBpOperation, IRFwOperation
from cube.graph.graph import IRGraph, IRSegment


class ExectuionPlan:

    def __init__(self, graph: IRGraph):
        assert isinstance(graph, IRGraph), "Expected an IRGraph"
        self._graph = graph
        self._seq: Dict[int, List[IRCell]] = dict()

        # execution sequence for each device  
        for node in graph.nodes():
            if len(node.device) == 0:
                raise RuntimeError(f"Node device not set: {node}")
            for device in node.device:
                if device not in self._seq:
                    self._seq[device] = []
                self._seq[device].append(node)

        # adapter/segment dispatch
        for devid in self.devices():
            nodes = [node for node in self.at(devid) if isinstance(node, (IRAdapter, IRSegment))]
            while len(nodes) > 0:
                # dispatch
                fnode = nodes[0]
                fidx = self.at(devid).index(fnode)
                fnode_dev = fnode.dispatch(devid)
                self.at(devid)[fidx] = fnode_dev
                nodes.pop(0)
                if fnode.mirror is not None:
                    bidx = self.at(devid).index(fnode.mirror)
                    nodes.remove(fnode.mirror)
                    self.at(devid)[bidx] = fnode_dev.mirror

        # TODO: adapter support for return consistency
        for output in graph.outputs():
            for devid in self.devices():
                ptensors = [pt for pt in output.parent.ptensors if pt == output and devid in pt.device]
                assert len(ptensors) >= 1, f"Missing full graph output tensor {output} in device {devid}"

    @property
    def graph(self) -> IRGraph:
        return self._graph

    def devices(self) -> List[int]:
        """
        Get device set
        """
        devices = list(self._seq.keys())
        devices.sort()
        return devices

    def seq(self, devid: int) -> List[IRCell]:
        """
        Get a view of execution sequence for device id

        Note changing the list content will not change the execution plan.
        """
        assert devid in self._seq, f"device id {devid} not exists"
        return copy.copy(self._seq[devid])

    def at(self, devid: int) -> List[IRCell]:
        """
        Access the sequence for device id

        Note changing the list content will change the execution plan.
        """
        assert devid in self._seq, f"device id {devid} not exists"
        return self._seq[devid]

    def flatten(self, devid: int) -> List[IRCell]:
        """
        Flatten the sequence by expanding segments
        """
        assert devid in self._seq, f"device id {devid} not exists"
        nodes = []
        for node in self._seq[devid]:
            if isinstance(node, IRSegment):
                nodes += node.nodes()
            else:
                nodes.append(node)
        return nodes

    def set(self, devid: int, seq: List[IRCell]):
        """
        Set device sequence
        """
        if not all([isinstance(su, IRCell) for su in seq]):
            raise TypeError("Expected a list of Cell")
        self._seq[devid] = seq

    def analyze(self,
                map2time: Optional[Callable] = None,
                map2mem: Optional[Callable] = None,
                map2name: Optional[Callable] = None,
                outfile = None):
        """
        Draw the execution timeline.

        Args:
            span (List[int]): 
                length equal to schedule unit num.
                Each element stands for the time span for corresponding Cell

            outfile:
                the output file name
        """
        ndevice = len(self.devices())
        # timeline [ [ (start_time, end_time), ... ], ... ]
        device_timeline = [list() for _ in range(ndevice)]
        device_nodes = [list() for _ in range(ndevice)]
        device_mem = [0] * ndevice
        device_peak_mem = [0] * ndevice

        if map2time is None:
            def map2time(node):
                if isinstance(node, IRGraph):
                    span = 0
                    for node in node.nodes():
                        span += map2time(node)
                if isinstance(node, IRFwOperation):
                    return 1
                if isinstance(node, IRBpOperation):
                    return 2
                if isinstance(node, IRAdapter):
                    return 0.5
                return 0
        
        if map2mem is None:
            def map2mem(node):
                if isinstance(node, IRGraph):
                    peak_mem = 0
                    curr_mem = 0
                    for node in node.nodes():
                        curr_mem += map2mem(node)
                    peak_mem = max(curr_mem, peak_mem)
                if isinstance(node, IRFwOperation):
                    return 1
                if isinstance(node, IRBpOperation):
                    return -1
                return 0

        if map2name is None:
            def map2name(node):
                if isinstance(node, IRGraph):
                    if all([isinstance(n, IRFwOperation) for n in node.nodes()]):
                        return f'f{node._id}'
                    if all([isinstance(n, IRBpOperation) for n in node.nodes()]):
                        if node.mirror is not None:
                            return f'b{node.mirror._id}'
                if isinstance(node, IRFwOperation):
                    return f'f{node._id}'
                if isinstance(node, IRBpOperation):
                    return f'b{node.mirror._id}'
                return str(node._id)

        def map2color(node):
            if isinstance(node, IRGraph):
                return map2color(node.nodes(0))
            if isinstance(node, IRFwOperation):
                return '#4472C4'  # excel blue
            if isinstance(node, IRBpOperation):
                return '#ED7D31'  # excel orange
            if isinstance(node, IRAdapter):
                return '#70AD47'  # excel green

        for node in self.graph.nodes():
            span, mem = map2time(node), map2mem(node)
            for device in node.device:
                # memory
                device_mem[device] += mem
                if device_peak_mem[device] < device_mem[device]:
                    device_peak_mem[device] = device_mem[device]
                # tight execution if no dependency
                if len(device_timeline[device]) == 0:
                    start_time = 1
                else:
                    start_time = device_timeline[device][-1][1]
                # check dependency
                for devid, timeline in enumerate(device_timeline):
                    dev_seq = device_nodes[devid]
                    if devid == device:
                        continue
                    for nid, (_, end_time) in enumerate(timeline[::-1]):
                        other_node = dev_seq[::-1][nid]
                        if other_node in node.predecessors():
                            start_time = max(start_time, end_time)
                            break
                device_timeline[device].append((start_time, start_time + span))
                device_nodes[device].append(node)

        max_time = max(
            [tline[-1][1] for tline in device_timeline if len(tline) != 0]
        )
        max_mem = max(device_peak_mem)
        # max_mem = sum(device_peak_mem)

        # draw the timeline
        if outfile is not None:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            from matplotlib.ticker import AutoMinorLocator
            plt.close('all')
            plt.rcParams['figure.figsize'] = (4.0 * max_time // ndevice, 4.0)
            fig, ax = plt.subplots()
            renderer = fig.canvas.get_renderer()

            # xaxis
            ax.set_xlim((1, max_time))
            plt.xticks(
                ticks=np.arange(1.5, max_time+0.5, 1.0, dtype=float),
                labels=np.arange(1, max_time, 1, dtype=int)
            )
            minor_locator = AutoMinorLocator(2)
            plt.gca().xaxis.set_minor_locator(minor_locator)
            ax.xaxis.grid(which='minor', linestyle='--')
            # yaxis
            ax.set_ylim((0.5, len(self.devices())+0.5))
            plt.yticks(list(range(1, len(self.devices())+1, 1)))
            ax.invert_yaxis()

            fontsize = 40
            txts = list()
            for devid in range(ndevice):
                timeline = device_timeline[devid]
                nodes = device_nodes[devid]
                for node, (start, end) in zip(nodes, timeline):
                    if end - start == 0:
                        continue
                    # draw 
                    color = map2color(node)
                    rec = Rectangle((start, devid + 0.5), end-start, 1,
                                    color=color, ec='black', lw=1.5)
                    ax.add_artist(rec)
                    rx, ry = rec.get_xy()
                    cx = rx + rec.get_width() / 2.0
                    cy = ry + rec.get_height() / 2.0
                    anno = map2name(node)
                    txt = ax.text(x=cx, y=cy, s=anno, fontsize=40, ha='center', va='center', color='w')

                    rbox = rec.get_window_extent(renderer)
                    for fs in range(fontsize, 1, -2):
                        txt.set_fontsize(fs)
                        tbox = txt.get_window_extent(renderer)
                        if tbox.x0 > rbox.x0 and tbox.x1 < rbox.x1 and tbox.y0 > rbox.y0 and tbox.y1 < rbox.y1:
                            break
                    fontsize = min(fontsize, fs)
                    txts.append(txt)
            
            # set font size to same
            for txt in txts:
                txt.set_fontsize(fontsize)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize)
            plt.xlabel('Time Step', fontsize=fontsize)
            plt.ylabel('Device ID', fontsize=fontsize)

            # plt.grid()
            plt.tight_layout()
            plt.savefig(outfile)

        return max_time, max_mem

    def __repr__(self):
        dscp = f'Execution Plan ({self.graph.name}):\n'
        for devid in self.devices():
            dscp += f'====> Device {devid}:\n'
            for node in self._seq(devid):
                dscp += f'{node}\n'
        return dscp
