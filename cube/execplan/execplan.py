from typing import List, Optional
import copy
from cube.graph.adapter.adapter import IRAdapter
from cube.graph.operator.operator import IRBpOperation, IRFwOperation

from cube.ir.cten import IRCell
from cube.graph.graph import IRGraph


class ExectuionPlan:

    def __init__(self, graph: IRGraph):
        if not isinstance(graph, IRGraph):
            raise TypeError("Expected a list of ScheduleUnit")
        self.graph = graph
        self.device_seq = dict()
        for node in graph.nodes():
            if len(node.device) == 0:
                raise RuntimeError(f"Node device not set: {node}")
            for device in node.device:
                if device not in self.device_seq:
                    self.device_seq[device] = [node]
                else:
                    self.device_seq[device].append(node)

    def devices(self) -> List[int]:
        """
        Get device set
        """
        devices = list(self.device_seq.keys())
        devices.sort()
        return devices

    def sequence(self, device_id: int) -> List[IRCell]:
        """
        Get a copy of execution sequence for device id

        Note changing the list content will not change the execution plan.
        """
        if device_id not in self.device_seq:
            return list()
        return copy.copy(self.device_seq[device_id])

    def at(self, device_id: int) -> List[IRCell]:
        """
        Access the sequence for device id

        Note changing the list content will change the execution plan.
        """
        if device_id not in self.device_seq:
            return list()
        return self.device_seq[device_id]

    def set(self, device_id: int, seq: List[IRCell]):
        """
        Set device sequence
        """
        if not all([isinstance(su, IRCell) for su in seq]):
            raise TypeError("Expected a list of Cell")
        self.device_seq[device_id] = seq

    def draw(self, spans: Optional[List[int]] = None, outfile='./execplan.png'):
        """
        Draw the execution timeline.

        Args:
            span (List[int]): 
                length equal to schedule unit num.
                Each element stands for the time span for corresponding Cell

            outfile:
                the output file name
        """
        self.graph.reset_dependency()
        ndevice = len(self.devices())
        # timeline [ [ (start_time, end_time), ... ], ... ]
        device_timeline = [list() for _ in range(ndevice)]
        device_nodes = [list() for _ in range(ndevice)]

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

        def map2color(node):
            if isinstance(node, IRGraph):
                return map2color(node.nodes(0))
            if isinstance(node, IRFwOperation):
                return '#4472C4'  # excel blue
            if isinstance(node, IRBpOperation):
                return '#ED7D31'  # excel orange
            if isinstance(node, IRAdapter):
                return '#70AD47'  # excel green

        def map2name(node):
            if isinstance(node, IRGraph):
                if all([isinstance(n, IRFwOperation) for n in node.nodes()]):
                    return f'f{node._id}'
                if all([isinstance(n, IRBpOperation) for n in node.nodes()]):
                    if node.mirror is not None:
                        return f'b{node.mirror._id}'
            return str(node._id)

        if spans is None:
            print("Using default timing: fwop=1, bwop=2, adapter=0.1")
            spans = list()
            for node in self.graph.nodes():
                span = map2time(node)
                spans.append(span)

        graph = self.graph
        for node, span_time in zip(self.graph.nodes(), spans):
            for device in node.device:
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
                        if graph.happen_before(other_node, node):
                            start_time = max(start_time, end_time)
                            break
                device_timeline[device].append((start_time, start_time + span_time))
                device_nodes[device].append(node)

        # draw the timeline
        if outfile is not None:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            max_time = max(
                [tline[-1][1] for tline in device_timeline if len(tline) != 0]
            )
            plt.rcParams['figure.figsize'] = (4.0 * max_time // ndevice, 4.0)
            fig, ax = plt.subplots()
            renderer = fig.canvas.get_renderer()

            # xaxis
            ax.set_xlim((1, max_time))
            plt.xticks(list(range(1, int(max_time)+1, 1)))
            ax.xaxis.grid(True, linestyle='--')
            # yaxis
            ax.set_ylim((0.5, len(self.devices())+0.5))
            plt.yticks(list(range(1, len(self.devices())+1, 1)))
            ax.invert_yaxis()

            ax.set_aspect('equal')

            fontsize = 100
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
                    for fs in range(40, 1, -2):
                        txt.set_fontsize(fs)
                        tbox = txt.get_window_extent(renderer)
                        if tbox.x0 >= rbox.x0 and tbox.x1 <= rbox.x1 and tbox.y0 >= rbox.y0 and tbox.y1 <= rbox.y1:
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


    def __repr__(self):
        dscp = f'Execution Plan ({self.graph.name}):\n'
        for devid in self.devices():
            dscp += f'====> Device {devid}:\n'
            for node in self.sequence(devid):
                dscp += f'{node.module_repr()}\n'
        return dscp
