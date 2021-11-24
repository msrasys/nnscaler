from typing import List, Optional
import copy

from cube.schedule.sugraph import SUGraph
from cube.schedule.su import SUType, ScheduleUnit


class ExectuionPlan:

    def __init__(self, sugraph: SUGraph):
        if not isinstance(sugraph, SUGraph):
            raise TypeError("Expected a list of ScheduleUnit")
        self.sugraph = sugraph
        self.device_seq = dict()
        for su in sugraph.sus():
            if len(su.device) == 0:
                raise RuntimeError(f"device not set: SU {su}")
            for device in su.device:
                if device not in self.device_seq:
                    self.device_seq[device] = [su]
                else:
                    self.device_seq[device].append(su)

    def devices(self) -> List[int]:
        """
        Get device set
        """
        devices = list(self.device_seq.keys())
        devices.sort()
        return devices

    def sequence(self, device_id: int) -> List[ScheduleUnit]:
        """
        Get a copy of execution sequence for device id

        Note changing the list content will not change the execution plan.
        """
        if device_id not in self.device_seq:
            return list()
        return copy.copy(self.device_seq[device_id])

    def at(self, device_id: int) -> List[ScheduleUnit]:
        """
        Access the sequence for device id

        Note changing the list content will change the execution plan.
        """
        if device_id not in self.device_seq:
            return list()
        return self.device_seq[device_id]

    def set(self, device_id: int, seq: List[ScheduleUnit]):
        """
        Set device sequence
        """
        if not all([isinstance(su, ScheduleUnit) for su in seq]):
            raise TypeError("Expected a list of ScheduleUnit")
        self.device_seq[device_id] = seq

    def draw(self, spans: Optional[List[int]] = None, outfile='./execplan.png'):
        """
        Draw the execution timeline.

        Args:
            span (List[int]): 
                length equal to schedule unit num.
                Each element stands for the time span for corresponding SU

            outfile:
                the output file name
        """
        ndevice = len(self.devices())
        # timeline [ [ (start_time, end_time), ... ], ... ]
        device_timeline = [list() for _ in range(ndevice)]
        device_sus = [list() for _ in range(ndevice)]

        if spans is None:
            spans = list()
            for su in self.seq.sus():
                span = 0
                if su.stype == SUType.Forward:
                    span = 1
                elif su.stype == SUType.Backward:
                    span = 2
                elif su.stype in [SUType.P2P, SUType.Transform]:
                    span = 0.1
                else:
                    span = 0
                spans.append(span)

        for su, span_time in zip(self.seq.sequence, spans):
            device = su.device[0]

            # tight execution if no dependency
            if len(device_timeline[device]) == 0:
                start_time = 1
            else:
                start_time = device_timeline[device][-1][1]

            # check dependency
            for devid, (timeline, dev_sus) in enumerate(zip(device_timeline, device_sus)):
                if devid == device:
                    continue
                for suid, (_, end_time) in enumerate(timeline[::-1]):
                    other_su = dev_sus[::-1][suid]
                    if other_su.happen_before(su):
                        start_time = max(start_time, end_time)
                        break

            device_timeline[device].append((start_time, start_time + span_time))
            device_sus[device].append(su)

        # draw the timeline
        if outfile is not None:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            plt.rcParams['figure.figsize'] = (12.0, 4.0)

            max_time = max(
                [tline[-1][1] for tline in device_timeline if len(tline) != 0]
            )

            fig, ax = plt.subplots()
            ax.set_xlim((1, max_time))
            plt.xticks(list(range(1, max_time+1, 1)))
            ax.xaxis.grid(True, linestyle='--')
            plt.xlabel('time')

            # yaxis
            ax.set_ylim((0.5, self.ndevice+0.5))
            plt.yticks(list(range(1, self.ndevice+1, 1)))
            ax.invert_yaxis()
            plt.ylabel('device id')

            ax.set_aspect('equal')

            for devid in range(ndevice):
                timeline = device_timeline[devid]
                sus = device_sus[devid]
                for su, (start, end) in zip(sus, timeline):
                    # draw 
                    color = 'blue' if (end - start) == 1 else 'orange'
                    rec = Rectangle((start, devid + 0.5), end-start, 1,
                                             color=color, ec='black', lw=1.5)
                    ax.add_artist(rec)
                    rx, ry = rec.get_xy()
                    cx = rx + rec.get_width() / 2.0
                    cy = ry + rec.get_height() / 2.0
                    anno = str(su.stype)
                    # anno = su.name if action.fid is None else action.fid
                    ax.annotate(anno, (cx, cy), color='w', weight='bold',
                                fontsize=10, ha='center', va='center')
            # plt.grid()
            plt.savefig(outfile)


    def __repr__(self):
        dscp = f'Execution Plan ({self.sugraph.name}):\n'
        for devid in self.devices():
            dscp += f'====> Device {devid}:\n'
            for su in self.sequence(devid):
                dscp += f'{su}\n'
        return dscp
