"""
Abstraction layer for microb-batch execution plan merge.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from enum import Enum
import time


class Block:
    """
    Execution block for a MicroPlan
    """

    class BType(Enum):
        FW = 'forward'
        BW = 'backward'

    def __init__(self, mid: int, btype: BType):
        self.mid: int = mid
        self.type = btype
        # dependency track
        self.before: List[Block] = list()
        self.after: List[Block] = list()

    @staticmethod
    def add_dependency(before, after):
        if not (isinstance(before, Block) and isinstance(after, Block)):
            raise ValueError("Expected before and after to be Block")
        if after not in before.after:
            before.after.append(after)
        if before not in after.before:
            after.before.append(before)

    def __repr__(self):
        return f'f{self.mid}' if self.type == Block.BType.FW else f'b{self.mid}'


class PlanBase:

    def __init__(self, ndevs: int):
        self.blocks: Dict[Tuple[int, int], Block] = dict()
        self.positions: Dict[int, Tuple[int, int]] = dict()
        self.plan = np.zeros((ndevs, ndevs * 2), dtype=int)

    @property
    def ndevs(self):
        return self.plan.shape[0]

    @property
    def nsteps(self):
        return self.plan.shape[1]

    def block(self, dev: int, step: int):
        """
        Get block given a position
        """
        if (dev, step) not in self.blocks:
            return None
        return self.blocks[(dev, step)]
    
    def position(self, block: Block) -> Optional[Tuple[int, int]]:
        """
        Get (dev, step) position given a block.
        If block not in this plan, return None
        """
        if id(block) in self.positions:
            return self.positions[id(block)]
        else:
            return None

    def squeeze(self):
        """
        remove redundant steps
        """
        execflag = np.sum(self.plan, axis=1)
        for idx in range(self.nsteps):
            if execflag[-idx-1] != 0:
                break
        self.plan = self.plan[:, :-idx] if idx > 0 else self.plan

    def __repr__(self):
        namelen = 2
        dscp = ''
        for dev in range(self.ndevs):
            for step in range(self.nsteps):
                block = self.block(dev, step)
                if block is None:
                    dscp += '-' * namelen + ' '
                else:
                    # TODO: 2 replace to namelen
                    dscp += '{: <2}'.format(repr(block)) + ' '
            dscp += '\n'
        return dscp

    def visualize(self, outfile=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.ticker import AutoMinorLocator
        plt.close('all')
        fig, ax = plt.subplots(figsize=(4 * self.nsteps // self.ndevs, 4))
        renderer = fig.canvas.get_renderer()

        # xaxis
        ax.set_xlim((0, self.nsteps))
        plt.xticks(
            ticks=np.arange(0.5, self.nsteps+0.5, 1.0, dtype=float),
            labels=np.arange(1, self.nsteps+1, 1, dtype=int)
        )
        minor_locator = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator)
        ax.xaxis.grid(which='minor', linestyle='--')
        # yaxis
        ax.set_ylim((0.5, self.ndevs+0.5))
        plt.yticks(np.arange(1, self.ndevs+1, 1, dtype=int))
        ax.invert_yaxis()

        fontsize = [40]
        txts = list()
        def draw_block(block: Block, position: Tuple[int, int], fontsize):
            color = '#4472C4' if block.type == Block.BType.FW else '#ED7D31'
            dev, step = position
            rec = Rectangle((step, dev+0.5), 1, 1, color=color, ec='black', lw=1.5)
            ax.add_artist(rec)
            rx, ry = rec.get_xy()
            cx = rx + rec.get_width() / 2.0
            cy = ry + rec.get_height() / 2.0
            anno = str(block.mid)
            txt = ax.text(x=cx, y=cy, s=anno, fontsize=40, ha='center', va='center', color='w')
            rbox = rec.get_window_extent(renderer)
            for fs in range(fontsize[0], 1, -2):
                txt.set_fontsize(fs)
                tbox = txt.get_window_extent(renderer)
                if tbox.x0 > rbox.x0 and tbox.x1 < rbox.x1 and tbox.y0 > rbox.y0 and tbox.y1 < rbox.y1:
                    break
            fontsize[0] = min(fontsize[0], fs)
            txts.append(txt)
        
        for dev in range(self.ndevs):
            for step in range(self.nsteps):
                block = self.block(dev, step)
                if block is not None:
                    draw_block(block, self.position(block), fontsize)
        # set fontsize to same
        fontsize = fontsize[0]
        for txt in txts:
            txt.set_fontsize(fontsize)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        plt.xlabel('Time Step', fontsize=fontsize)
        plt.ylabel('Device', fontsize=fontsize)
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()


class MicroPlan(PlanBase):

    def __init__(self, mid: int, ndevs: int):
        """
        Create an empty microbatch execution plan

        mid: microbatch id
        ndevs: number of devices
        """
        super().__init__(ndevs)
        self.mid = mid

    def expand_to(self, nsteps: int):
        if self.nsteps < nsteps:
            extend = nsteps - self.nsteps
            self.plan = np.pad(self.plan, ((0,0),(0,extend)))

    def add_block(self, pos: Tuple[int, int], btype: Block.BType) -> Block:
        """
        Add a execution block
        """
        dev, step = pos
        if dev >= self.ndevs:
            raise ValueError("device out of scope")
        if step >= self.nsteps:
            self.expand_to(step + 1)
        if self.plan[dev, step] != 0:
            raise ValueError(f"Postition {pos} already has blocks")
        block = Block(self.mid, btype)
        self.plan[dev, step] += 1
        self.blocks[(dev, step)] = block
        self.positions[id(block)] = (dev, step)
        return block

    def add_dependency(self, blocks: List[Block]):
        """
        Add dependent blocks:
        block[0] < block[1] < block[2] < ...
        """
        for idx in range(len(blocks) - 1):
            Block.add_dependency(blocks[idx], blocks[idx+1])

    def copy(self):
        """
        copy a micro plan
        """
        micro = MicroPlan(self.mid, self.ndevs)
        micro.plan = np.array(self.plan, copy=True)
        micro.blocks.update(self.blocks)
        micro.positions.update(self.positions)
        return micro

    def shift(self, block: Block, inplace=True):
        """
        The primitive: shift a block by pushing one step later
        """
        micro = self if inplace else self.copy()
        # check block in this plan
        if block not in micro.blocks.values():
            raise ValueError("Block not in this micro plan")
        dev, step = micro.position(block)
        for after_block in block.after:
            if step + 1 == micro.position(after_block)[1]:
                micro.shift(after_block, inplace=True)
        micro.plan[dev, step] = 0
        if step + 1 >= micro.nsteps:
            micro.expand_to(micro.nsteps + 1)
        micro.plan[dev, step+1] = 1
        # update blocks and positions
        del micro.blocks[(dev, step)]
        micro.blocks[(dev, step+1)] = block
        micro.positions[id(block)] = (dev, step+1)
        return micro

    def unshift(self, block: Block, inplace=True):
        """
        reverse shift, for search only
        """
        micro = self if inplace else self.copy()
        dev, step = micro.position(block)
        if step == 0:
            raise ValueError("unshift a block with step = 0")
        # shift back
        micro.plan[dev, step] = 0
        micro.plan[dev, step-1] = 1
        del micro.blocks[(dev, step)]
        micro.blocks[(dev, step-1)] = 1
        micro.positions[id(block)] = (dev, step-1)
        # shift back shifted blocks
        for after_block in block.after:
            if step + 1 == micro.position(after_block)[1]:
                micro.unshift(after_block, inplace=True)
        return micro


class SchedulePlan(PlanBase):

    def __init__(self, micros: List[MicroPlan]):
        ndevs = [micro.ndevs for micro in micros]
        if len(set(ndevs)) != 1:
            raise ValueError(f"Device number not same: {ndevs}")
        ndevs = ndevs[0]

        super().__init__(ndevs)
        self.micros = micros

        # get schedule plans
        max_steps = max(micro.nsteps for micro in micros)
        for micro in micros:
            micro.expand_to(max_steps)
        plans = tuple(micro.plan for micro in micros)
        schedule = np.sum(np.stack(plans, axis=-1), axis=-1)
        if len(np.where(schedule > 1)[0]) > 0:
            raise ValueError("micro plans are not composable")
        self.plan = schedule
        self.squeeze()

        # set blocks and positions
        for micro in micros:
            self.blocks.update(micro.blocks)
            self.positions.update(micro.positions)

    @staticmethod
    def composable(micros: List[MicroPlan]) -> bool:
        max_steps = max(micro.nsteps for micro in micros)
        for micro in micros:
            micro.expand_to(max_steps)
        plans = tuple(micro.plan for micro in micros)
        schedule = np.sum(np.stack(plans, axis=-1), axis=-1)
        devids = np.where(schedule > 1)[0]
        return len(devids) == 0        

    @staticmethod
    def conflict(micros: List[MicroPlan], step: int) -> Dict[int, List[Tuple[MicroPlan, Block]]]:
        """
        Get conflict blocks at `step`.
        Return the conflicted (MicroPlan, Block) grouped by device id
        """
        max_steps = max(micro.nsteps for micro in micros)
        for micro in micros:
            micro.expand_to(max_steps)
        plans = tuple(micro.plan[:,step] for micro in micros)
        schedule = np.sum(np.stack(plans, axis=-1), axis=-1)
        conflicts = dict()
        devids = np.where(schedule > 1)[0]
        for dev in devids:
            conflicts[dev] = []
            for micro in micros:
                cblock = micro.block(dev, step)
                if cblock is not None:
                    conflicts[dev].append((micro, cblock))
        return conflicts


class Composer:

    @staticmethod
    def premise(fn, ndevs: int, nmicros: int):
        micros = fn(ndevs, nmicros)
        return micros


    @staticmethod
    def bfs_schedule(micros: List[MicroPlan]):
        micros.sort(key=lambda m: m.mid)
        step = 0
        prev: List[List[MicroPlan]] = [micros]
        next: List[List[MicroPlan]] = []
        output: List[List[MicroPlan]] = []
        while True:
            find = False
            print(f'solving step {step}, candidates {len(prev)}...')
            for micros in prev:
                # get and solve conflicts
                conflicts = SchedulePlan.conflict(micros, step)
                # input(f'conflicts: dev: {list(conflicts.keys())}, mids: {[[conf[0].mid for conf in c] for c in conflicts.values()]} | >>>')
                search_devs = []
                # direct shift on symetrics
                for dev, microblocks in conflicts.items():
                    cmicros = [micro for (micro, _) in microblocks]
                    cblocks = [block for (_, block) in microblocks]
                    if Composer.same_plans(cmicros, start_step=step):
                        for cmicro, cblock in zip(cmicros[1:], cblocks[1:]):
                            # print(f'shift(micro{cmicro.mid}, block<{cmicro.position(cblock)}>)')
                            cmicro.shift(cblock, inplace=True)
                    else:
                        search_devs.append(dev)
                if len(search_devs) == 0:
                    micros = [micro.copy() for micro in micros]
                    next.append(micros)
                    if SchedulePlan.composable(micros):
                        output.append(micros)
                        find = True
                # search space using different shift choice
                else:
                    slots = [[micro.mid for (micro, _) in conflicts[dev]] for dev in search_devs]
                    # input(f'search devs: {search_devs}, slots: {slots} | >>>')
                    for keep_mids in Composer.otho_iter(slots):
                        shifted_micros = [micro.copy() for micro in micros]
                        shift_mids = [
                            [mid for mid in slot if mid != kmid] for kmid, slot in zip(keep_mids, slots)
                        ]
                        for dev, mids in zip(search_devs, shift_mids):
                            for mid in mids:
                                block = micros[mid].block(dev, step)
                                # print(f'shift(micro{mid}, block<{(dev, step)}>)')
                                shifted_micros[mid] = micros[mid].shift(block, inplace=False)
                        next.append(shifted_micros)
                        if SchedulePlan.composable(shifted_micros):
                            output.append(shifted_micros)
                            shifted_micros=None
                            find = True
            prev, next = next, []
            if find:
                prev = output
                break
            step += 1
        schedules = [SchedulePlan(micros) for micros in prev]
        return schedules


    @staticmethod
    def same_plans(micros: List[MicroPlan], start_step: int = 0) -> bool:
        Composer.to_same_step(micros)
        plans = [micro.plan[:,start_step:] for micro in micros]
        plan = plans[0]
        for other in plans[1:]:
            if not np.array_equal(plan, other):
                return False
        return True
    
    @staticmethod
    def to_same_step(micros: List[MicroPlan]):
        """
        extend micros to same step
        """
        nsteps = max(micro.nsteps for micro in micros)
        for micro in micros:
            micro.expand_to(nsteps)
        return micros

    @staticmethod
    def otho_iter(slots: List[List[Any]]):
        """
        othogonal pickers

        item for each slot can be randomly selected
        """
        if len(slots) == 0:
            yield []
            return
        slot = slots[0]
        if len(slots) == 1:
            for item in slot:
                yield [item]
        else:
            slots = slots[1:]
            for item in slot:
                for res in Composer.otho_iter(slots):
                    yield [item] + res
        return


if __name__ == '__main__':
    
    def uniform_staging(ndevs: int, nmicros=4):
        micros = []
        for mid in range(nmicros):
            micro = MicroPlan(mid, ndevs)
            fblocks = [micro.add_block((sid, sid), Block.BType.FW) for sid in range(ndevs)]
            bblocks = [micro.add_block((ndevs-1-sid, sid+ndevs), Block.BType.BW) for sid in range(ndevs)]
            blocks = fblocks + bblocks
            micro.add_dependency(blocks)
            micros.append(micro)
        return  micros
    
    def compose_1F1B(ndevs, nmicros):
        # premise
        micros = uniform_staging(ndevs, nmicros)
        print('premise micros:')
        for micro in micros:
            print(micro)
        # shift
        for mid, micro in enumerate(micros):
            block = micro.block(0, 0)
            for _ in range(2 * mid):
                micro.shift(block)
        print('shifted micros:')
        for micro in micros:
            print(micro)
        assert SchedulePlan.composable(micros)
        schedule = SchedulePlan(micros)
        print(f'schedule (step={schedule.nsteps}):')
        print(schedule)
        return schedule

    def search(ndevs, nmicros):
        # premise
        micros = Composer.premise(uniform_staging, ndevs, nmicros)
        
        # search shift
        tic = time.time()
        schedules = Composer.bfs_schedule(micros)
        toc = time.time()
        print('search done. time {:.2f}s'.format(toc - tic))

        
        steps = set(schedule.nsteps for schedule in schedules)
        assert len(steps) == 1, f"got un-consistent step set: {steps}"
        nsteps = list(steps)[0]
        print(f'find {len(schedules)} step-optimal schedules of step: {nsteps}')
        for idx, schedule in enumerate(schedules):
            print(f'Schedule #{idx+1}:')
            print(schedule)
            

    ndevs = 4
    nmicros = 4

    # schedule = compose_1F1B(ndevs, nmicros)
    # schedule.visualize('out.png')
    search(ndevs, nmicros)
