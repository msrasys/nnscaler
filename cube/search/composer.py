"""
Abstraction layer for microb-batch execution plan merge.
"""

from typing import Dict, List, Tuple
import numpy as np
from enum import Enum


class Block:
    """
    Execution block for a MicroPlan
    """

    class BType(Enum):
        FW = 'forward'
        BW = 'backward'

    def __init__(self, mid: int, pos: Tuple[int, int], btype: BType):
        self.mid: int = mid
        self.type = btype
        self._position = tuple(pos)
        # dependency track
        self.before: List[Block] = list()
        self.after: List[Block] = list()

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos: Tuple[int, int]):
        if len(pos) != 2:
            raise ValueError("Expected positition to be Tuple[int, int]")
        self._position = pos

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
        self.plan = np.zeros((ndevs, ndevs * 2), dtype=int)

    @property
    def ndevs(self):
        return self.plan.shape[0]

    @property
    def nsteps(self):
        return self.plan.shape[1]

    def block(self, dev: int, step: int):
        if (dev, step) not in self.blocks:
            return None
        return self.blocks[(dev, step)]

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
        def draw_block(block: Block, fontsize):
            color = '#4472C4' if block.type == Block.BType.FW else '#ED7D31'
            dev, step = block.position
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
                    draw_block(block, fontsize)
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
        block = Block(self.mid, pos, btype)
        self.plan[dev, step] += 1
        self.blocks[(dev, step)] = block
        return block

    def add_dependency(self, blocks: List[Block]):
        """
        Add dependent blocks:
        block[0] < block[1] < block[2] < ...
        """
        for idx in range(len(blocks) - 1):
            Block.add_dependency(blocks[idx], blocks[idx+1])

    def shift(self, block: Block):
        """
        The primitive during search
        """
        # check block in this plan
        if block not in self.blocks.values():
            raise ValueError("Block not in this micro plan")
        dev, step = block.position
        for after_block in block.after:
            if step + 1 == after_block.position[1]:
                self.shift(after_block)
        self.plan[dev, step] = 0
        if step + 1 >= self.nsteps:
            self.expand_to(self.nsteps + 1)
        self.plan[dev, step+1] = 1
        # update block and self.blocks
        block.position = (dev, step+1)
        del self.blocks[(dev, step)]
        self.blocks[(dev, step+1)] = block

    def unshift(self, block: Block):
        """
        reverse shift, for search only
        """
        dev, step = block.position
        if step == 0:
            raise ValueError("unshift a block with step = 0")
        # shift back
        self.plan[dev, step] = 0
        self.plan[dev, step-1] = 1
        block.position = (dev, step-1)
        del self.blocks[(dev, step)]
        self.blocks[(dev, step-1)] = 1
        # shift back shifted blocks
        for after_block in block.after:
            if step + 1 == after_block.position[1]:
                self.unshift(after_block)


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

        # set blocks
        for micro in micros:
            self.blocks.update(micro.blocks)

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
    def conflict(micros: List[MicroPlan], step: int) -> bool:
        max_steps = max(micro.nsteps for micro in micros)
        for micro in micros:
            micro.expand_to(max_steps)
        plans = tuple(micro.plan[:,step] for micro in micros)
        schedule = np.sum(np.stack(plans, axis=-1), axis=-1)
        cmicros = []
        cblocks = []
        devids, steps = np.where(schedule > 1)
        for dev, step in zip(devids, steps):
            for micro in micros:
                if micro.block[dev, step] is not None:
                    cmicros.append(micro)
                    cblocks.append(micro.block[dev, step])
        return cmicros, cblocks


class Composer:

    @staticmethod
    def premise(fn, ndevs: int):
        micros = fn(ndevs)
        return micros

    @staticmethod
    def schedule(micros, step=0):
        # DFS search
        while not SchedulePlan.composable(micros):
            cmicros, cblocks = SchedulePlan.conflict(micros, step)
            if len(cmicros) == 0:
                step += 1
            else:
                for micro, block in zip(cmicros, cblocks):
                    micro.shift(block)
                    Composer.schedule(micros, step=step)
                    micro.unshift(block)
        print(f'search a plan with step {step}')



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
            

    ndevs = 4
    nmicros = 8

    # for test
    # micros = Composer.premise(uniform_staging, ndevs)
    # for mid, micro in enumerate(micros):
    #     print(f'Microbatch #{mid}:')
    #     print(micro)

    schedule = compose_1F1B(ndevs, nmicros)
    schedule.visualize('out.png')
