"""
The Tetris.

Abstraction layer for microb-batch execution plan merge.
"""

from typing import Any, Callable, Dict, List, Tuple, Optional, Union
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
        self.blocks: Dict[Tuple[Tuple[int], int], Block] = dict()
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
    
    def position(self, block: Block) -> Optional[Tuple[Tuple[int], int]]:
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
        def draw_block(block: Block, position: Tuple[Tuple[int], int], fontsize):
            color = '#4472C4' if block.type == Block.BType.FW else '#ED7D31'
            devs, step = position
            for dev in devs:
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

    def add_block(self, pos: Tuple[Union[int, Tuple[int]], int], btype: Block.BType) -> Block:
        """
        Add a execution block.
        pos: [dev(s), step]
        """
        devs, step = pos
        if isinstance(devs, int):
            devs = (devs,)
        else:
            devs = tuple(devs)
        if max(devs) >= self.ndevs:
            raise ValueError("device out of scope")
        if step >= self.nsteps:
            self.expand_to(step + 1)
        if not all([self.plan[dev, step] == 0 for dev in devs]):
            raise ValueError(f"Postition {pos} already has blocks")
        block = Block(self.mid, btype)
        for dev in devs:
            self.plan[dev, step] += 1
            self.blocks[(dev, step)] = block
        self.positions[id(block)] = (devs, step)
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
        devs, step = micro.position(block)
        # shift later blocks
        for after_block in block.after:
            if step + 1 == micro.position(after_block)[1]:
                micro.shift(after_block, inplace=True)
        for dev in devs:
            next_block = self.block(dev, step+1)
            if next_block is not None:
                micro.shift(next_block, inplace=True)
        # shift this one
        for dev in devs:
            micro.plan[dev, step] = 0
            if step + 1 >= micro.nsteps:
                micro.expand_to(micro.nsteps + 1)
            micro.plan[dev, step+1] = 1
            # update blocks and positions
            del micro.blocks[(dev, step)]
            micro.blocks[(dev, step+1)] = block
        micro.positions[id(block)] = (devs, step+1)
        return micro

    def unshift(self, block: Block, inplace=True):
        """
        reverse shift, for search only
        """
        micro = self if inplace else self.copy()
        devs, step = micro.position(block)
        if step == 0:
            raise ValueError("unshift a block with step = 0")
        # shift back
        for dev in devs:
            micro.plan[dev, step] = 0
            micro.plan[dev, step-1] = 1
            del micro.blocks[(dev, step)]
            micro.blocks[(dev, step-1)] = 1
        micro.positions[id(block)] = (devs, step-1)
        # shift back shifted blocks
        for after_block in block.after:
            if step + 1 == micro.position(after_block)[1]:
                micro.unshift(after_block, inplace=True)
        # TODO: how can I know the independent blocks got shifted?
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
    def bfs_schedule(micros: List[MicroPlan], mem_opt=True):
        total_status = 1
        micros.sort(key=lambda m: m.mid)
        block_hash = Composer.construct_hash(micros) # False # Composer.same_plans(micros, start_step=0)
        step = 0
        opt_step = sum(micro.nsteps for micro in micros)  # initial
        prev: List[List[MicroPlan]] = [micros]
        next: List[List[MicroPlan]] = []
        schedules: List[SchedulePlan] = []
        while step < opt_step:
            print(f'solving step {step}, candidates {len(prev)}...')
            for micros in prev:
                # get and solve conflicts
                conflicts = SchedulePlan.conflict(micros, step)
                # input(f'conflicts: dev: {list(conflicts.keys())}, mids: {[[conf[0].mid for conf in c] for c in conflicts.values()]} | >>>')
                for shifts in Composer.iter_shifts(conflicts, step, prune_same_micro=True, block_hash=block_hash):
                    # print(f"step {step}: {shifts}")
                    shifted_micros = [micro.copy() for micro in micros]
                    for cblock in shifts:
                        cmid = cblock.mid
                        cmicro = shifted_micros[cmid]
                        cmicro.shift(cblock, inplace=True)
                    # print(f"solved results: ")
                    # for micro in shifted_micros:
                    #     print(f'microbatch #{micro.mid}:')
                    #     print(micro)
                    if SchedulePlan.composable(shifted_micros):
                        schedule = SchedulePlan(shifted_micros)
                        schedules.append(schedule)
                        if schedule.nsteps < opt_step:
                            print(f'find fewer steps: {schedule.nsteps}')
                        opt_step = min(opt_step, schedule.nsteps)
                    else:
                        next.append(shifted_micros)
            total_status += len(next)
            prev, next = next, []
            step += 1
        total_status += len(schedules)
        schedules = [schedule for schedule in schedules if schedule.nsteps == opt_step]
        if mem_opt:
            schedules = [SchedulePlan(Composer.memory_opt(schedule.micros)) for schedule in schedules]
        print(f'searched {total_status} status.')
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
    def construct_hash(micros: List[MicroPlan]) -> Callable:
        """
        construct a hashing function to map "same" blocks into a same integer.

        The "same" blocks refer to the same-position blocks of same micro plans.
        """
        # group same micro plans
        same_plans: List[List[MicroPlan]] = [[]]
        for micro in micros:
            for smicros in same_plans:
                if Composer.same_plans(smicros + [micro]):
                    smicros.append(micro)
                    break
            else:
                same_plans.append([micro])
        print(f'detecting {len(same_plans)} same-microplan groups: {[[plan.mid for plan in smicros] for smicros in same_plans]}')
        # for each micro plan group, group same hash functions
        gid = 0
        block2gid: Dict[int, int] = dict()
        for smicros in same_plans:
            positions: Dict[Tuple[Tuple[int], int], List[Block]] = dict()
            for micro in smicros:
                for pos, block in micro.blocks.items():
                    if pos not in positions:
                        positions[pos] = []
                    positions[pos].append(block)
            for blocks in positions.values():
                for block in blocks:
                    block2gid[id(block)] = gid
                gid += 1
        def blockhash(block: Block) -> int:
            return block2gid[id(block)]
        return blockhash

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
    def iter_shifts(conflicts: Dict[int, List[Tuple[MicroPlan, Block]]],
                    step: int,
                    prune_same_micro = True,
                    block_hash = Union[None, Callable]) -> List[Block]:
        """
        Enumerate shifted blocks to resolve conflicts on step `step`.
        """
        devs = list(conflicts.keys())
        prev_shifts: List[List[Block]] = [[],]
        next_shifts: List[List[Block]] = []
        for dev in devs:
            for shifts in prev_shifts:
                cmicros = [c[0] for c in conflicts[dev]]
                cblocks = [c[1] for c in conflicts[dev]]
                # since a same block can be on multiple devices (e.g., tensor parallel)
                # we need to remove shifted blocks if it is decided before
                for sblock in shifts:
                    if sblock in cblocks:
                        idx = cblocks.index(sblock)
                        cblocks = cblocks[:idx] + cblocks[idx+1:]
                        cmicros = cmicros[:idx] + cmicros[idx+1:]
                if len(cblocks) <= 1:
                    next_shifts.append(shifts)
                    continue

                candidates = []
                if block_hash is not None:
                    gids = [block_hash(cblock) for cblock in cblocks]
                    for gid in set(gids):
                        gblocks = [cblock for cblock, cgid in zip(cblocks, gids) if cgid == gid]
                        gmids = [gblock.mid for gblock in gblocks]
                        idx = gmids.index(min(gmids))
                        candidates.append(gblocks[idx])
                else:
                    candidates = cblocks
                
                if prune_same_micro:
                    if Composer.same_plans(cmicros, start_step=step):
                        candidates = [candidates[0]]

                for kblock in candidates:
                    idx = cblocks.index(kblock)
                    # keep blocks on the idx while shifts the rest
                    nshifts = shifts + cblocks[:idx] + cblocks[idx+1:]
                    # if the reserved block executes on multiple devices,
                    # then the rest device must shift all other blocks
                    for odev in cmicros[idx].position(kblock)[0]:
                        if odev != dev and odev in conflicts:
                            for _, ocblock in conflicts[odev]:
                                if ocblock != cblocks[idx] and ocblock not in nshifts:
                                    nshifts.append(ocblock)
                    next_shifts.append(nshifts)
            prev_shifts, next_shifts = next_shifts, []
        for shifts in prev_shifts:
            yield shifts

    @staticmethod
    def memory_opt(micros: List[MicroPlan]):
        """
        optimize memory given a schedule plan.
        The micros are composable.
        """
        nsteps = max(micro.nsteps for micro in micros)
        for step in range(nsteps-1, -1, -1):
            micros = Composer.memory_opt_step(micros, step)
        return micros

    @staticmethod
    def memory_opt_step(micros: List[MicroPlan], step: int):
        splan = sum(micro.plan for micro in micros)
        free_steps = [np.where(splan[dev,:] == 0)[0] for dev in range(micros[0].ndevs)]
        for micro in micros:
            devs = np.where(micro.plan[:,step] > 0)[0]
            fblocks = []
            # find forward blocks
            for dev in devs:
                block = micro.block(dev, step)
                if block.type != Block.BType.FW:
                    continue
                if block not in fblocks:
                    fblocks.append(block)
            # find non-critical forward blocks
            for block in fblocks:
                maxstep = min(micro.position(nblock)[1] for nblock in block.after) - 1
                if maxstep == step: # no room for shift => critical
                    continue
                # find maximal shift distance
                maxshift = None
                for t in range(maxstep, step, -1):
                    if all([t in free_steps[dev] for dev in micro.position(block)[0]]):
                        maxshift = t - step
                        break
                # apply shift by `distance` times
                if maxshift is not None:
                    for _ in range(maxshift):
                        micro.shift(block, inplace=True)
        return micros


if __name__ == '__main__':
    
    def uniform_staging(ndevs: int, nmicros=4) -> List[MicroPlan]:
        """
        f             b
          f         b  
            f     b    
              f b      
        """
        micros = []
        for mid in range(nmicros):
            micro = MicroPlan(mid, ndevs)
            fblocks = [micro.add_block((sid, sid), Block.BType.FW) for sid in range(ndevs)]
            bblocks = [micro.add_block((ndevs-1-sid, sid+ndevs), Block.BType.BW) for sid in range(ndevs)]
            blocks = fblocks + bblocks
            micro.add_dependency(blocks)
            micros.append(micro)
        return micros

    def mbart_staging(ndevs: int, nmicros=4) -> List[MicroPlan]:
        """
        f f  f         b   b b
        f  f f         b b   b
        f    f f     b b     b
        f    f   f b   b     b
        """
        micros = []
        for mid in range(nmicros):
            micro = MicroPlan(mid, ndevs)
            fblocks = []
            bblocks = []
            for step in range(ndevs+2):
                if step in [0, ndevs // 2+1]:
                    fblock = micro.add_block((tuple(range(ndevs)), step), Block.BType.FW)
                    bblock = micro.add_block((tuple(range(ndevs)), (ndevs+2)*2-1-step), Block.BType.BW)
                else:
                    dev = step - 1 if step < ndevs//2+1 else step - 2
                    fblock = micro.add_block((dev, step), Block.BType.FW)
                    bblock = micro.add_block((dev, (ndevs+2)*2-1-step), Block.BType.BW)
                fblocks.append(fblock)
                bblocks.append(bblock)
            micro.add_dependency(fblocks+bblocks[::-1])
            micros.append(micro)
        return micros

    def chimera_staging(ndevs: int, nmicros: int) -> List[MicroPlan]:
        """
        f             b        f b
          f         b        f     b
            f     b        f         b
              f b        f             b
        """
        micros = []
        assert nmicros % 2 == 0, "require microbatch# can be divided by 2."
        for mid in range(nmicros // 2):  # V shape
            micro = MicroPlan(mid, ndevs)
            fblocks = [micro.add_block((sid, sid), Block.BType.FW) for sid in range(ndevs)]
            bblocks = [micro.add_block((ndevs-1-sid, sid+ndevs), Block.BType.BW) for sid in range(ndevs)]
            blocks = fblocks + bblocks
            micro.add_dependency(blocks)
            micros.append(micro)
        for mid in range(nmicros // 2): # ^ shape
            micro = MicroPlan(mid + nmicros // 2, ndevs)
            fblocks = [micro.add_block((ndevs-1-sid, sid), Block.BType.FW) for sid in range(ndevs)]
            bblocks = [micro.add_block((sid, sid+ndevs), Block.BType.BW) for sid in range(ndevs)]
            blocks = fblocks + bblocks
            micro.add_dependency(blocks)
            micros.append(micro)
        return micros
    
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

    def search(ndevs, nmicros, visualize=False):
        # premise
        # micros = Composer.premise(uniform_staging, ndevs, nmicros)
        # micros = Composer.premise(chimera_staging, ndevs, nmicros)
        micros = Composer.premise(uniform_staging, ndevs, nmicros)
        print('============== Premise ================')
        for idx, micro in enumerate(micros):
            print(f'microbatch #{idx}:')
            print(micro)
            if visualize:
                micro.visualize(f'planlog/micro{idx}.png')
        print('============== Premise ================')
        
        # search shift
        tic = time.time()
        schedules = Composer.bfs_schedule(micros, mem_opt=True)
        toc = time.time()
        print('search done. time {:.2f}s'.format(toc - tic))

        
        steps = set(schedule.nsteps for schedule in schedules)
        assert len(steps) == 1, f"got un-consistent step set: {steps}"
        nsteps = list(steps)[0]
        print(f'find {len(schedules)} step-optimal plans (step={nsteps})')
        for idx, schedule in enumerate(schedules):
            print(f'Schedule #{idx+1}:')
            print(schedule)
            if visualize:
                schedule.visualize(f'planlog/plan{idx+1}.png')
            

    ndevs = 4
    nmicros = 4

    # schedule = compose_1F1B(ndevs, nmicros)
    # schedule.visualize('out.png')
    search(ndevs, nmicros, visualize=False)

    # micros = mbart_staging(ndevs, nmicros)
    # for idx, micro in enumerate(micros):
    #     print(f'microbatch #{idx}:')
    #     print(micro)
    # 
    # micros[0].shift(micros[0].block(0, 0))
    # micros[0].shift(micros[0].block(0, 2))
    # micros[0].shift(micros[0].block(0, 5))
    # print(micros[0])