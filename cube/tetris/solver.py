"""
A solver based solution for scheduling plan
"""

from typing import List, Optional, Tuple
from enum import Enum

from z3 import *
import time
import copy



gsolver = Solver()


class Block:

    class BType(Enum):
        FW = 'forward'
        BW = 'backward'

    def __init__(self, mid: int, btype: BType, name: str, mem=1):
        global _uid
        global gsolver
        self.name = name
        self.mid = mid
        self.step = Int(name)
        self.memory = mem if btype == Block.BType.FW else 0-mem
        gsolver.add(self.step >= 1)
        self.btype = btype

    @staticmethod
    def add_dependency(blk1, blk2):
        """
        add dependency: blk1 -> blk2
        """
        global gsolver
        gsolver.add(blk1.step < blk2.step)

    def __repr__(self):
        return f'f{self.mid}' if self.btype == Block.BType.FW else f'b{self.mid}'


class SchedulePlan:

    def __init__(self, ndevs: int) -> None:
        
        self._blocks: List[List[Block]] = [[] for _ in range(ndevs)]
        self.ndevs = ndevs
        self._nsteps = None
        self._mem = None
        self._solution: Optional[z3.z3.ModelRef] = None

    @property
    def nblocks(self) -> int:
        return sum(len(blks) for blks in self._blocks)

    @property
    def nsteps(self) -> int:
        if self._solution is None:
            return -1
        return self._solution.eval(self._nsteps).as_long()

    @property
    def mem(self) -> int:
        if self._mem is None:
            return -1
        return self._solution.eval(self._mem).as_long()

    def blocks(self, devid: Optional[int] = None) -> List[Block]:
        if isinstance(devid, int):
            return copy.copy(self._blocks[devid])
        else:
            allblocks = []
            for blks in self._blocks:
                allblocks += blks
            return allblocks

    def position(self, block: Block) -> Tuple[int, int]:
        """
        get block position (device, time) after the search
        """
        # device
        for devid in range(self.ndevs):
            if block in self.blocks(devid):
                break
        else:
            assert False, 'block not in schedule plan'
        # time step
        step = None
        if self._solution is not None:
            step = self._solution[block.step]
        return (devid, step)

    def add_block(self, block: Block, devices: Tuple[int]):
        global gsolver
        devices = (devices,) if isinstance(devices, int) else devices
        for device in devices:
            for blk in self._blocks[device]:
                gsolver.add(blk.step != block.step)
            self._blocks[device].append(block)
        # set plan step variable
        if self._nsteps is None:
            self._nsteps = block.step
        else:
            self._nsteps = If(block.step > self._nsteps, block.step, self._nsteps)

    def set_memory(self):
        nblocks = max(len(blks) for blks in self._blocks)
        # mems = [IntVector(f'memdev{devid}', nblocks) for devid in range(self.ndevs)]
        peaks = []
        for devid in range(self.ndevs):
            peak = 0
            curr = 0
            for step in range(0, nblocks):
                mem = 0
                for block in self.blocks(devid):
                    mem = If(block.step == step, block.memory, mem)
                curr = mem + curr
                peak = If(curr > peak, curr, peak)
            peaks.append(peak)
        # global peak
        globalpeak = peaks[0]
        for devid in range(1, self.ndevs):
            globalpeak = If(peaks[devid] > globalpeak, peaks[devid], globalpeak)
        self._mem = globalpeak
        return globalpeak

    def set_solution(self, solution: z3.z3.ModelRef):
        self._solution = solution

    def solve(self, decrease = True):
        global gsolver
        tic = time.time()
        min_step = max(len(blks) for blks in self._blocks)
        max_step = self.nblocks
        opt_step = max_step if decrease else min_step
        while True:
            assert min_step <= opt_step and opt_step <= max_step, "out of step boundary. consider this as a bug."
            gsolver.push()
            gsolver.add(self._nsteps == opt_step)
            if gsolver.check() == sat:
                print(f'find scheduling plan in {opt_step} steps')
                solution = gsolver.model()
                self.set_solution(solution)
                gsolver.pop()
                if not decrease: break
            else:
                print(f'fail to find solution for {opt_step} steps')
                gsolver.pop()
                if decrease:
                    opt_step += 1
                    break
            opt_step = opt_step - 1 if decrease else opt_step + 1
        toc = time.time()
        print('search time: {:.2f} seconds. find optimal step: {}'.format(toc-tic, opt_step))
        print('solution:')
        print(self)

        # search for optimal memory
        tic = time.time()
        min_mem = max(min(blk.memory for blk in blks if blk.btype == Block.BType.FW) for blks in self._blocks)
        max_mem = max(sum(blk.memory for blk in blks if blk.btype == Block.BType.FW) for blks in self._blocks)
        opt_mem = max_mem if decrease else min_mem
        self.set_memory()
        gsolver.push()
        gsolver.add(self._nsteps == opt_step)
        while True:
            assert min_mem <= opt_mem and opt_mem <= max_mem, "out of memory boundary. consider this as a bug"
            gsolver.push()
            gsolver.add(self._mem == opt_mem)
            if gsolver.check() == sat:
                print(f'find scheduling plan in {opt_mem} memory')
                solution = gsolver.model()
                self.set_solution(solution)
                gsolver.pop()
                if not decrease: break
            else:
                print(f'fail to find solution for memory {opt_mem}')
                gsolver.pop()
                if decrease:
                    opt_mem += 1
                    break
            opt_mem = opt_mem - 1 if decrease else opt_mem + 1
        gsolver.pop()
        toc = time.time()
        print('search memory time: {:.2f} seconds. opt-memory: {}'.format(toc-tic, opt_mem))
        print('solution:')
        print(self)

        tic = time.time()
        self.iter_space(opt_step, opt_mem)
        toc = time.time()
        print('iterate all plans: {:.2f} seconds.'.format(toc-tic))

    def solve_mconstraints(self, memory: int, decrease=True):
        global gsolver
        tic = time.time()
        min_step = max(len(blks) for blks in self._blocks)
        max_step = self.nblocks
        opt_step = max_step if decrease else min_step

        self.set_memory()
        
        # memory constraints
        gsolver.push()
        gsolver.add(self._mem <= memory)
        # find optimal step
        while True:
            assert min_step <= opt_step and opt_step <= max_step, "out of step boundary. consider this as a bug."
            gsolver.push()
            gsolver.add(self._nsteps == opt_step)
            if gsolver.check() == sat:
                print(f'find scheduling plan in {opt_step} steps')
                solution = gsolver.model()
                self.set_solution(solution)
                gsolver.pop()
                if not decrease: break
            else:
                print(f'fail to find solution for {opt_step} steps')
                gsolver.pop()
                if decrease:
                    opt_step += 1
                    break
            opt_step = opt_step - 1 if decrease else opt_step + 1
        toc = time.time()
        print('search time: {:.2f} seconds. find optimal step: {}'.format(toc-tic, opt_step))
        print('solution:')
        print(self)

        tic = time.time()
        self.iter_space(opt_step)
        toc = time.time()
        print('iterate all plans: {:.2f} seconds.'.format(toc-tic))


    def iter_space(self, nsteps: int, memory: int = None):
        """
        iterate all solutions find by solver
        """
        global gsolver
        gsolver.push()
        gsolver.add(self._nsteps == nsteps)
        if memory is not None:
            gsolver.add(self._mem == memory)
        models = []
        while gsolver.check() == sat:
            model = gsolver.model()
            models.append(model)
            block = []
            for d in model:
                assert not d.arity() > 0, 'uniterpreted functions found'
                c = d()
                block.append(c != model[d])
            gsolver.add(Or(block))
            if len(models) % 100 == 0:
                print(f'find {len(models)} solutions..')
        gsolver.pop()
        print(f'find {len(models)} possible models')


    def __repr__(self) -> str:
        if self._solution is None:
            return 'Unsolved Schedule Plan.'
        namelen = 2
        dscp = ''
        for devid in range(self.ndevs):
            blocks = self.blocks(devid)
            steps = [self.position(blk)[1] for blk in blocks]
            for step in range(1, self.nsteps+1):
                if step not in steps:
                    dscp += '-' * namelen + ' '
                else:
                    idx = steps.index(step)
                    dscp += '{: <2}'.format(repr(blocks[idx])) + ' '
            dscp += '\n'
        return dscp


if __name__ == '__main__':

    def uniform_staging(ndevs: int, nmicros) -> SchedulePlan:
        """
        f             b
          f         b  
            f     b    
              f b      
        """
        sched = SchedulePlan(ndevs)
        for mid in range(nmicros):
            fblocks = [Block(mid, Block.BType.FW, f'f{mid}d{devid}') for devid in range(ndevs)]
            bblocks = [Block(mid, Block.BType.BW, f'b{mid}d{devid}') for devid in range(ndevs)][::-1]
            blocks = fblocks + bblocks
            for idx in range(ndevs * 2 - 1):
                Block.add_dependency(blocks[idx], blocks[idx+1])
            for devid in range(ndevs):
                sched.add_block(fblocks[devid], devid)
                sched.add_block(bblocks[ndevs-1-devid], devid)
        return sched

    def chimera_staging(ndevs: int, nmicros: int) -> SchedulePlan:
        """
        f             b        f b
          f         b        f     b
            f     b        f         b
              f b        f             b
        """
        sched = SchedulePlan(ndevs)
        assert nmicros % 2 == 0, "require microbatch# can be devided by 2"
        for mid in range(nmicros // 2): # V shape
            fblocks = [Block(mid, Block.BType.FW, f'f{mid}d{devid}', mem=1) for devid in range(ndevs)]
            bblocks = [Block(mid, Block.BType.BW, f'b{mid}d{devid}', mem=1) for devid in range(ndevs-1,-1,-1)]
            blocks = fblocks + bblocks
            for idx in range(ndevs * 2 - 1):
                Block.add_dependency(blocks[idx], blocks[idx+1])
            for devid in range(ndevs):
                sched.add_block(fblocks[devid], devid)
                sched.add_block(bblocks[ndevs-1-devid], devid)
        for mid in range(nmicros // 2): # ^ shape
            mid = mid + nmicros // 2
            fblocks = [Block(mid, Block.BType.FW, f'f{mid}d{devid}', mem=1) for devid in range(ndevs-1,-1,-1)]
            bblocks = [Block(mid, Block.BType.BW, f'b{mid}d{devid}', mem=1) for devid in range(ndevs)]
            blocks = fblocks + bblocks
            for idx in range(ndevs * 2 - 1):
                Block.add_dependency(blocks[idx], blocks[idx+1])
            for devid in range(ndevs):
                sched.add_block(fblocks[ndevs-1-devid], devid)
                sched.add_block(bblocks[devid], devid)
        return sched

    def mbart_staging(ndevs: int, nmicros: int) -> SchedulePlan:
        """
        f f   f         b   b b
        f   f f         b b   b
        f     f f     b b     b
        f     f   f b   b     b
        """
        sched = SchedulePlan(ndevs)
        for mid in range(nmicros):
            fblocks = []
            bblocks = []
            for step in range(ndevs+2):
                if step in [0, ndevs // 2 + 1]:
                    fdevid = bdevid = tuple(range(ndevs))
                    fblock = Block(mid, Block.BType.FW, f'fe{step}{mid}devall', mem=1)
                    bblock = Block(mid, Block.BType.BW, f'be{step}{mid}devall', mem=1)
                else:
                    fdevid = bdevid = step - 1 if step < ndevs // 2 + 1 else step - 2
                    fblock = Block(mid, Block.BType.FW, f'f{mid}dev{fdevid}', mem=1)
                    bblock = Block(mid, Block.BType.BW, f'b{mid}dev{bdevid}', mem=1)
                fblocks.append(fblock)
                bblocks.append(bblock)
                sched.add_block(fblock, fdevid)
                sched.add_block(bblock, bdevid)
            blocks = fblocks + bblocks[::-1]
            for idx in range((ndevs + 2) * 2 - 1):
                Block.add_dependency(blocks[idx], blocks[idx+1])
        return sched

    ndevs = 8
    nmicros = 8

    sched = uniform_staging(ndevs, nmicros)
    # sched = chimera_staging(ndevs, nmicros)
    # sched = mbart_staging(ndevs, nmicros)  # ndev=4, nmicro=4 => solution: step=30
    sched.solve_mconstraints(memory=ndevs, decrease=True)
