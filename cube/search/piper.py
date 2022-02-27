"""
Abstraction layer for microb-batch execution plan merge.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


class MicroPlan:

    def __init__(self, plan: np.ndarray, name: str = None, summation=None):
        """
        positions:
            List of [spatial, temporal] slots to anchor the action
        """
        assert len(plan.shape) == 2
        self.name = name
        self.plan = plan
        self.summation = [self] if summation is None else summation

    @property
    def ndevs(self):
        return self.plan.shape[0]

    @property
    def nsteps(self):
        return self.plan.shape[1]

    def valid(self) -> bool:
        """
        Check runnability
        """
        return np.max(self.plan) <= 1

    def __add__(self, other):
        if not isinstance(other, MicroPlan):
            raise TypeError("Expect MicroPlan")
        lhs, rhs = self, other
        ndevs = max(lhs.ndevs, rhs.ndevs)
        nsteps = max(lhs.nsteps, rhs.nsteps)
        lhs_plan = np.pad(
            lhs.plan, ((0, ndevs-lhs.ndevs),(0, nsteps-lhs.nsteps))
        )
        rhs_plan = np.pad(
            rhs.plan, ((0, ndevs-rhs.ndevs), (0, nsteps-rhs.nsteps))
        )
        plan = lhs_plan + rhs_plan
        if np.max(plan) <= 1:
            return (True, MicroPlan(plan, summation=lhs.summation+rhs.summation))
        else:
            # find conflict
            sidx, tidx = (plan > 1).nonzero()
            return (False, (sidx, tidx))
    
    def shift(self, position: Tuple[int, int], distance: int) -> bool:
        """
        shift the task at position to later (+) or previous (-) steps

        MicroPlan requires there is no more than one task on same temporal slot
        
        Args:
            position: tuple of (spatial_idx (row), step_idx (column))
        """
        s, t = position
        if self.plan[s][t] != 1:
            raise KeyError("No task is on this possition")
        if t + distance < 0:
            return False
        if distance == 0:
            return True
        if distance > 0:
            slots = np.zeros((self.ndevs, distance), dtype=int)
            self.plan = np.insert(self.plan, slice(t, t+distance), slots, axis=1)
            return True
        if distance < 0:
            slots = self.plan[:,t+distance:t]
            if np.max(slots) != 0:
                return False
            self.plan = np.delete(self.plan, slice(t+distance, t), axis=1)
            return True
        return False
    
    def __repr__(self):
        return repr(self.plan)


def create_microbatch(n_stage: int, n_dev: int, placement: List[int], name=None):
    plan = np.zeros((n_dev, n_stage * 2), dtype=int)
    for sid, devid in enumerate(placement):
        # forward
        plan[devid, sid] += 1
        # backward
        plan[devid, 2 * n_stage - 1 - sid] += 1
    return MicroPlan(plan, name)


def get_conflict(micros: List[MicroPlan], step: int):
    """
    Get conflicting postition at temporal step T
    """
    plans = []
    for micro in micros:
        if step >= micro.nsteps:
            plans.append(np.zeros((micro.ndevs, 1), dtype=int))
        else:
            plans.append(micro.plan[:,step:step+1])
    # [ndev, nmicros]
    plans = np.hstack(tuple(plans))
    # devid [int] -> (micro_id, step)
    conflicts = dict()
    # conflict device ids
    devids = np.where(np.sum(plans, axis=1) > 1)[0]
    for devid in devids:
        positions = plans[devid].nonzero()[0]
        positions = [(mid, step) for mid in positions]
        conflicts[devid] = positions
    return conflicts


def solve(micros: List[MicroPlan], conflicts: Dict[int, Tuple[int, int]]):
    # always address first conflicts
    print(f'solve conflicts: {conflicts}')
    devid = list(conflicts.keys())[0]
    mid, tid = conflicts[devid][0]
    print(f'select device: {devid}, micro id: {mid}, step: {tid} to solve')
    micros[mid].shift((devid, tid), 1)
    print(f'shift results: microbatch-{mid}')
    print(micros[mid])
    return (mid, devid, tid)


def search(n_microbatch: int, n_stage: int, n_dev: int):
    placement = [sid % n_dev for sid in range(n_stage)]
    micros = [create_microbatch(n_stage, n_dev, placement, name=mid) for mid in range(n_microbatch)]
    tidx = 0
    #TODO: justify: why firstly sovle early-step conflicts
    while tidx < max([micro.nsteps for micro in micros]):
        while True:
            # conflict point Dict[device_id, (mid, step_id)]
            conflicts = get_conflict(micros, step=tidx)
            if len(conflicts) > 0:
                # solve conflicts
                #TODO: justify: whom: which microbatch should apply shift
                #TODO: justify: how:  shift distance
                solve(micros, conflicts)
            else:
                tidx += 1
                break
    span = max([micro.nsteps for micro in micros])
    print(f'find plan: {span} steps')
    for mid, micro in enumerate(micros):
        print(f'microbatch-{mid}:')
        print(micro)


if __name__ == '__main__':
    num_microbatch = 4
    num_stage = 4
    num_device = 4
    search(num_microbatch, num_stage, num_device)
