from itertools import combinations
from typing import Any, List


def comb_iter(candidates: List, pick_num: int):
    """
    combination pickers
    """
    return combinations(candidates, pick_num)


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
            for res in otho_iter(slots):
                yield [item] + res
    return


def factorization(K: int, num=1):
    """
    Decompose K into `depth` numbers that
    a1 * a2 * ... * a_depth = K
    ($\prod\limits_{i=1}^depth a_i = K$)

    Yield:
        List[int]
    """
    if num == 1:
        yield [K]
    else:
        for i in range(1, K+1):
            if K % i == 0:
                for res in factorization(K // i, num-1):
                    yield [i] + res

def diff_balls_diff_boxes(nballs: int, nboxes: int, remain = None, placement = None):
    balls_per_box = nballs // nboxes
    if placement is None and remain is None:
            # placement[ball_id] = box_id
            placement = []
            # remain slots: remain_slots[box_id] = int
            remain = [balls_per_box] * nboxes
    if len(placement) == nballs:
        yield placement
    for box_id, remain_balls in enumerate(remain):
        if remain_balls > 0:
            placement.append(box_id)
            remain[box_id] -= 1
            for seq in diff_balls_diff_boxes(nballs, nboxes, remain, placement):
                yield seq
            remain[box_id] += 1
            placement = placement[:-1]
    


if __name__ == '__main__':

    # for seq in otho_iter([[1,2,3], [4,5], [6,7,8]]):
    #     print(seq)

    for seq in factorization(8, 2):
        print(seq)