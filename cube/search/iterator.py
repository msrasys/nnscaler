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


if __name__ == '__main__':

    # for seq in otho_iter([[1,2,3], [4,5], [6,7,8]]):
    #     print(seq)

    for seq in factorization(8, 2):
        print(seq)