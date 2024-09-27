#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Note this is not for test.

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    tests/adapter/test_inter_rvd.py
"""

from typing import List, Tuple
import nnscaler
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph.gener.rvd.layout import RVDLayout, RVDInspector
from nnscaler.graph.gener.rvd.inter import InterPathFinder
import numpy as np

from nnscaler.graph.gener.utils import tensor_vd_repr


nnscaler.init()


def factors(k: int, num: int) -> List[Tuple[int]]:
    """
    get all possible sequence k1 * k2 * .. k_{num} = k
    """
    if num == 1: return [(k,)]
    res = []
    for i in range(1, k):
        if k % i != 0: continue
        for sub_res in factors(k // i, num - 1):
            res.append((i,) + sub_res)
    return res


def test_one_f_case():

    fshape = [128, 256, 512]

    src_r, src_v, src_d = 1,4,(1,1,2)
    dst_r, dst_v, dst_d = 2,1,(2,1,2)
    src_rvd = (src_r, src_v) + src_d
    dst_rvd = (dst_r, dst_v) + dst_d

    pndevs = np.prod(src_rvd)
    cndevs = np.prod(dst_rvd)
    
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    pdevs = list(range(pndevs))
    fp_rvd = RVDLayout.grid(ftensor, r=src_r, v=src_v, dims=src_d, devices=pdevs)

    cdevs = list(range(pndevs, pndevs + cndevs))
    fc_rvd = RVDLayout.grid(ftensor, r=dst_r, v=dst_v, dims=dst_d, devices=cdevs)

    rvds = InterPathFinder.get_optimal_path(ftensor, src_rvd, dst_rvd)
    print(f"optimal path: {' -> '.join(str(rvd) for rvd in rvds)}")

    fprims = InterPathFinder.path(fp_rvd, fc_rvd)
    for prim in fprims:
        print(prim)


def test_all_f_cases_fix_placement():

    fshape = [128, 256, 512]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    pndevs = 4
    cndevs = 8
    
    ndims = len(fshape) + 2
    for src_rvd in factors(pndevs, ndims):
        for dst_rvd in factors(cndevs, ndims):
            if src_rvd == dst_rvd or src_rvd[1] < dst_rvd[1]: continue
            print(f'test generating | source rvd: {src_rvd}, destination rvd: {dst_rvd}')
            pdevs = list(range(pndevs))
            fp_rvd = RVDLayout.grid(ftensor, r=src_rvd[0], v=src_rvd[1], dims=src_rvd[2:], devices=pdevs)

            cdevs = list(range(pndevs, pndevs + cndevs))
            fc_rvd = RVDLayout.grid(ftensor, r=dst_rvd[0], v=dst_rvd[1], dims=dst_rvd[2:],devices=cdevs)

            _ = InterPathFinder.path(fp_rvd, fc_rvd)
            rvds = InterPathFinder.get_optimal_path(ftensor, src_rvd, dst_rvd)
            print(f"==> path: {'->'.join(str(rvd) for rvd in rvds)}")


if __name__ == '__main__':

    # test_one_f_case()
    test_all_f_cases_fix_placement()