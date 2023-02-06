"""
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    tests/adapter/test_intra_rvd.py
"""

from typing import List, Tuple
import cube
from cube.ir.tensor import IRFullTensor
from cube.graph.gener.rvd.layout import RVDLayout, RVDInspector
from cube.graph.gener.rvd.intra import IntraPathFinder, IntraAutoPlacer, IntraTransition
import numpy as np

from cube.graph.gener.utils import tensor_vd_repr


cube.init()


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


def test_intra_transition():

    fshape = [256, 256]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    src = (1, 2, 1, 4)
    dst = (1, 1, 1, 8)
    
    devs = list(range(8))
    src_rvd = RVDLayout.grid(ftensor, r=src[0], v=src[1], dims=src[2:], devices=devs)

    rets = IntraTransition.transition(src, dst, src_rvd, True)
    for idx, (layout, prims) in enumerate(rets):
        RVDInspector.draw(src_rvd, layout, f'rvd-trans-{idx}.png')



def test_transition_space():

    fshape = [256, 256]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    src = (1, 2, 1, 4)
    dst = (1, 1, 1, 8)
    devs = list(range(8))

    choices = IntraPathFinder.get_device_space(ftensor, [src, dst], src_placement=devs)
    print('choices:', choices)

    reverse_choices = IntraPathFinder.get_device_space(ftensor, [src, dst], dst_placement=devs)
    print('reverse_choices:', reverse_choices)

    # draw reverse output
    for idx, choice in enumerate(choices):
        src_rvd = RVDLayout.grid(ftensor, r=src[0], v=src[1], dims=src[2:], devices=devs)
        dst_rvd = RVDLayout.grid(ftensor, r=dst[0], v=dst[1], dims=dst[2:], devices=choice)
        RVDInspector.draw(src_rvd, dst_rvd, f'rvd-{idx}.png')

    # draw reverse output
    for idx, choice in enumerate(reverse_choices):
        src_rvd = RVDLayout.grid(ftensor, r=src[0], v=src[1], dims=src[2:], devices=choice)
        dst_rvd = RVDLayout.grid(ftensor, r=dst[0], v=dst[1], dims=dst[2:], devices=devs)
        RVDInspector.draw(src_rvd, dst_rvd, f'rvd-reverse-{idx}.png')


def test_one_f_case():

    fshape = [128, 256, 512]

    src_r, src_v, src_d = 1,4,(1,1,2)
    dst_r, dst_v, dst_d = 2,1,(2,1,2)
    src_rvd = (src_r, src_v) + src_d
    dst_rvd = (dst_r, dst_v) + dst_d
    ndevs = src_r * src_v * np.prod(np.array(src_d))
    
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    pdevs = list(range(ndevs))
    fp_rvd = RVDLayout.grid(ftensor, r=src_r, v=src_v, dims=src_d, devices=pdevs)

    cdevs = list(range(ndevs))
    fc_rvd = RVDLayout.grid(ftensor, r=dst_r, v=dst_v, dims=dst_d, devices=cdevs)

    rvds = IntraPathFinder.get_optimal_path(ftensor, src_rvd, dst_rvd)
    print(f"optimal path: {' -> '.join(str(rvd) for rvd in rvds)}")

    fprims = IntraPathFinder.path(fp_rvd, fc_rvd)
    for prim in fprims:
        print(prim)


def test_all_f_cases_fix_placement():

    fshape = [128, 256, 512]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    ndevs = 8
    ndims = len(fshape) + 2
    for src_rvd in factors(ndevs, ndims):
        for dst_rvd in factors(ndevs, ndims):
            if src_rvd == dst_rvd or src_rvd[1] < dst_rvd[1]: continue
            print(f'test generating | source rvd: {src_rvd}, destination rvd: {dst_rvd}')
            pdevs = list(range(ndevs))
            fp_rvd = RVDLayout.grid(ftensor, r=src_rvd[0], v=src_rvd[1], dims=src_rvd[2:], devices=pdevs)
            fptensors = fp_rvd.subtensors

            cdevs = list(range(ndevs))
            fc_rvd = RVDLayout.grid(ftensor, r=dst_rvd[0], v=dst_rvd[1], dims=dst_rvd[2:],devices=cdevs)
            fctensors = fc_rvd.subtensors

            fprims = IntraPathFinder.path(fp_rvd, fc_rvd)


def test_all_f_cases_auto_placement():

    fshape = [128, 256, 512]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    ndevs = 8
    ndims = len(fshape) + 2
    for src_rvd in factors(ndevs, ndims):
        for dst_rvd in factors(ndevs, ndims):
            if src_rvd == dst_rvd or src_rvd[1] < dst_rvd[1]: continue
            print(f'test generating | source rvd: {src_rvd}, destination rvd: {dst_rvd}')
            pdevs = list(range(ndevs))
            fp_rvd = RVDLayout.grid(ftensor, r=src_rvd[0], v=src_rvd[1], dims=src_rvd[2:], devices=pdevs)

            placement, cost = IntraAutoPlacer.auto_place(
                ftensor.shape,
                src_rvd, dst_rvd, None, None,
                src_placement=pdevs
            )
            fc_rvd = RVDLayout.grid(ftensor, r=dst_rvd[0], v=dst_rvd[1], dims=dst_rvd[2:],devices=placement)

            fprims = IntraPathFinder.path(fp_rvd, fc_rvd)
            print(f'cost: {cost}')


def test_one_fb_case():

    fshape = [128, 256, 512]

    # forward
    fsrc_r, fsrc_v, fsrc_d = 2,2,(1,1,2)
    fdst_r, fdst_v, fdst_d = 2,1,(1,1,4)
    bsrc_r, bsrc_v, bsrc_d = 1,2,(1,1,4)
    bdst_r, bdst_v, bdst_d = 4,1,(1,1,2)
    ndevs = fsrc_r * fsrc_v * np.prod(np.array(fsrc_d))

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=True)
    btensor: IRFullTensor = ftensor.grad

    # forward producer / backward consumer
    fpdevs = list(range(ndevs))
    fp_rvd = RVDLayout.grid(ftensor, r=fsrc_r, v=fsrc_v, dims=fsrc_d, devices=fpdevs)
    # print('forward producer tensor:')
    # for t in fp_rvd.mat.flatten():
    #     print('\t'+tensor_vd_repr(t))
    bc_rvd = RVDLayout.grid(btensor, r=bdst_r, v=bdst_v, dims=bdst_d, devices=fpdevs)

    # forward consumer / backward producer
    fcdevs, _ = IntraAutoPlacer.auto_place(
        fshape, (fsrc_r, fsrc_v) + fsrc_d, (fdst_r, fdst_v) + fdst_d,
        (bsrc_r, bsrc_v) + bsrc_d, (bdst_r, bdst_v) + bdst_d, fpdevs)
    
    fc_rvd = RVDLayout.grid(ftensor, r=fdst_r, v=fdst_v, dims=fdst_d, devices=fcdevs)
    # print('forward consumer tensor:')
    # for t in fc_rvd.mat.flatten():
    #     print('\t'+tensor_vd_repr(t))
    bp_rvd = RVDLayout.grid(btensor, r=bsrc_r, v=bsrc_v, dims=bsrc_d, devices=fcdevs)

    fprims = IntraPathFinder.path(fp_rvd, fc_rvd)
    bprims = IntraPathFinder.path(bp_rvd, bc_rvd)

    print('forward prims:')
    for prim in fprims:
        print('\t', prim)
    print('backward prims:')
    for prim in bprims:
        print('\t', prim)


def test_all_fb_cases_fix_placement():

    fshape = [128, 256, 512]
    ndevs = 8

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=True)
    btensor: IRFullTensor = ftensor.grad

    ndims = len(fshape) + 2
    for fp_rvd in factors(ndevs, ndims):

        fdevs = list(range(ndevs))
        fp = RVDLayout.grid(ftensor, r=fp_rvd[0], v=fp_rvd[1], dims=fp_rvd[2:], devices=fdevs)
        
        for fc_rvd in factors(ndevs, ndims):
            if fc_rvd[1] != 1: continue
            fc = RVDLayout.grid(ftensor, r=fc_rvd[0], v=fc_rvd[1], dims=fc_rvd[2:], devices=fdevs)
            
            # case1: forward replica -> backward replica
            bp_rvd = fc_rvd
            bc_rvd = (fp_rvd[0] * fp_rvd[1], 1) + fp_rvd[2:]
            print(f'test generating | fp rvd: {fp_rvd}, fc rvd: {fc_rvd}, bp rvd: {bp_rvd}, bc rvd: {bc_rvd}')
            
            bp = RVDLayout.grid(btensor, r=bp_rvd[0], v=bp_rvd[1], dims=bp_rvd[2:], devices=fdevs)
            bc = RVDLayout.grid(btensor, r=bc_rvd[0], v=bc_rvd[1], dims=bc_rvd[2:], devices=fdevs)

            fprims = IntraPathFinder.path(fp, fc)
            bprims = IntraPathFinder.path(bp, bc)

            # case2: forward replica -> backward accum
            bp_rvd = (1, fc_rvd[0] * fc_rvd[1]) + fc_rvd[2:]
            bc_rvd = (fp_rvd[0] * fp_rvd[1], 1) + fp_rvd[2:]
            print(f'test generating | fp rvd: {fp_rvd}, fc rvd: {fc_rvd}, bp rvd: {bp_rvd}, bc rvd: {bc_rvd}')

            bp = RVDLayout.grid(btensor, r=bp_rvd[0], v=bp_rvd[1], dims=bp_rvd[2:], devices=fdevs)
            bc = RVDLayout.grid(btensor, r=bc_rvd[0], v=bc_rvd[1], dims=bc_rvd[2:], devices=fdevs)

            fprims = IntraPathFinder.path(fp, fc)
            bprims = IntraPathFinder.path(bp, bc)


def test_all_fb_cases_advisor():

    fshape = [128, 256, 512]
    ndevs = 8

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=True)
    btensor: IRFullTensor = ftensor.grad

    ndims = len(fshape) + 2
    for fp_rvd in factors(ndevs, ndims):

        fdevs = list(range(ndevs))
        fp = RVDLayout.grid(ftensor, r=fp_rvd[0], v=fp_rvd[1], dims=fp_rvd[2:], devices=fdevs)
        
        for fc_rvd in factors(ndevs, ndims):
            if fc_rvd[1] != 1: continue
            
            # case1: forward replica -> backward replica
            bp_rvd = fc_rvd
            bc_rvd = (fp_rvd[0] * fp_rvd[1], 1) + fp_rvd[2:]
            print(f'test generating | fp rvd: {fp_rvd}, fc rvd: {fc_rvd}, bp rvd: {bp_rvd}, bc rvd: {bc_rvd}')
            
            placement, cost = IntraAutoPlacer.advice(
                fshape, fp_rvd, fc_rvd, bp_rvd, bc_rvd, fdevs)

            fc = RVDLayout.grid(ftensor, r=fc_rvd[0], v=fc_rvd[1], dims=fc_rvd[2:], devices=placement)
            bp = RVDLayout.grid(btensor, r=bp_rvd[0], v=bp_rvd[1], dims=bp_rvd[2:], devices=placement)
            bc = RVDLayout.grid(btensor, r=bc_rvd[0], v=bc_rvd[1], dims=bc_rvd[2:], devices=fdevs)

            fprims = IntraPathFinder.path(fp, fc)
            bprims = IntraPathFinder.path(bp, bc)

            # case2: forward replica -> backward accum
            bp_rvd = (1, fc_rvd[0] * fc_rvd[1]) + fc_rvd[2:]
            bc_rvd = (fp_rvd[0] * fp_rvd[1], 1) + fp_rvd[2:]
            print(f'test generating | fp rvd: {fp_rvd}, fc rvd: {fc_rvd}, bp rvd: {bp_rvd}, bc rvd: {bc_rvd}')

            placement, cost = IntraAutoPlacer.advice(
                fshape, fp_rvd, fc_rvd, bp_rvd, bc_rvd, fdevs)

            fc = RVDLayout.grid(ftensor, r=fc_rvd[0], v=fc_rvd[1], dims=fc_rvd[2:], devices=placement)
            bp = RVDLayout.grid(btensor, r=bp_rvd[0], v=bp_rvd[1], dims=bp_rvd[2:], devices=placement)
            bc = RVDLayout.grid(btensor, r=bc_rvd[0], v=bc_rvd[1], dims=bc_rvd[2:], devices=fdevs)
    
            fprims = IntraPathFinder.path(fp, fc)
            bprims = IntraPathFinder.path(bp, bc)


if __name__ == '__main__':
    # test_intra_transition()
    # test_transition_space()
    # test_one_f_case()
    # test_all_f_cases_fix_placement()
    # test_all_f_cases_auto_placement()
    # test_one_fb_case()
    # test_all_fb_cases_fix_placement()
    test_all_fb_cases_advisor()