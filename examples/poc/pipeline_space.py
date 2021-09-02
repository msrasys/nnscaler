from cube.schedule.action import Action, add_flow
from cube.schedule.iterator import sequence_space, sequence_space_batched, placement_space
from cube.schedule.plan import ExecutionPlan
from cube.schedule.checker import correct_check

import argparse
import re
import json
import time
import os
import multiprocessing as mp
from functools import partial


def get_semantic(forward_fn, backward_fn, num_stage, num_microbatch):
    forward_time = 1
    backward_time = 2

    actions = list()
    relations = list()
    for mid in range(num_microbatch):
        # forward
        for stage in range(num_stage):
            action = Action(forward_fn)
            action.est_latency = forward_time
            action.est_memory = 1
            action.tag('fS{}D{}'.format(stage, mid))
            if stage != 0:
                relation = (actions[-1], action)
                add_flow(actions[-1], action)
                relations.append(relation)
            else:
                action.fid = mid
            actions.append(action)
        # backward
        for stage in range(num_stage):
            action = Action(backward_fn)
            action.est_latency = backward_time
            action.est_memory = -1
            action.tag('bS{}D{}'.format(num_stage - 1 - stage, mid))
            # relation
            relation = (actions[-1], action)
            add_flow(actions[-1], action)
            # append to relation sets
            relations.append(relation)
            actions.append(action)
    return actions, relations


def get_stage_and_mid(action):
    ids = re.findall(r"S(\d+)D(\d+)", action.name)
    stage, mid = int(ids[0][0]), int(ids[0][1])
    return stage, mid


def full_grid_search(actions, relations, ndevice, nmb, outpath='./figs'):
    """
    Search minimal time plan under the memory constraints
    """

    memory_buckets = dict()
    for activation_num in range(1, nmb+1):
        memory_buckets[activation_num] = None

    tic = time.time()
    for cnt, seq in enumerate(sequence_space(actions, relations)):
        for dev_num, dev_seq in enumerate(placement_space(seq, ndevice, fb_same=True)):
            # print(f'on sequence > {dev_seq}')
            execplan = ExecutionPlan(dev_seq, ndevice)
            execplan.gen()
            span = execplan.get_time()
            memory = execplan.get_memory()
            # update plan
            for upper_mem in memory_buckets:
                if memory <= upper_mem:
                    if memory_buckets[upper_mem] is None:
                        memory_buckets[upper_mem] = execplan
                        execplan.draw(outfile=os.path.join(outpath, f'{ndevice}nmb{nmb}dev.mem{memory}.png'))
                    if span < memory_buckets[upper_mem].get_time():
                        memory_buckets[upper_mem] = execplan
                        execplan.draw(outfile=os.path.join(outpath, f'{ndevice}nmb{nmb}dev.mem{memory}.png'))
                        print(f'> found a better seq {seq} time {span} mem {memory}')
        # input(f'>>> done on {dev_num+1} device placement ')
        if (cnt+1) % 1000 == 0:
            throughput = 1000 * (nmb ** ndevice) / (time.time() - tic)
            tic = time.time()
            print('> search [{}-{}] throughput {:.2f} spatial sequences / sec'.format(cnt+1-1000, cnt+1, throughput))
    # dump to json
    print(f'> totally done search on {cnt+1} sequences')
    for key in memory_buckets:
        memory_buckets[key] = memory_buckets[key].to_json()
    with open(os.path.join(outpath, 'results.json'), 'w') as outfile:
        json.dump(memory_buckets, outfile)


def worker_search(seqs, nmb, ndevice):
    sub_memory_buckets = dict()
    for activation_num in range(1, nmb+1):
        sub_memory_buckets[activation_num] = None
    for seq in seqs:
        for dev_seq in placement_space(seq, ndevice, fb_same=True):
            execplan = ExecutionPlan(dev_seq, ndevice)
            execplan.gen()
            span = execplan.get_time()
            memory = execplan.get_memory()
            # update plan
            for upper_mem in sub_memory_buckets:
                if memory <= upper_mem:
                    if sub_memory_buckets[upper_mem] is None:
                        sub_memory_buckets[upper_mem] = execplan
                    if span < sub_memory_buckets[upper_mem].get_time():
                        sub_memory_buckets[upper_mem] = execplan
    return sub_memory_buckets


def full_grid_search_mp(actions, relations, ndevice, nmb, outpath='./figs', nworker=40):
    """
    Search minimal time plan under the memory constraints
    """
    pool = mp.Pool(processes=nworker)

    memory_buckets = dict()
    for activation_num in range(1, nmb+1):
        memory_buckets[activation_num] = None
    
    def merge(sub_memory_buckets):
        for upper_mem in sub_memory_buckets:
            if sub_memory_buckets[upper_mem] is None:
                continue
            execplan = sub_memory_buckets[upper_mem]
            span = execplan.get_time()
            memory = execplan.get_memory()
            if memory_buckets[upper_mem] is None:
                memory_buckets[upper_mem] = execplan
                execplan.draw(outfile=os.path.join(outpath, f'{ndevice}nmb{nmb}dev.mem{memory}.png'))
                print(f'> found a better seq {execplan.seq} time {span} mem {memory}')
            if span < memory_buckets[upper_mem].get_time():
                memory_buckets[upper_mem] = execplan
                execplan.draw(outfile=os.path.join(outpath, f'{ndevice}nmb{nmb}dev.mem{memory}.png'))
                print(f'> found a better seq {execplan.seq} time {span} mem {memory}')

    bs = (nworker, 20)
    nseqs = 0
    for seqs in sequence_space_batched(actions, relations, bs=bs):
        results = list()
        for wid in range(nworker):
            res = pool.apply_async(worker_search, args=(seqs[wid], nmb, ndevice))
            results.append(res)
        nseqs += sum([len(worker_seqs) for worker_seqs in seqs])
        print(f'assigned {nseqs} sequences')
        for res in results:
            sub_buckets = res.get()
            merge(sub_buckets)
    
    pool.close()
    pool.join()

    # dump to json
    print(f'> totally done search on {nseqs} sequences')
    for key in memory_buckets:
        memory_buckets[key] = memory_buckets[key].to_json()
    with open(os.path.join(outpath, 'results.json'), 'w') as outfile:
        json.dump(memory_buckets, outfile)


def fixed_placement_search(actions, relations, ndevice, max_time):
    for cnt, seq in enumerate(sequence_space(actions, relations)):
        # assign device
        for action in seq:
            stage, mid = get_stage_and_mid(action)
            action.device = stage % ndevice
        execplan = ExecutionPlan(seq, ndevice)
        execplan.gen()
        iter_time = execplan.get_time()
        print(f'found seq > {seq} \t time {iter_time}')
        if iter_time > max_time:
            continue
        execplan.draw(outfile='tmp.png')
        input('>>> ')
    print('total found {} legal sequences'.format(cnt + 1))


def pipe_1f1b(actions, relations, nstage, ndevice, nmb):
    num_stage = nstage
    num_microbatch = nmb

    f = lambda stage, micro_batch_id: actions[2 * micro_batch_id * num_stage + stage]
    b = lambda stage, micro_batch_id: actions[(2 * micro_batch_id + 1) * num_stage + num_stage - 1 - stage]

    # action placement
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            f(stage, mid).device = stage % ndevice
            print(f(stage, mid), f'stage={stage}, mid={mid}, device={stage % ndevice}')
            b(stage, mid).device = stage % ndevice
            print(b(stage, mid), f'stage={stage}, mid={mid}')

    sequence = list()

    # warmup:
    for stage in range(num_stage):
        for mid in range(num_stage-stage):
            sequence.append(f(stage, mid))

    # steady + cooldown:
    for mid in range(num_microbatch):
        # enqueue backward
        for stage in range(num_stage-1, -1, -1):
            sequence.append(b(stage, mid))
        # enqueue forward
        for stage in range(num_stage):
            f_mid = mid + num_stage - stage
            if f_mid >= num_microbatch:
                continue
            sequence.append(f(stage, f_mid))
    print(sequence)
    assert correct_check(sequence, actions, relations)
    execplan = ExecutionPlan(sequence, ndevice)
    execplan.draw(outfile='./pipeline-1f1b.png')


def gpipe(actions, relations, nstage, ndevice, nmb):
    num_stage = nstage
    num_microbatch = nmb

    f = lambda stage, micro_batch_id: actions[2 * micro_batch_id * num_stage + stage]
    b = lambda stage, micro_batch_id: actions[(2 * micro_batch_id + 1) * num_stage + num_stage - 1 - stage]

    # action placement
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            f(stage, mid).device = stage % ndevice
            print(f(stage, mid), f'stage={stage}, mid={mid}, device={stage % ndevice}')
            b(stage, mid).device = stage % ndevice
            print(b(stage, mid), f'stage={stage}, mid={mid}')

    sequence = list()

    # warmup:
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            sequence.append(f(stage, mid))
    
    # backward
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            sequence.append(b(num_stage - 1 - stage, mid))

    print(sequence)
    # assert correct_check(sequence, actions, relations)
    execplan = ExecutionPlan(sequence, ndevice)
    execplan.draw(outfile='./gpipe.png')


def forward(data):
    pass

def backward(grad):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nstage', type=int, default=4,
                        help='number of stages')
    parser.add_argument('--nmb', type=int, default=4,
                        help='number of micro-batch')
    parser.add_argument('--ndev', type=int, default=4,
                        help='number of devices')
    parser.add_argument('--outpath', type=str, default='/mydata/MagicCube/search/pipeline/')
    args = parser.parse_args()

    actions, relations = get_semantic(forward, backward, args.nstage, args.nmb)

    # pipe_1f1b(actions, relations, args.nstage, args.ndev, args.nmb)
    # gpipe(actions, relations, args.nstage, args.ndev, args.nmb)

    # fixed_placement_search(actions, relations, args.ndev, max_time=100)
    full_grid_search_mp(actions, relations, args.ndev, args.nmb, args.outpath)