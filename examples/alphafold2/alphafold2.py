import torch
import math
import cube

from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank

from examples.alphafold2.model import *
import examples.alphafold2.policy.spmd as spmd

from cube.ir.operator import IRFwOperation, IRBpOperation
from cube.profiler.database import ProfileDataBase
from cube.algorithm.ops.dimops import gen_partitions
from cube.graph.function.anchor import IRGraphAnchor

def run(size_config, other_config, policy):
    bs, s, r, cm, cz = size_config
    dtype, evo_num, use_chunk, is_train, is_extra = other_config

    model = AlphaFold2(s,
                       cm,
                       cz,
                       evo_num,
                       use_chunk=use_chunk,
                       is_extra=is_extra,
                       is_train=is_train).to(dtype)
    if not is_train:
        model.eval()

    model = cube.SemanticModel(model,
                               input_shapes=([bs, s, r, cm], [bs, r, r, cz]))

    dataloader = cube.runtime.syndata.SynDataLoader(shapes=([bs, s, r, cm],
                                                            [bs, r, r, cz]),
                                                    dtypes=(dtype, dtype),
                                                    batch_dims=(0, 0))

    @cube.compile(model, dataloader, PAS=policy, override=True)
    def train_iter(model, dataloader):
        msa_repr, pair_repr = next(dataloader)
        loss = model(msa_repr, pair_repr)
        if is_train:
            loss.backward()
        else:
            return loss

    model = model.get_gen_module()

    if is_train:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    warm_up = 2
    iter_num = 4
    CudaTimer(enable=False).warmup()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    for i in range(iter_num):
        if i >= warm_up:
            CudaTimer(enable=True).start('e2e')
        train_iter(model, dataloader)
        if is_train:
            optimizer.step()
            optimizer.zero_grad()
        if i >= warm_up:
            CudaTimer().stop('e2e')
        if i > 0 and (i + 1) % 20 == 0:
            print_each_rank(f'iter [{i + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
        CudaTimer().duration(iter_num - warm_up, field_name='e2e')))
    CudaTimer().print_all(times=iter_num - warm_up)
    print_each_rank('memory consumption: {} MB'.format(
        int(torch.cuda.max_memory_allocated() / 1024 / 1024)))


def profile(graph, resource):
        db = ProfileDataBase()
        mem_sum = 0
        for node in graph.select(ntype=IRFwOperation):
            if isinstance(node, IRGraphAnchor):
                continue
            partition_nodes = gen_partitions(node, 1)
            for partition_node in partition_nodes:
                in_mem, param_mem, fw_span, bw_span, infer_mem, train_mem = db.profile(partition_node)
                mem_sum = mem_sum + train_mem
                print(node.signature, train_mem)
        db.dump('db.json', override=True)
        print('estimated train mem: ', mem_sum / 1024 / 1024 / 1024)

        for node in graph.nodes():
            if not isinstance(node, IRBpOperation):
                graph.assign(node, 0)

        return graph

def test_main():
    # Training && Evoformer Stack
    # initial training
    bs, s, r, cm, cz = 1, 128, 256, 256, 128
    # first fine-tuning
    # bs, s, r, cm, cz = 1, 512, 256, 256, 128
    # second fine-tuning
    # bs, s, r, cm, cz = 1, 512, 384, 256, 128

    dtype, evo_num, use_chunk, is_train, is_extra = torch.float16, 3, False, True, False
    policy = profile
    # policy = spmd.PASDAP

    # Training && Extra Sequence
    # initial training
    # bs, s, r, cm, cz = 1, 1024, 256, 64, 128
    # second fine-tuning
    # bs, s, r, cm, cz = 1, 1024, 384, 64, 128
    # bs, s, r, cm, cz = 1, 5120, 384, 64, 128

    # dtype, evo_num, use_chunk, is_train, is_extra = torch.float16, 4, True, True, True
    # policy = spmd.PASExtraSingle

    # Inference
    # bs, s, r, cm, cz = 1, 128, 2048, 256, 128
    # dtype, evo_num, use_chunk, is_train, is_extra = torch.float32, 48, True, False, False
    # policy = spmd.PASSingleInference
    # policy = spmd.PASDAPInference

    run((bs, s, r, cm, cz), (dtype, evo_num, use_chunk, is_train, is_extra),
        policy)


if __name__ == '__main__':
    cube.init()
    test_main()
