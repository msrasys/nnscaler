from cube.schedule.su import SUType

def transformation_policy(graph, resource):

    for op in graph.nodes():
        if op.type == SUType.Forward:
            algorithm = op.algorithms('data_parallelism')
            sub_graph = graph.partition(op, algorithm, config=dict(chunk_size=resource.ngpus))
        if op.type == SUType.Optimizer:
            algorithm = op.algorithms('split_axis_0')
            sub_graph = graph.partition(op, algorithm, config=dict(chunk_size=resource.ngpus))

    return graph


def schedule_policy(sugraph, resource):

    semantic_ops = dict()
    for su in sugraph.sus():
        if su.nodes(0).semantic_ops not in semantic_ops:
            semantic_ops[su.nodes(0).semantic_ops] = list()
        semantic_ops[su.nodes(0).semantic_ops].append(su)
    for semantic_op in semantic_ops:
        for idx, su in enumerate(semantic_ops[semantic_op]):
            gpu_id = idx % resource.ngpus 
            sugraph.assign(su, gpu_id)
