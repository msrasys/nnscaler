
```
function PADataParallel(Graph G, Resource R, Config C):
    for node in G.nodes() do
        algorithm <- getPartitionAlgo(node, 'data parallelism')
        subnodes <- G.partition(node, algorithm, C.data_parallel_size)
        for dp_idx in 0 to C.data_parallel_size do
            rank <- mapDpToRank(dp_idx, R)
            G.assign(subnodes[dp_idx], rank)
    return G


function PATensorParallel(Graph G, Resource R, Config C):
    for node in G.nodes() do
        algorithm <- getPartitionAlgo(node, 'tensor parallelism')
        subnodes <- G.partition(node, algorithm, C.tensor_parallel_size)
        for tp_idx in 0 to C.tensor_parallel_size do
            rank <- mapTpToRank(tp_idx, R)
            G.assign(subnodes[tp_idx], rank)
    return G


function PAPipelineParallel(Graph G, Resource R, Config C):

    for node in G.nodes() do
        algorithm <- getPartitionAlgo(node, 'data parallelism')
        G.partition(node, algorithm, C.num_micro_batches)

    for node in G.nodes() do
        stage_id <- getStageID(node, G, C.num_stages) // policy
        rank <- mapStageToRank(stage_id, R)
        G.assign(node, stage)

    groupStageAndMicroBatch(G, C.num_stages, C.num_micro_batches)
    return G


function PSPipelineParallel(Graph G, Resource R, Config C):
    // each node in G stands for a stage (sub-graph)
    sequence <- EmptyArray[]
    // warmup phase
    for micro_batch_id in 0 to C.num_micro_batches do
        for stage_id in 0 to C.num_stages - micro_batch_id do
            node <- getForwardStage(G, micro_batch_id, stage_id)
            arrayPush(sequence, node)
    # steady and cooldown phase
    for micro_batch_id in 0 to C.num_micro_batches do
        // enqueue backward
        for stage_id in C.num_stages to 0 do
            node <- getBackwardStage(G, micro_batch_id, stage_id)
            arrayPush(sequence, node)
        // enqueue forward
        for stage_id in 0 to C.num_stages do
            mid <- micro_batch_id + C.num_stages - stage_id
            if mid <= C.num_stages then
                node <- getForwardStage(G, mid, stage_id)
                arrayPush(sequence, node)
    G.schedule(sequence)
    return G


function Megatron(Graph G, Resource R, Config C):
    // Resource split
    R_data, R_pipe, R_tensor <- splitResource(R, C)
    // split to stages
    G <- PAPipelineParallel(G, R_pipe, C)
    // inner stage: data + tensor parallelism
    for stage in G.nodes:
        PADataParallel(stage, R_data, C)
        PATensorParallel(stage, R_tensor, C)
    // inter stage: 1F1B scheduling
    G <- PSPipelineParallel(G, R_pipe, C)
    return G
```