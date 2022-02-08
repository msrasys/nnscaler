
```
function PADataParallel(Graph G, Resource R, Config C):
    for node in G.nodes do
        algorithm <- getPartitionAlgo(node, 'data parallelism')
        subnodes <- G.partition(node, algorithm, C.data_parallel_size)
        for dp_idx in 0 to C.data_parallel_size do
            rank <- mapDpToRank(dp_idx, R)
            G.assign(subnodes[dp_idx], rank)
    return G


function PATensorParallel(Graph G, Resource R, Config C):
    for node in G.nodes do
        algorithm <- getPartitionAlgo(node, 'tensor parallelism')
        subnodes <- G.partition(node, algorithm, C.tensor_parallel_size)
        for tp_idx in 0 to C.tensor_parallel_size do
            rank <- mapTpToRank(tp_idx, R)
            G.assign(subnodes[tp_idx], rank)
    return G


function PAPipelineParallel(Graph G, Resource R, Config C):

    for node in G.nodes do
        algorithm <- getPartitionAlgo(node, 'data parallelism')
        G.partition(node, algorithm, C.num_micro_batches)

    for node in G.nodes do
        stage_id <- getStageID(node, G, C.pipeline_parallel_size) // policy
        rank <- mapStageToRank(stage_id, R)
        G.assign(node, stage)

    // group to a sub-graph (block): A microbatch on one stage 
    groupStageAndMicroBatch(G, C.pipeline_parallel_size, C.num_micro_batches)
    return G


function PSPipelineParallel(Graph G, Resource R, Config C):
    // each node in G stands for a block (sub-graph)
    sequence <- EmptyArray[]
    // warmup phase
    for micro_batch_id in 0 to C.num_micro_batches do
        for stage_id in 0 to C.pipeline_parallel_size - micro_batch_id do
            node <- getForwardBlock(G, micro_batch_id, stage_id)
            arrayPush(sequence, node)
    # steady and cooldown phase
    for micro_batch_id in 0 to C.num_micro_batches do
        // enqueue backward
        for stage_id in C.pipeline_parallel_size to 0 do
            node <- getBackwardBlock(G, micro_batch_id, stage_id)
            arrayPush(sequence, node)
        // enqueue forward
        for stage_id in 0 to C.pipeline_parallel_size do
            mid <- micro_batch_id + C.pipeline_parallel_size - stage_id
            if mid <= C.pipeline_parallel_size then
                node <- getForwardStage(G, mid, stage_id)
                arrayPush(sequence, node)
    G.schedule(sequence)
    return G


function Megatron(Graph G, Resource R, Config C):
    // Graph    G: Dataflow graph containing operators as nodes
    // Resource R: Environment Resource including GPU numbers and topology
    // Config   C: policy user configuration including:
    //               data_parallel_size,
    //               tensor_parallel_size,
    //               pipeline_parallel_size,
    //               num_micro_batches

    // Resource split: group resources
    Rs <- splitResource(R, C)
    R_pp <- getResourceForPP(Rs, C)

    // group into blocks (each block is a microbatch on a stage)
    G <- PAPipelineParallel(G, R_pp, C)

    // inter block scheduling: 1F1B scheduling
    G <- PSPipelineParallel(G, R_pp, C)

    // inner block parallelism: hybrid parallelism
    for block in G.nodes do
        stage_id <- getStageID(G, block)
        // data parallelism
        R_dp <- getResourceForDP(Rs, stage_id)
        PADataParallel(block, R_dp, C)
        // tensor parallelism
        R_tp <- getResourceForTP(Rs, stage_id)
        PATensorParallel(block, R_tp, C)

    return G
```