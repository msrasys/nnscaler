

def policy(DAG, resources):
    """
    Args:
        * DAG: semantic (logical) computation graph
        * Resources: Environment inlcuding devices, network topology etc
    Returns:
        * DAGs (list[DAG]) execution (local & physical) DAG for each rank
    """
    for inputs, op, outputs in DAG:
        # tensor placement / lifecycle adapter
        if is_annotated(inputs):
            placement_lifecycle_adapter(DAG, inputs)
        if is_annotated(op):
            # distributed op adapter
            dist_op = select(op, inputs, resources)
            replace(DAG, op, dist_op)
            # input tensor segmentation adapter
            input_adapter(DAG, dist_op, inputs)
            # output tensor segmentation adapter
            output_adapter(DAG, dist_op, outputs)
        # tensor move / destroy
        if is_annotated(outputs):
            placement_lifecycle_adapter(DAG, outputs)
    DAGs = generate_for_each_rank(DAG, resources)
    return DAGs


def select(op, inputs, resources):
    op_candidates = get_distributed_ops(type(op))
    for candidate in op_candidates:
        if candidate.same_segmentation(inputs):
            return candidate
