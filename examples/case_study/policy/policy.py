"""
DAG interface:

    add_op

    delete_op

    update_op

    find_op / iter_op
"""

def policy(DAG, resources):
    """
    Args:
        * DAG: semantic (logical) computation graph
        * Resources: Environment inlcuding devices, network topology etc
    Returns:
        * DAGs (list[DAG]) execution (local & physical) DAG for each rank
    """
    for inputs, op, outputs in iter_op(DAG):
        if is_annotated(op):
            # distributed op adapter
            dist_op = select_dist_op(op, inputs, resources)
            replace_op(DAG, op, dist_op)
            # input tensor segmentation adapter
            inputs = input_adapter(DAG, dist_op, inputs)
            # output tensor segmentation adapter
            outputs = output_adapter(DAG, dist_op, outputs)
        # tensor placement / lifecycle adapter
        if is_annotated(inputs):
            placement_lifecycle_adapter(DAG, inputs)
        # tensor move / destroy
        if is_annotated(outputs):
            placement_lifecycle_adapter(DAG, outputs)
    # scheduling
    # TODO: do we need to include scheduling in the DAG?
    # materialize to physical op
    DAGs = generate_for_each_rank(DAG, resources)
    return DAGs


def select_dist_op(op, inputs, resources):
    op_candidates = get_distributed_ops(type(op))
    for candidate in op_candidates:
        if candidate.same_segmentation(inputs):
            return candidate
