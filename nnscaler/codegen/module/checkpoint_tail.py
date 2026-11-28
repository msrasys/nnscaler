# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from nnscaler.codegen.syntax.blocks import Block
from nnscaler.flags import CompileFlag
from nnscaler.graph.schedule.module_streams import checkpoint_stream, normalize_module_stream_rules
from nnscaler.ir.cten import IRCell, IRTensor
from nnscaler.ir.operator import IRFwOperation


def emit_checkpoint_tail(codegen, nodes, outputs):
    """Emit the replay branch only for a proven terminal producer/consumer edge."""
    rules = [r for r in normalize_module_stream_rules(CompileFlag.module_streams) if r.recompute_saved_state]
    if not rules or len(nodes) < 2 or CompileFlag.use_zero > 1:
        return None
    producer, consumer = nodes[-2:]
    key = (producer.signature, consumer.signature)
    index = codegen.checkpoint_tails.get(key)
    if index is None or checkpoint_stream(nodes, rules) is None:
        return None
    for node in (producer, consumer):
        if (not isinstance(node, IRFwOperation) or node.pre_hook or node.post_hook
                or codegen._emit_rules.map(node.signature) != codegen._emit_rules.emit_common
                or len(node.outputs()) != 1 or not isinstance(node.output(0), IRTensor)
                or not node.output(0).requires_grad):
            return None
        # Explicit modes and customized emitters can change the backend's ABI.
        ctx = node.op_context
        if ctx and (not ctx['grad_mode']['grad_mode'] or ctx['autocast_info']['nesting']):
            return None
    intermediate = producer.output(0)
    if tuple(outputs) != (consumer.output(0),) or index >= len(consumer.inputs()):
        return None
    args = consumer.inputs()
    if args[index] != intermediate or intermediate in IRCell.get_objects_from_complex(
            (args[:index], args[index + 1:], consumer.kwargs)):
        return None

    def operands(op, omitted=None):
        args = [codegen.tensor_name(value, prefix_attr='self.') if i != omitted else 'None'
                for i, value in enumerate(op.inputs())]
        kwargs = codegen.kwargs_dict(**op.kwargs)
        return ('(' + ', '.join(args) + (',' if args else '') + '), {'
                + ', '.join(f'{k!r}: {v}' for k, v in kwargs.items()) + '}')

    with Block(f'if _rebuilding and nnscaler.runtime.checkpoint_tail.rebuild('
               f'{key!r}, {index}, {operands(producer)}, {operands(consumer, index)}):') as branch:
        branch.insert_body('return')
    return branch.code
