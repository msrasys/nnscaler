import ast
import inspect
import textwrap

from nnscaler.cli.trainer import Trainer


def test_optimizer_preparation_trace_ranges_are_narrow() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(Trainer._train_epoch)))
    observed = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        context = node.items[0].context_expr
        if (
            not isinstance(context, ast.Call)
            or ast.unparse(context.func) != "ct.range"
            or len(context.args) < 2
        ):
            continue
        entity = ast.literal_eval(context.args[1])
        if entity not in {
            "trainer.output.aggregate",
            "optimizer.sync_shard_grad",
            "optimizer.scale_grads",
            "optimizer.clip_gnorm",
            "optimizer.grad_norm.item",
        }:
            continue
        calls = {
            ast.unparse(call.func)
            for statement in node.body
            for call in ast.walk(statement)
            if isinstance(call, ast.Call)
        }
        observed[entity] = {
            "kind": ast.unparse(context.args[0]),
            "calls": calls,
            "process_scope": next(
                (
                    ast.literal_eval(keyword.value)
                    for keyword in context.keywords
                    if keyword.arg == "process_scope"
                ),
                None,
            ),
        }

    expected = {
        "trainer.output.aggregate": ("ct.Kind.REDUCE", "aggregate_outputs"),
        "optimizer.sync_shard_grad": (
            "ct.Kind.REDUCE",
            "self.optimizer.sync_shard_grad",
        ),
        "optimizer.scale_grads": (
            "ct.Kind.OPTIMIZER",
            "self.optimizer.scale_grads",
        ),
        "optimizer.clip_gnorm": (
            "ct.Kind.OPTIMIZER",
            "self.optimizer.clip_gnorm",
        ),
        "optimizer.grad_norm.item": ("ct.Kind.OPTIMIZER", "step_stat.gnorm.item"),
    }
    assert set(observed) == set(expected)
    for entity, (kind, direct_call) in expected.items():
        assert observed[entity]["kind"] == kind
        assert direct_call in observed[entity]["calls"]
        assert observed[entity]["process_scope"] is False
