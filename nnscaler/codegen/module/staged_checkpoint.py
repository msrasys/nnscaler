# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit explicit-parameter replay functions for a checkpoint prefix.

This transformation only applies to a compiler-selected checkpoint group.
Its original forward remains a whole function. The replay functions exchange
explicit live values, with no model names or backend-specific operators.
"""
import ast
import copy
from fnmatch import fnmatchcase
import textwrap


def emit_staged_checkpoint(codes, point, attributes, patterns, output_grad_flags):
    tree = ast.parse(textwrap.dedent('\n'.join(codes)))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'recompute')
    call = next(n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'checkpoint')
    attributes = list(attributes)
    if any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
           and n.value.id == 'self' and n.attr == 'training' for n in ast.walk(function)):
        attributes.append('self.training')
    attributes = list(dict.fromkeys(attributes))
    aliases = {name.removeprefix('self.'): f'_staged_attribute_{i}' for i, name in enumerate(attributes)}

    class Parameters(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                if node.attr not in aliases or not isinstance(node.ctx, ast.Load):
                    raise ValueError(f'Staged replay requires explicit, read-only module attributes: {ast.unparse(node)}')
                return ast.copy_location(ast.Name(aliases[node.attr], ast.Load()), node)
            return self.generic_visit(node)

    function = Parameters().visit(function)
    function.args.args.extend(ast.arg(name) for name in aliases.values())
    matches = []
    for i, node in enumerate(function.body):
        if any(isinstance(c, ast.Call) and any(fnmatchcase(ast.unparse(c.func), p) for p in patterns)
               for c in ast.walk(node)):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                raise ValueError('Staged replay prefix must end at a top-level operator assignment')
            matches.append(i)
    if not matches:
        raise ValueError(f'No replay prefix boundary matched {patterns!r}')
    cut = matches[0]
    live = dict.fromkeys(arg.arg for arg in function.args.args)
    reads = set()
    for node in function.body[:cut + 1]:
        for leaf in ast.walk(node):
            if isinstance(leaf, ast.Name):
                if isinstance(leaf.ctx, ast.Store):
                    live[leaf.id] = None
                elif isinstance(leaf.ctx, ast.Del):
                    live.pop(leaf.id, None)
                elif isinstance(leaf.ctx, ast.Load):
                    reads.add(leaf.id)
    # Tail-only metadata cannot change prefix allocations. Excluding those
    # arguments from the learned-bound key avoids unrelated shape misses.
    key_inputs = tuple(i for i, arg in enumerate(function.args.args) if arg.arg in reads)
    prefix = copy.deepcopy(function)
    prefix.name = 'recompute_prefix'
    prefix.body = copy.deepcopy(function.body[:cut + 1]) + [ast.Return(ast.Dict(
        [ast.Constant(name) for name in live], [ast.Name(name, ast.Load()) for name in live]))]
    tail = copy.deepcopy(function)
    tail.name = 'recompute_tail'
    tail.args = ast.arguments(posonlyargs=[], args=[ast.arg('_replay_values')], vararg=None,
                              kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    tail.body = [ast.Assign([ast.Name(name, ast.Store())], ast.Subscript(
        ast.Name('_replay_values', ast.Load()), ast.Constant(name), ast.Load())) for name in live]
    tail.body += copy.deepcopy(function.body[cut + 1:])
    setup = ast.Expr(ast.Call(ast.Attribute(ast.Name(point, ast.Load()), 'set_stages', ast.Load()),
        [ast.Name(prefix.name, ast.Load()), ast.Name(tail.name, ast.Load()),
         ast.Constant(key_inputs), ast.Constant(tuple(output_grad_flags))], []))
    call.value = ast.Call(ast.Attribute(ast.Name(point, ast.Load()), 'checkpoint', ast.Load()),
                         call.value.args + [ast.parse(name, mode='eval').body for name in attributes], [])
    index = tree.body.index(call)
    tree.body[index:index] = [prefix, tail, setup]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).splitlines()
