# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .concrete_tracer import ConcreteTracer

import sys
import ast
import builtins
import inspect
import logging

from textwrap import dedent
from types import MethodType, FunctionType
from typing import List, Optional, Callable, Dict, Tuple

import torch

from .utils import (
    _orig_type,
    _orig_isinstance,
    _orig_len,
    _orig_dict,
    _orig_zip,
    _orig_tuple,
)

_logger = logging.getLogger(__name__)


class TrackedTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.modified = False


class OperatorTransformer(TrackedTransformer):
    func_map = {
            ast.Not: 'not_',     # operator.not_
            ast.Is: 'is_',       # operator.is_
            ast.IsNot: 'is_not', # operator.is_not
            ast.In: 'contains',  # operator.contains
    }
    def visit_UnaryOp(self, node: ast.UnaryOp):
        if _orig_isinstance(node.op, ast.Not):
            self.modified = True
            return self.generic_visit(ast.Call(
                func=ast.Name(id=self.func_map[ast.Not], ctx=ast.Load()),
                args=[node.operand],
                keywords=[]
            ))
        else:
            return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        if not any(_orig_isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)) for op in node.ops):
            return self.generic_visit(node)
        if _orig_len(node.ops) != 1:
            raise RuntimeError('Chained Comparison is not supported')
        self.modified = True
        if _orig_isinstance(node.ops[0], (ast.In, ast.NotIn)):
            args = [node.comparators[0], node.left]
        else:
            args = [node.left, node.comparators[0]]

        if not _orig_isinstance(node.ops[0], ast.NotIn):
            ret_node = ast.Call(
                    func=ast.Name(id=self.func_map[type(node.ops[0])], ctx=ast.Load()),
                    args=args,
                    keywords=[],
            )
        else:
            # not in => operator.not_(operator.contains())
            in_node = ast.Call(
                    func=ast.Name(id=self.func_map[ast.In], ctx=ast.Load()),
                    args=args,
                    keywords=[],
            )
            ret_node = ast.Call(
                func=ast.Name(id=self.func_map[ast.Not], ctx=ast.Load()),
                args=[in_node],
                keywords=[]
            )

        return self.generic_visit(ret_node)


class SuperTransformer(TrackedTransformer):
    """
    Convert super() to super(self.__class__, self)
    Because in Patcher, we only patch funtions (instead of class).
    super() is not supported for a standalone function.
    """
    def visit_Call(self, node: ast.Call):
        if _orig_isinstance(node.func, ast.Name) and node.func.id == 'super' and _orig_len(node.args) == 0:
            self.modified = True
            # convert super() to super(self.__class__, self)
            return self.generic_visit(ast.Call(
                func=ast.Name(id='super', ctx=ast.Load()),
                args=[
                    ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='__class__', ctx=ast.Load()),
                    ast.Name(id='self', ctx=ast.Load()),
                ],
                keywords=node.keywords,
            ))
        else:
            return self.generic_visit(node)


class ProxyCallTransformer(TrackedTransformer):
    def __init__(self, proxy_call_name: str, ignore_funcs: Optional[List[str]] = None) -> None:
        """
        Args:
            proxy_call_name: the name of the proxy function
            ignore_funcs: a list of function names that should not be transformed
        """
        super().__init__()
        self.proxy_call_name = proxy_call_name
        self.ignore_funcs = ignore_funcs or []

    def visit_Call(self, node: ast.Call):
        # will transform all function call to `proxy_call_name(func_name, *args, **kwargs)`
        # node.func can be expression, in that case, node.func.id is undefined.
        if not _orig_isinstance(node.func, ast.Name) or (
            node.func.id != self.proxy_call_name and node.func.id not in self.ignore_funcs
        ):
            self.modified = True
            return self.generic_visit(ast.Call(
                func=ast.Name(id=self.proxy_call_name, ctx=ast.Load()),
                args=[node.func, *node.args],
                keywords=node.keywords,
            ))
        else:
            return self.generic_visit(node)


def transform(node: ast.AST, transformers: List[TrackedTransformer]) -> Tuple[bool, ast.AST]:
    modified = False
    for transformer in transformers:
        node = transformer.visit(node)
        modified = modified or transformer.modified

    if modified:
        return True, ast.fix_missing_locations(node)
    else:
        return False, node


class OperatorPatcher:
    """
    An function patcher, to patch the un-wrappable operator 'not/is/is not/in/not in' to wrappable functions.
    """

    def __init__(self, use_operator_patch: bool, operator_patch_backlist: List[str]):
        self.use_operator_patch = use_operator_patch
        self.operator_patch_backlist = operator_patch_backlist
        self.proxy_call_name = OperatorPatcherContext.patch_run.__name__

    def patch_inner(self, func):
        return self.patch_inner_helper(func)

    def patch_inner_helper(self, func):
        if not hasattr(func, '__module__') or func.__module__ is None or func.__module__.startswith('torch'):
            return func
        # those flags are set by fx _Patcher when a method is patched
        # we don't want to patch it again
        # _Patcher__fx_already_patched is for torch 2.0.1+
        # __fx_already_patched is for torch 2.0.0
        if hasattr(func, '_Patcher__fx_already_patched') or hasattr(func, '__fx_already_patched'):
            return func
        if self.use_operator_patch == (func in self.operator_patch_backlist):
            return func
        if _orig_isinstance(func, torch.nn.Module):
            func = func.forward
        if _orig_isinstance(func, MethodType):
            func_inner = func.__func__
            the_self = func.__self__
        else:
            func_inner = func
            the_self = None
        if not _orig_isinstance(func_inner, FunctionType) or not hasattr(func_inner, '__code__'):
            return func

        lines, lnum = inspect.findsource(func_inner)
        func_name = getattr(func, '__name__', 'new_func')
        # align with original source code
        source = ''.join(('\n' * lnum, *inspect.getblock(lines[lnum:])))
        dedent_src = dedent(source)
        tree = ast.parse(dedent_src)

        # transformers have states, so we can't reuse them.
        is_transformed, new_tree = transform(tree, [
            OperatorTransformer(),
            SuperTransformer(),
            ProxyCallTransformer(self.proxy_call_name)
        ])
        if not is_transformed:
            return func
        else:
            body0: ast.FunctionDef = new_tree.body[0]
            body0.body = [
                # equals to:
                # from operator import not_, is_, is_not, contains
                ast.ImportFrom(
                    module='operator',
                    names=[
                        ast.alias(name='not_'),
                        ast.alias(name='is_'),
                        ast.alias(name='is_not'),
                        ast.alias(name='contains'),
                    ],
                    level=0
                ),
                *body0.body
            ]
            body0.name = func_name
            # for deleting some annotations like 'add_start_docstrings_to_model_forward' or 'add_code_sample_docstrings'
            # these decorators are used for tranformers model docstrings generation, can be removed in trace
            transform_useless_decorators = ('add_start_docstrings_to_model_forward', 'add_code_sample_docstrings', 'replace_return_docstrings')
            body0.decorator_list = [i for i in body0.decorator_list
                if isinstance(i, ast.Call) and isinstance(i.func, ast.Name) and i.func.id == self.proxy_call_name and
                    isinstance(i.args[0], ast.Name) and
                    i.args[0].id not in transform_useless_decorators]
            ast.fix_missing_locations(new_tree)

            # closure info
            closure_dict = {}
            closures = func_inner.__closure__
            co_freevars = func_inner.__code__.co_freevars
            if (closures != None and _orig_len(closures) != 0) or _orig_len(co_freevars) != 0:
                assert _orig_len(closures) == _orig_len(co_freevars)
                closure_dict = _orig_dict(_orig_zip(co_freevars, [c.cell_contents for c in closures]))

            tuple_wrapped = tuple
            try:
                if sys.version_info < (3, 9):
                    setattr(builtins, 'tuple', _orig_tuple)
                var_dict = {}
                exec(
                    # use func.__code__.co_filename to make the new function easily debuggable.
                    compile(new_tree, func_inner.__code__.co_filename, 'exec'),
                    {
                        self.proxy_call_name: OperatorPatcherContext.patch_run,
                        **func_inner.__globals__,
                        **closure_dict,
                    },
                    var_dict)
                if the_self is not None:
                    return var_dict[func_name].__get__(the_self)
                else:
                    return var_dict[func_name]
            finally:
                if sys.version_info < (3, 9):
                    setattr(builtins, 'tuple', tuple_wrapped)


class OperatorPatcherContext:
    ctx_tracer: Optional['ConcreteTracer'] = None
    ctx_patcher: Optional[OperatorPatcher] = None

    def __init__(self, tracer: 'ConcreteTracer', use_operator_patch: bool, operator_patch_backlist: List[str]):
        self.tracer = tracer
        self.patcher = OperatorPatcher(use_operator_patch, operator_patch_backlist)

    def __enter__(self):
        assert OperatorPatcherContext.ctx_tracer is None
        assert OperatorPatcherContext.ctx_patcher is None
        OperatorPatcherContext.ctx_tracer = self.tracer
        OperatorPatcherContext.ctx_patcher = self.patcher

    def __exit__(self, exc_type, exc_value, tb):
        assert OperatorPatcherContext.ctx_tracer == self.tracer
        OperatorPatcherContext.ctx_tracer = None
        OperatorPatcherContext.ctx_patcher = None
        return exc_type is None

    @staticmethod
    def patch_run(func, *args, **kwargs):
        assert OperatorPatcherContext.ctx_tracer is not None
        assert OperatorPatcherContext.ctx_patcher is not None
        with OperatorPatcherContext.ctx_tracer.do_temp_disable(True, True, True):
            new_func = OperatorPatcherContext.ctx_patcher.patch_inner(func)
        return new_func(*args, **kwargs)
