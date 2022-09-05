"""
Generate Pytorch code given the model DAG and the transformation config
"""
import itertools
from typing import Dict, Generator, Iterable, List, Any, Optional, Set, Tuple, Union
import torch
import copy
from more_itertools import split_when
from cube.graph.parser.mapping import Sign2Op

from cube.ir.cten import IRCell, IRTensor
from cube.ir.dtype import IRDType

from cube.ir.tensor import IRSubTensor
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation
from cube.ir.adapter import IRWeightReducer, IRAdapter
from cube.ir.adapter.prim import CollectivePrim, IRAdapterPrim
from cube.graph.graph import IRSegment
from cube.graph.schedule import IRScheduleStrategy

from cube.execplan import ExecutionPlan

from cube.codegen.syntax.symtable import SymbolTable
from cube.codegen.syntax.blocks import ClassBlock, FunctionBlock
from cube.codegen.frontend_mapping import Sign2EmitRule

import os

def get_backward_callsite_io_tensors(bp_segment:IRSegment):
    """
    Returns:
    ```
    (input_tensors, output_tensors, output_grads, input_grads)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~
    #inputs to 'backward'                         outputs of 'backward'
    ```
    """
    assert isinstance(bp_segment, IRSegment) and not bp_segment.isfw()

    input_tensors = [t for t in bp_segment.mirror.inputs() if \
        isinstance(t, IRSubTensor) and \
        t.requires_grad and \
        not t.is_attr()
    ]
    output_tensors = [t for t in bp_segment.mirror.outputs() if isinstance(t, IRSubTensor)]
    input_grads = [t.grad for t in input_tensors]

    # WARNING !!!
    # non-tensor gradients like scalar '1.0f' are removed in 'bpSeg.inputs()'
    # so the items of 'bpSeg.inputs()' are generally disaligned with 'output_grads' here.
    output_grads = [t.grad for t in output_tensors]

    return input_tensors, output_tensors, output_grads, input_grads

# TODO this could be applied in Adapters
def calc_tenvars_lifetime(
        nodes:Iterable[IRCell], 
        subgraph_outputs:Iterable[IRTensor],
        subgraph_inputs:Iterable[IRTensor] = []
    ) -> Dict[IRTensor, int]:
    """
    Calculate the lifetime of tensor variables ahead-of-time.
    So that during schedule the GC on those variables can take place in time.

    E.g. at what timings may a tensor variable O_i be discarded (i.e. no longer referred)?
    ```
    ..., O_i, O_j, ... = f(I_1, ..., I_M)
    # Case 1, immediately, because it's never used
    O_i = None
    # Case 2, after some invocation, because it's no longer referred
    ... = g(..., O_j, ...)
    O_j = None
    ```

    Returns: `Dict[IRTensor, int]`

        For each kv-pair `(t, i)` it indicates the last reference of tensor `t`
        is at the `i`-th (0-based) node's inputs, 
        i.e. the variable for tensor `t` could be released *BEFORE* the `i`-th statement 
        in codegen.

        If an input of the subgraph is never used, its corresponding `i` is `0` -- this will
        lead to an immediate release at the beginning of a function.
        Tensors that exist till the end of the subgraph will have lifetime greater than the
        size of that subgraph. Generally we don't need to manually release those tensors,
        since they are automatically released when the generated function returns.
    """

    lifetime : Dict[IRTensor, int] = dict()

    def is_temp_tensor(v):
        return isinstance(v, IRSubTensor) and not v.is_attr()

    lifetime.update((tsin, 0) for tsin in subgraph_inputs if is_temp_tensor(tsin))

    for i, node in enumerate(nodes):

        outputs : Iterable[IRTensor]
        inputs : Iterable[IRTensor]

        if isinstance(node, IRSegment):
            if node.isfw():
                outputs = node.outputs()
                inputs = node.inputs()
            else:
                # NOTE
                # An backward 'IRSegment' does not explicitly record all tensors that are
                # inputs-and-outputs-to-its-correspondeding-autograd.grad-call after codegen.
                #
                # Where a call to 'torch.autograd.grad' in Python is like:
                # ```
                # grad_inputs : Tuple[torch.Tensor, ...] = torch.autograd.grad(outputs, inputs, grad_outputs)
                # len(grad_inputs) == len(inputs)
                # len(grad_outputs) == len(outputs)
                # ```
                #
                # But a backward 'IRSegment' itself only records _extra_ information to take
                # gradients for inputs to a forward 'IRSegment':
                #
                # - Inputs of the backward 'IRSegment' are
                #   gradient tensors for outputs of the corresponding forward 'IRSegment'
                #
                #   WARNING: non-tensor gradients like scalar '1.0f' are removed,
                #   so the items of 'bpSeg.inputs()' are generally disaligned with 'fw_outputs()'
                #
                # - Outputs of the backward 'IRSegment' are
                #   gradient tensors for both explicit and implicit inputs of the forward 'IRSeg'
                #
                #   P.S. the implicit inputs of the forward 'IRSeg' are like 'nn.Parameter's
                #   which are model fields and accessed by e.g. 'self.weights'.
                #   Generally, by viewing a gradient tensor of some input, we cannot distinguish
                #   whether the corresponding input is explicit or implicit.

                fw_inputs, fw_outputs, output_grads, input_grads = \
                    get_backward_callsite_io_tensors(node)

                outputs = input_grads
                inputs = list(itertools.chain(fw_inputs, fw_outputs, output_grads))

        else:
            outputs = node.outputs()
            inputs = node.inputs()

        # aggressively mark all outputs for immediate deletion,
        # namely *before* 'i+1'-th statement, in case it's never used.
        lifetime.update((tout, i+1) for tout in outputs if is_temp_tensor(tout))

        # "fast-forward" all inputs to the current statement, namely before 'i+1'-th node.
        lifetime.update((tin, i+1) for tin in inputs if is_temp_tensor(tin))

    # end of 'for'

    # Here (i+1) is always greater than 'len(nodes)'
    # Generally we don't manually release those tensors since the enclosing function is about to
    # return, all local variables are automatically released.
    # But we do need to update the lifetime of all outputs, to avoid early releasing.
    lifetime.update((tsout, i+1) for tsout in subgraph_outputs if is_temp_tensor(tsout))

    return lifetime


class CodeGen:
    """
    Generate code for the model
    """
    def __init__(self, execplan: ExecutionPlan):
        if not isinstance(execplan, ExecutionPlan):
            raise TypeError("execplan should be ExecutionPlan")
        self.execplan = execplan

    def dtype_map(self, dtype: IRDType) -> str:
        if not isinstance(dtype, IRDType):
            raise TypeError("Expected IRDType")
        return 'torch.' + dtype.value

    def node_naming(self, node: IRCell) -> str:
        return f"{node.name}{node._id}"

    def tensor_naming(self, tensor: Any, prefix_attr: Optional[str] = None) -> str:
        """
        Return the var name.
        For tensor, return the {prefix}{tensor.name}_{tensor.tid}
        For non-tensor, return its string

        @param tensor Any: any value
        @attr_prefix Optional[str]: prefix for a attributed tensor

        @return str
        """
        if isinstance(tensor, IRTensor):
            tensor_name = tensor.name
            if '.' in tensor_name:
                tensor_name = tensor_name.split('.')[0]
            name = '_'.join([tensor_name, str(tensor.tid)])
            if prefix_attr is not None and tensor.is_attr():
                name = prefix_attr + name
        else:
            name = str(tensor)
        return name

    def tuple_naming(self, tensors: List[Any],
                     skip_attr: bool = False, prefix_attr: Optional[str] = None) -> str:
        """
        Return the tupled tensor name.

        @param tensors List[Any]: list of any value
        @param skip_attr bool: whether to skip graph attribute in the tensors
        @param prefix_attr bool: whether to add a prefix for graph attribute

        @return name str: the tupled tensor name
        """
        names = []
        for t in tensors:
            if isinstance(t, IRTensor) and skip_attr and t.is_attr():
                continue
            names.append(self.tensor_naming(t, prefix_attr))
        name = '(' + ', '.join(names + ['']) + ')'
        return name

    def return_naming(self, tensors: List[Any],
                      skip_attr: bool = False, prefix_attr: Optional[str] = None) -> str:
        """
        Return the tensors in return format, i.e. tupled name without brackets.

        @param tensors List[Any]: list of any value
        @param skip_attr bool: whether to skip graph attribute in the tensors
        @param prefix_attr bool: whether to add a prefix for graph attribute

        @return name str: the tupled tensor name
        """
        names = []
        for t in tensors:
            if isinstance(t, IRTensor) and skip_attr and t.is_attr():
                continue
            names.append(self.tensor_naming(t, prefix_attr))
        names = '_' if len(names) == 0 else ', '.join(names)
        return names

    def kwargs_naming(self, **kwargs) -> str:
        """
        Return the kwarg naming, connected by ', '

        @param kwargs Dict[str, Any]: kwargs

        @return name str
        """
        names = []
        for name, val in kwargs.items():
            names.append(f'{name}={val}')
        name = ', '.join(names)
        return name

    def emit_tensors_release(self, tensors:Iterable[IRTensor]) -> str:
        tnames : Generator = (self.tensor_naming(t) for t in tensors)
        return 'del ' + ', '.join(tnames)


class AutogradAdapterCodeGen(CodeGen):
    """
    Generate autograd adapter code (PyTorch)
    """
    def __init__(self):

        self.fw_ins: List[IRSubTensor] = list()
        self.fw_body: List[str] = list()
        self.fw_ous: List[IRSubTensor] = list()

        self.bw_ins: List[IRSubTensor] = list()
        self.bw_body: List[str] = list()
        self.bw_ous: List[IRSubTensor] = list()

    def emit_prim(self, prim: IRAdapterPrim) -> str:
        if len(prim.inputs()) == 1:
            itensors = self.tensor_naming(prim.inputs()[0])
        else:
            itensors = self.tuple_naming(prim.inputs())
        kwargs = list()
        for name, val in prim.kwargs.items():
            kwargs.append(f'{name}={val}')
        kwargs = ', '.join(kwargs)
        outputs = self.return_naming(prim.outputs())
        code = f'{outputs} = {prim.signature}({itensors}, {kwargs})'
        return code

    def gen(self, fadapter: IRAdapter) -> List[str]:
        assert fadapter.forward and fadapter.differentiable and fadapter.custom, "generate autograd for a non-differentiable adapter"
        assert fadapter.mirror is not None
        name = AutogradAdapterCodeGen.name(fadapter)
        with ClassBlock(class_name=name, derived=['torch.autograd.Function']) as cb:
            # forward
            cb.insert_body('@staticmethod')
            finputs = [self.tensor_naming(t) for t in fadapter.inputs()]
            with FunctionBlock(func_name='forward', args=['ctx']+finputs) as fw:
                for prim in fadapter.prims:
                    fw.insert_body(self.emit_prim(prim))
                outputs = self.return_naming(fadapter.outputs())
                fw.insert_body(f'return {outputs}')
            cb.insert_body(fw.code)
            # backward
            cb.insert_body('@staticmethod')
            badapter: IRAdapter = fadapter.mirror
            binputs = [self.tensor_naming(t) for t in badapter.inputs()]
            with FunctionBlock(func_name='backward', args=['ctx']+binputs) as bw:
                for prim in badapter.prims:
                    bw.insert_body(self.emit_prim(prim))
                outputs = self.return_naming(badapter.outputs())
                bw.insert_body(f'return {outputs}')
            cb.insert_body(bw.code)
        return cb.code
    
    @staticmethod
    def name(adapter: IRAdapter) -> str:
        return f'Adapter{adapter.cid}'


class ModelCodeGen(CodeGen):
    """
    Generate model code

    `ModelCodeGen` traverses all IR nodes and categorizes their intermediately generated
    codes into different parts, 
    then reorders and concatenates these parts into the final code for PyTorch to run.

    These parts are progressively stored into fields of `ModelCodeGen`
    
    - `init_code : List[str]`
        Statements like `import torch`

    - `model_init_statements : List[str]`
        Statements of the `__init__` constructor of the final `nn.Module` in codegen,

        E.g. (lines are split into `List[str]`)
        ```python
        self.init_group(ranks=[0, 1, 2, 3])
        self.weight_63 = torch.nn.Parameter(torch.empty((2048, 8192), dtype=torch.float32))
        self.add_full_map('weight_63', 3, (slice(0, 2048, None), slice(0, 8192, None)), 1)
        ```

        including:
        -- initialization of model weights, which are class fields;

    - `model_methods_bodies : List[List[str]]`
        Definitions of the Python code for forward computations like Segments or Adapters

        Note that codes within this field haven't been organized into valid Python methods,
        namely without signatures and return statements, both of which will be extracted
        from corresponding IRSegment/IRAdapter in later processes.
        E.g.
        ```
        [
            # intermediate codes for 'segment123(self, tensor_2222)'
            [
                'tensor_3333 = torch.view(tensor_2222, [1,2,3,4])'
            ]
            
            # intermediate codes for 'adapter456(self, tensor_4444)'
            [
                'tensor_5555 = cube.runtime.adapter.all_reduce(tensor_4444, ranks=[0,1,2,3])'
            ]
        ]
        ```
    """

    def __init__(self, execplan: ExecutionPlan):
        super().__init__(execplan)
        # model full code
        self.init_code: List[str] = [
            '\n\n########## Generated Model Code ###########',
            'from typing import *',
            'import torch', 'import torch.utils.checkpoint as ckpt',
            'import cube', '', '']

        use_nnfusion = os.environ.get('USE_NNFUSION')
        if use_nnfusion:
            self.init_code.extend(['import nnfusion', ''])

        # customized op code
        for _, op_impl in Sign2Op.kOpCodeDef.items():
            self.init_code.append(op_impl)
            self.init_code += ['']
        # module init code
        self.model_init_statements: List[str] = list()
        # module method bodies for forward computations, e.g. Segments, Adapters.
        self.model_methods_bodies: List[List[str]] = list()
        # module member name
        self.symbols = SymbolTable()
        # ref module to check shared variables
        self._ref_module = torch.nn.Module()

    def init_comm_groups(self):
        """
        Get all communication groups.

        Creating communication group requires all the devices
        enter the same call.

        The fields storing intermediate codes that are populated by this method:
        - `model_init_statements`
        """
        graph = self.execplan.graph
        sign = 'self.init_group(ranks={ranks})'
        # collect groups from weight reducer
        comm_groups: Dict[Tuple[int]] = list()
        for node in graph.nodes():
            if isinstance(node, IRWeightReducer):
                ranks = list(node.device)
                ranks.sort()
                ranks = tuple(ranks)
                if ranks not in comm_groups:
                    comm_groups.append(ranks)
        # collect groups from p2p fusion
        adapters = [n for n in graph.nodes(flatten=True) if isinstance(n, IRAdapter)]
        for adapter in adapters:
            for prim in adapter.prims:
                if isinstance(prim, CollectivePrim):
                    ranks = list(prim.kwargs['ranks'])
                    ranks.sort()
                    ranks = tuple(ranks)
                    if ranks not in comm_groups:
                        comm_groups.append(ranks)
        # create communication group
        self.model_init_statements.append('# communication groups')
        for ranks in comm_groups:
            code = sign.format(ranks=list(ranks))
            self.model_init_statements.append(code)
        self.model_init_statements.append(' ')

    def gen(self, device: int, outfile=None, attach=False) -> str:
        """
        Generate model implementation code based on the given graph.
        """
        gencode = copy.copy(self.init_code)
        node_args: List[List[str]] = list()
        gen_nodes: List[IRCell] = list()

        # init customized adapter
        for seg in [seg for seg in self.execplan.seq(device) if isinstance(seg, IRSegment)]:
            for adapter in [n for n in seg.nodes() if isinstance(n, IRAdapter)]:
                if adapter.forward and adapter.differentiable and adapter.custom:
                    gencode += AutogradAdapterCodeGen().gen(adapter) + ['', '']
                    adapter.signature = AutogradAdapterCodeGen.name(adapter) + '.apply'

        # initialize communication groups
        self.init_comm_groups()

        # parse graph body
        for node in self.execplan.seq(device):
            if isinstance(node, IRSegment):
                if not node.isfw(): continue  # skip backward segment
                codes = self.emit_segment_code(node)
            elif isinstance(node, IRFwOperation):
                raise RuntimeError(f"Unexcepted global-level op call: {node}")
            elif isinstance(node, IRAdapter):
                codes = self.emit_adapter_code(node)
            elif isinstance(node, IRWeightReducer):
                self.emit_reducer_init(node)
                codes = self.emit_reducer_call(node)
            elif isinstance(node, IRBpOperation):
                continue
            elif isinstance(node, IRDataOperation):
                continue
            else:
                raise RuntimeError(f"Un-recognized IRCell type: {type(node)}")

            # emit node tensor declaration into `__init__`
            # typically it's about the `nn.Parameter`
            self.emit_node_tensors_declare(node)

            # emit node code
            # codes : List[str]
            self.model_methods_bodies.append(codes)
            gen_nodes.append(node)

            args = list()
            for t in node.inputs():
                if isinstance(t, IRSubTensor):
                    if not t.is_attr():
                        args.append(self.tensor_naming(t))
                else:
                    args.append(self.tensor_naming(t))
            node_args.append(args)

        use_nnfusion = os.environ.get('USE_NNFUSION')
        # generate full code
        with ClassBlock(class_name='GenModel', derived=['cube.runtime.module.CubeModule']) as cb:
            with FunctionBlock(func_name='__init__', args=['self']) as ib:
                ib.insert_body(self.model_init_statements)
                # switch to training or inference mode
                if self.execplan.inference:
                    ib.insert_body('self.eval()')
                else:
                    ib.insert_body('self.train()')
            cb.insert_body('')
            cb.insert_body(ib.code)
            for idx, node in enumerate(gen_nodes):
                name = self.node_naming(node)
                input_args = ['self'] + node_args[idx]
                forward_code = self.model_methods_bodies[idx]

                with FunctionBlock(func_name=name, args=input_args) as fb:
                    fb.insert_body(forward_code)
                    # generate output
                    outputs = [self.tensor_naming(t) for t in node.outputs()]
                    return_code = f"return {', '.join(outputs)}"
                    fb.insert_body(return_code)
                cb.insert_body('')
                if use_nnfusion and name.startswith('segment'):
                    cb.insert_body('@nnfusion.jit')
                cb.insert_body(fb.code)


        gencode += cb.code
        gencode += ['']

        code = '\n'.join(gencode)
        # write to file
        if outfile:
            with open(outfile, 'a' if attach else 'w') as f:
                f.write(code)
        
        # clear used buffer
        self.clear()
        return code

    def emit_node_tensors_declare(self, node: IRCell):
        """
        Emit tensor declaration code

        The fields storing intermediate codes that are populated by this method:
        - `model_init_statements`

        This method also populates `self.symbols : SymbolTable` to record
        the names of the variables for the tensors ever encountered.
        """
        psign = "self.register_parameter('{name}', torch.nn.Parameter(torch.empty({shape}, dtype={dtype})))"
        bsign = "self.register_buffer('{name}', torch.empty({shape}, dtype={dtype}))"
        map_sign = "self.add_full_map('{attr}', {tid}, {slicers}, {val_chunks})"
        if not isinstance(node, IRSegment):
            for itensor in node.inputs():
                name = self.tensor_naming(itensor, prefix_attr='self.')
                if isinstance(itensor, IRSubTensor):
                    if itensor.is_attr() and not self.symbols.exist(name):
                        self.symbols.create(name)
                        sign = psign if itensor.is_param() else bsign
                        code = sign.format(
                            name=self.tensor_naming(itensor),
                            shape=tuple(itensor.shape),
                            dtype=self.dtype_map(itensor.dtype)
                        )
                        self.model_init_statements.append(code)
                        tid = itensor.parent.tid
                        slicers = tuple(slice(start, stop) for (start, stop) in itensor.indmap)
                        val_chunks = itensor.valmap[1]
                        code = map_sign.format(
                            attr=self.tensor_naming(itensor), tid=tid,
                            slicers=str(slicers), val_chunks=val_chunks
                        )
                        self.model_init_statements.append(code)
                        self.model_init_statements.append('')
                if isinstance(itensor, str):
                    if name.startswith('self.'):
                        if not hasattr(self._ref_module, name[5:]):
                            raise NotImplementedError("member attribute is not added")
            for output in node.outputs():
                self.symbols.create(self.tensor_naming(output, prefix_attr='self.'))
        else:
            for sub_node in node.nodes():
                self.emit_node_tensors_declare(sub_node)
        return

    def emit_segment_code(self, segment: IRSegment) -> List[str]:
        """
        Emit IRSegment code.

        The resultant `List[str]` will be lines of the statements of the final
        Python method for the targeted Segment.
        The resultant lines will not include the signature and the return statement
        of the generated Python method. These lines will be put into `model_methods_bodies`
        and the missing Python-syntactic parts will be injected later on.

        e.g.
        ```
        [
            # no method signature
            'tensor_3333 = torch.view(tensor_2222, [1,2,3,4])',
            'tensor_2222 = None',   # if in dataflow there is no more reference
            'tensor_4444 = torch.sum(tensor_3333)',
            'def recompute(...):',
            '    return ...',
            'tensor_5555 = torch.utils.checkpoint(recompute, tensor_4444)',
            'tensor_4444 = None',   # if in dataflow there is no more reference
            # no return statement
        ]
        ```

        Nodes in the segment will group into recompute region

        The fields storing intermediate codes that are populated by this method:
        - NONE
        """

        codes = []
                    
        def emit_nodes_invocations(i_nodes: List[Tuple[int, IRCell]], 
                                   lifetime_by_line_id: Dict[int, List[IRTensor]]) -> List[str]:
            """
            Emit code to invoke operations and adapter, 
            e.g. (the lines are split into `List[str]`)

            ```
            tensor_2222 = torch.view(tensor_1111, size=[3,6,9])
            tensor_1111 = None      # if no more reference
            tensor_3333 = cube.runtime.adapter.allgather_reducescatter(tensor_2222, dim=1, rank=[0,1])
            tensor_2222 = None      # if no more reference
            ```

            The fields storing intermediate codes that are populated by this method:
            - NONE
            """
            node_codes = []
            for i, node in i_nodes:

                # NOTE
                # If a tensor is still referenced in any later recomputing group, its lifetime is
                # definitely greater than the current sequence of statements here.
                # Therefore we get chance to extend the lifetime of tensors like that,
                # and properly release them after the call to 'torch.utils.checkpoint'.
                #
                tensors_to_del : Optional[List[IRTensor]] = lifetime_by_line_id.get(i, None)
                if tensors_to_del is not None:
                    node_codes.append(self.emit_tensors_release(tensors_to_del))

                if isinstance(node, IRFwOperation):
                    code = self.emit_op_code(node)
                    node_codes += code
                elif isinstance(node, IRAdapter):
                    code = self.emit_adapter_code(node)
                    node_codes += code
                else:
                    raise RuntimeError(f"unexpected type {type(node)} in IRSegment")

            return node_codes

        # returns: (code_lines, group_inputs, group_outputs)
        def emit_rc_nodes(i_nodes: List[Tuple[int, IRCell]], lifetime_by_line_id: dict) \
            -> Tuple[List[str], List[IRTensor], List[IRTensor]]:
            """
            Emit code to define a Python function for ReComputing and invoke it
            e.g. (the lines are split into `List[str]`)

            ```
            def recompute(tensor_2222):
                tensor_3333 = torch.view(tensor_2222, size=[3,6,9])
                tensor_2222 = None      # no more reference
                return tensor_3333
            # in the beginning we have `import torch.utils.checkpoint as ckpt`
            tensor_4444 = ckpt.checkpoint(recompute, tensor_1111)            
            ```

            REMARK:
            -   In the example above, 'tensor_2222' can be released within the RC subgraph, which also means that
                the variable for this tensor can also be released within the enclosing graph, after the 'checkpoint' call.
            -   The generated RC subgraph will have no "free variables".
                All involved tensors that are defined outside of the RC group are made explicit inputs;
                All tensors, that are defined within the RC group and are referenced after RC subgraph ends, are made explicit outputs;
                And if a within-RC-group tensors are not used anymore, it's not returned.

            The fields storing intermediate codes that are populated by this method:
            - NONE
            """
            assert len(i_nodes) > 0
            node_codes = []

            nodes : List[IRCell] = [node for i, node in i_nodes]

            subseg = self.execplan.graph.segment(nodes)

            inputs = [t for t in subseg.inputs() if not t.is_attr()]
            input_names = [self.tensor_naming(t) for t in inputs]
            input_names_tuple = ', '.join(input_names)
            outputs = [t for t in subseg.outputs()]
            output_names = [self.tensor_naming(t) for t in outputs]
            output_names_tuple = ', '.join(output_names)

            # 'graph.segment(nodes)' ensures that if a tensor is no longer used (in RC group or in later code), 
            # it's not included in 'outputs'.
            # And we will not generate 'return' statement for it, since it will cause the error
            # that the variable is not defined (because it has been 'del'-ed).

            with FunctionBlock('recompute', input_names, False) as fb:
                # The nodes to recompute share the same space of line_ids (or "node ids") with non-recomputable nodes.
                # e.g. those ids in subgraphs are not 0-based, and incremented after the preceding non-rc nodes and so on.
                #
                # So within the recomputing subgraph, tensors can be released if they are no longer used
                # i.e. not returned by the 'def recompute(...)' 
                # since 'execplan.graph.segment(nodes)' will make all "free variables" as explicit inputs/outputs
                # to that subgraph.
                for ncode in emit_nodes_invocations(i_nodes, lifetime_by_line_id):
                    fb.insert_body(ncode)
                fb.insert_body(f'return {output_names_tuple}')
            node_codes += [''] + fb.code + ['']
            node_codes.append(
                f'{output_names_tuple} = ckpt.checkpoint(recompute, {input_names_tuple})'
            )

            return node_codes, inputs, outputs

        def get_equiv_recompute_gid(node:Union[IRFwOperation, IRAdapter]) -> Optional[int]:
            if isinstance(node, IRAdapter):
                # IRAdapter is equivalent to be non-recomputable. And it always terminates the
                # nodes sequence of any recomputing group before it.
                return None
            elif isinstance(node, IRFwOperation):
                return node.recompute
            else:
                raise ValueError(f'Unexcepted node type {type(node)}')

        def should_start_new_recompute_group(i_prev, i_cur) -> bool:
            # i_prev, i_cur: Tuple[int, Union[IRFwOp,IRAdapter]]
            prev_gid = get_equiv_recompute_gid(i_prev[1])
            cur_gid = get_equiv_recompute_gid(i_cur[1])
            return cur_gid != prev_gid

        nodes : List[IRCell] = segment.nodes()

        # After calculating the recompute groups, for each group, its input tensors' lifetime
        # should be extend to at least beyond the lifetime of that group.
        lifetime : Dict[IRTensor, int] = calc_tenvars_lifetime(nodes, segment.outputs(), segment.inputs())
        lifetime_by_line_id : Dict[int, List[IRTensor]] = dict()
        for tensor, line_id in lifetime.items():
            lifetime_by_line_id.setdefault(line_id, []).append(tensor)

        # more_itertools.split_when # type: (Iterable[T], (T,T)->bool) -> Iterator[List[T]]
        recompute_groups : List[List[Tuple[int, IRCell]]] \
            = list(split_when(enumerate(nodes), should_start_new_recompute_group))

        for rc_group in recompute_groups:
            # all FwOps/Adapters in a group have the same (equivalent) group id, 
            # check that of the head item, and 'rc_group' will not be empty here.
            gid : Optional[int] = get_equiv_recompute_gid(rc_group[0][1])
            if gid is None:
                codes += emit_nodes_invocations(rc_group, lifetime_by_line_id)
            else:
                assert len(rc_group) > 0

                # Step 1: when entering a RC group:
                #
                # We insert tensor releasing statement *before* emitting each node.
                # But here we are entering the scope of a RC group i.e. 'def recompute(...)'.
                # Any releasing before the first node of the RC group, 
                # should be done before and outside of the RC group.
                rc_first_line_id, _rc_first_node = rc_group[0]
                # ... and to avoid emitting again, 'pop' the lifetime record.
                # Specify the default collection since there might not be any.
                rel_tensors_before_rc : Optional[list] = lifetime_by_line_id.pop(rc_first_line_id, None)
                if rel_tensors_before_rc is not None:
                    codes.append(self.emit_tensors_release(rel_tensors_before_rc))

                # Step 2
                rc_codes, rc_inputs, rc_outputs = emit_rc_nodes(rc_group, lifetime_by_line_id)
                codes += rc_codes

                # Step 3: when exiting a RC group:
                #
                # `emit_rc_nodes` will not emit 'del`-statement for output tensors of the last
                # node in the RC group, since those tensors will be immediately released
                # as soon as 'recompute(...)' returns.
                # We need to remove those tensors from the linearized lifetime 
                # (namely those with lifetime 'rc_next_line_id')
                # and do not release them before the next node after the RC group.
                rc_last_line_id, _rc_last_node = rc_group[-1]
                rc_next_line_id = rc_last_line_id + 1
                lifetime_by_line_id.pop(rc_next_line_id, None) # specify a default to avoid KeyError
                
                # Step 4: after exiting a RC group:
                #
                # We need to release some argument tensors to the 'def recompute(...)' if they are
                # no longer used.
                # NOTE those tensors may have resulted in some 'del'-statements within the RC
                # subfunction. But we need to release them again in the enclosing function,
                # after the call to 'torch.checkpoint(recompute, *input_tensors)'.

                # Only release an RC input if:
                # - its lifetime does not exceed the lifetime of the RC group;
                # - not the case that the function returns after 'checkpoint' the RC subgraph.
                if rc_next_line_id != len(nodes):
                    inputs_to_rel = [rcin for rcin in rc_inputs if lifetime[rcin] <= rc_next_line_id]
                    if len(inputs_to_rel) > 0:
                        del_stmt = self.emit_tensors_release(inputs_to_rel)
                        codes.append(del_stmt)
                
                # any resultant tensors *defined within the RC group and not used after the group*
                # will not be returned from the generate 'def recompute(...)',
                # so here we have no resultant tensors (namely 'rc_outputs') to release.

        return codes

    def emit_op_code(self, node: IRFwOperation) -> List[str]:
        """
        Emit the statement to call the op in the forward code
        (e.g. in Segments, Adapter or CodeGen.Main)

        The result will look like (the lines are split into `List[str]`)
        ```
        tensor_3333 = torch.view(tensor_2222, [1,2,3,4,5])
        ```

        The fields storing intermediate codes that are populated by this method:
        -   NONE
        """
        codes = []
        # insert comment
        if node.comment is not None:
            codes.append(f'# {node.comment}')
        signature = node.signature
        inputs = [self.tensor_naming(t, prefix_attr='self.') for t in node.inputs()]
        kwargs = {}
        for key in node.kwargs:
            val = node.kwargs[key]
            if isinstance(val, str) and 'self.' not in val:
                val = '"' + val + '"'
            kwargs[key] = val

        emit_rule = Sign2EmitRule.map(signature)
        body = emit_rule(node, inputs, kwargs)

        if len(node.outputs()) == 0:
            code = body
        else:
            outputs = [self.tensor_naming(t) for t in node.outputs()]
            outputs = ', '.join(outputs)
            code = f'{outputs} = {body}'
        codes.append(code)
        return codes

    def emit_adapter_code(self, node: IRAdapter) -> List[str]:
        """
        Emit the statment of the adapter call
        
        The resultant `List[str]` will be lines of the statements of the final
        Python method for the targeted Segment, 
        without the method signature and the return statement.

        The fields storing intermediate codes that are populated by this method:
        -   NONE
        """
        codes = []
        assert len(node.device) == 1, f"Expected adapter to be dispatched:\n{node.extra_repr()}"
        prims = [node] if node.differentiable and node.custom else [prim for prim in node.prims]

        for prim in prims:
            if len(prim.inputs()) == 1:
                itensors = self.tensor_naming(prim.inputs()[0], prefix_attr='self.')
            else:
                itensors = self.tuple_naming(prim.inputs(), prefix_attr='self.')
            kwargs = self.kwargs_naming(**prim.kwargs)
            outputs = self.return_naming(prim.outputs())
            code = f'{outputs} = {prim.signature}({itensors}, {kwargs})'
            codes.append(code)
        return codes

    def emit_reducer_init(self, node: IRWeightReducer) -> None:
        """
        Emit code to initialize involved reducer objects in `__init__`.

        The fields storing intermediate codes that are populated by this method:
        -   `model_init_statements`
        """
        # reducer init interface
        reducer_init = '{reducer} = cube.runtime.adapter.Reducer(ranks={ranks})'
        reducer_add = 'self.add_reducer({reducer})'
        add_param = '{reducer}.add_param({weight})'
        # create reducer in declare region
        weights = node.inputs()
        reducer_name = f'self.wreducer{node._id}'
        self.model_init_statements.append('')
        init_code = reducer_init.format(reducer=reducer_name, ranks=node.device)
        self.model_init_statements.append(init_code)
        weights = [self.tensor_naming(t, prefix_attr='self.') for t in weights]
        for weight in weights:
            add_param_code = add_param.format(reducer=reducer_name, weight=weight)
            self.model_init_statements.append(add_param_code)
        add_code = reducer_add.format(reducer=reducer_name)
        self.model_init_statements.append(add_code)

    def emit_reducer_call(self, node: IRWeightReducer):
        """
        Emit the statment to invoke a reducer object.
        
        The fields storing intermediate codes that are populated by this method:
        -   NONE
        """
        reducer_name = f'self.wreducer{node._id}'
        code = f'{reducer_name}.allreduce()'
        return [code]

    def clear(self):
        """
        Clear buffer that used for generating code
        """
        # module init code
        self.model_init_statements: List[str] = list()
        # module forward code
        self.model_methods_bodies: List[List[str]] = list()
        # module member name
        self.symbols = SymbolTable()


class ScheduleCodeGen(CodeGen):

    def __init__(self, execplan: ExecutionPlan):
        super().__init__(execplan)
        # model full code
        self.init_code: List[str] = [
            '\n\n########## Generated Schedule Code ###########',
            'import torch', 'import cube', '']
        # module member name
        self.symbols = SymbolTable()

    def gen(self, device: int, outfile=None, attach=False) -> str:
        """
        Generate scheduling code based on the given sus
        """
        gencode = copy.copy(self.init_code)

        device_nodes = self.execplan.seq(device)

        lifetime : Dict[IRTensor, int] = calc_tenvars_lifetime(device_nodes, self.execplan.graph.outputs())
        lifetime_by_line_id : Dict[int, List[IRTensor]] = dict()
        for tensor, line_id in lifetime.items():
            lifetime_by_line_id.setdefault(line_id, []).append(tensor)

        with FunctionBlock(func_name='_train_step', 
                           args=['model', 'dataloader']) as fb:
            fb.insert_body('_ = None')
            # body code
            if len(device_nodes) == 0:
                fb.insert_body('pass')
            elif self.execplan.graph.sched:
                code = self.emit_schedule_plan(self.execplan.graph.sched, device)
                fb.insert_body(code)
            else:
                for i, node in enumerate(device_nodes):
                    # Decrement reference counts for output tensors that are no longer used
                    # Tensors here need to release *before* the i-th statement.
                    tensors : Optional[List[IRTensor]] = lifetime_by_line_id.get(i, None)
                    if tensors is not None: # not necessarily to have one after each line
                        fb.insert_body(self.emit_tensors_release(tensors))

                    name = self.node_naming(node)
                    code = self.emit_node(node, name=name)
                    fb.insert_body(code)
            
            # return code
            outputs = self.return_naming(self.execplan.graph.outputs())
            code = f'return {outputs}'
            fb.insert_body(code)
        gencode += fb.code
        gencode += ['']

        code = '\n'.join(gencode)
        # write to file
        if outfile:
            with open(outfile, 'a' if attach else 'w') as f:
                f.write(code)
        return code

    def emit_schedule_plan(self, schedplan: IRScheduleStrategy, devid: int):
        signature = schedplan.signature
        kwargs: Dict[str, Any] = schedplan.kwargs(devid)
        strkwargs = dict()
        for kwarg, val in kwargs.items():
            name = str(val) if not isinstance(val, IRCell) else 'model.'+self.node_naming(val)
            strkwargs[kwarg] = name
        code = ', '.join(f'{kwarg}={name}' for kwarg, name in strkwargs.items())
        code = f'{signature}({code})'
        return code

    def emit_node(self, node: IRCell, name: str) -> str:
        """
        Emit node / subgraph code
        """
        fsign = '{outputs} = cube.runtime.executor.fexecute({model}, *{inputs}, requires_grad={req_grad})'
        bsign = '{input_grads} = cube.runtime.executor.backward({input_tensors}, {output_tensors}, {output_grads})'
        
        inputs = self.tuple_naming(node.inputs(), skip_attr=True, prefix_attr='model.')
        outputs = self.return_naming(node.outputs(), skip_attr=True, prefix_attr='model.')
        req_grad = any(t.requires_grad for t in node.outputs() if isinstance(t, IRTensor))

        if isinstance(node, IRSegment):
            # emit forward
            if node.isfw():
                code = fsign.format(
                    outputs = outputs,
                    model = f'model.{name}',
                    inputs = inputs,
                    req_grad = req_grad
                )
            # emit backward
            else:
                input_tensors, output_tensors, output_grads, input_grads = \
                    get_backward_callsite_io_tensors(node)
                
                for idx, tensor in enumerate(output_grads):
                    if isinstance(tensor, float):
                        assert tensor == 1.0, "Loss gradient should be 1.0"
                        output_grads[idx] = None
                code = bsign.format(
                    input_grads = self.return_naming(input_grads),
                    input_tensors = self.tuple_naming(input_tensors, skip_attr=True, prefix_attr='model.'),
                    output_tensors = self.tuple_naming(output_tensors, skip_attr=True, prefix_attr='model.'),
                    output_grads = self.tuple_naming(output_grads, skip_attr=True, prefix_attr='model.')
                )

        elif isinstance(node, IRDataOperation):
            if len(node.inputs()) != 0:
                raise RuntimeError("Expect Dataloader node has no inputs")
            outputs = [self.tensor_naming(output) for output in node.outputs()]
            outputs = self.return_naming(outputs)
            code = f'{outputs} = next(dataloader)'

        elif isinstance(node, IRAdapter):
            code = fsign.format(
                outputs = outputs,
                model = f'model.{name}',
                inputs = inputs,
                req_grad = req_grad
            )

        elif isinstance(node, IRWeightReducer):
            code = fsign.format(
                outputs = outputs,
                model=f'model.{name}',
                inputs='()',
                req_grad=req_grad
            )

        else:
            raise RuntimeError(f"Unspported node type: {type(node)}")
        return code
