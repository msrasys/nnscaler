from typing import Dict, List, Optional, Tuple
from more_itertools import split_when
import warnings
import copy
import torch

from cube.ir.cten import IRCell
from cube.ir.tensor import IRSubTensor
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation
from cube.ir.adapter import IRWeightReducer, IRAdapter
from cube.ir.adapter.prim import CollectivePrim

from cube.graph.graph import IRSegment
from cube.graph.parser.mapping import Sign2Op

from cube.execplan import ExecutionPlan
from cube.execplan.execplan import ExeRepetend, ExeReuseCell

from cube.codegen.syntax.symtable import SymbolTable
from cube.codegen.syntax.blocks import ClassBlock, FunctionBlock

from cube.codegen.emit import FuncEmission
from cube.codegen.module.autograd import AutogradAdapterCodeGen
from cube.codegen.lifecycle import LifeCycle

from cube.flags import CompileFlag


class ModuleCodeGen(FuncEmission):
    """
    Generate module code

    `ModuleCodeGen` traverses all IR nodes and categorizes their intermediately generated
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

    def __init__(self, execplan: ExecutionPlan) -> None:
        super().__init__()
        self.execplan: ExecutionPlan = execplan

        self.init_code: List[str] = [
            '\n\n########## Generated Model Code ###########',
            'from typing import *',
            'import torch', 'import torch.utils.checkpoint as ckpt',
            'import cube', 'import _operator', 'from numpy import inf', '', '']
        
        if CompileFlag.use_nnfusion:
            self.init_code.extend(['import nnfusion', ''])

        # customized op code
        for _, op_impl in Sign2Op.kOpCodeDef.items():
            # self.init_code.append('@torch.jit.script')
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
        # batch size
        self.batch_size = None

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
                if adapter.isfw() and adapter.differentiable and adapter.custom:
                    gencode += AutogradAdapterCodeGen().gen(adapter) + ['', '']
                    adapter.signature = AutogradAdapterCodeGen.name(adapter) + '.apply'

        # initialize communication groups
        self.init_comm_groups()

        # parse graph body
        unrolled_seqs = []

        for node in self.execplan.seq(device):
            # unwrap from ExeReuseCell and ExeRepetend
            if isinstance(node, ExeReuseCell):
                node = node.cell
            if isinstance(node, ExeRepetend):
                for node in node.nodes():
                    if isinstance(node, ExeReuseCell):
                        node = node.cell
                    unrolled_seqs.append(node)
            else:
                unrolled_seqs.append(node)
        # we use ordered dict as ordered set
        unrolled_seqs = tuple(dict.fromkeys(unrolled_seqs))
        # emit code
        for node in unrolled_seqs:
            if isinstance(node, IRSegment):
                if not node.isfw(): continue  # skip backward segment
                codes = self.emit_segment(node)
            elif isinstance(node, IRFwOperation):
                raise RuntimeError(f"Unexcepted global-level op call: {node}")
            elif isinstance(node, IRAdapter):
                codes = self.emit_adapter(node, prefix_attr='self.')
            elif isinstance(node, IRWeightReducer):
                self.init_reducer(node)
                codes = self.emit_reducer(node)
            elif isinstance(node, IRBpOperation):
                continue
            elif isinstance(node, IRDataOperation):
                self.init_batchsize(node)
                continue
            else:
                raise RuntimeError(f"Un-recognized IRCell type: {type(node)}")

            # emit node tensor declaration into `__init__`
            # typically it's about the `nn.Parameter`
            self.init_attributes(node)

            # emit node code
            # codes : List[str]
            self.model_methods_bodies.append(codes)
            gen_nodes.append(node)

            args = list()
            for t in node.inputs():
                if isinstance(t, IRSubTensor):
                    if not t.is_attr():
                        args.append(ModuleCodeGen.tensor_name(t))
                else:
                    args.append(ModuleCodeGen.tensor_name(t))
            node_args.append(args)

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
                name = ModuleCodeGen.node_name(node)
                input_args = ['self'] + node_args[idx]
                forward_code = self.model_methods_bodies[idx]

                with FunctionBlock(func_name=name, args=input_args) as fb:
                    fb.insert_body(forward_code)
                    # generate output
                    outputs = [ModuleCodeGen.tensor_name(t) for t in node.outputs()]
                    return_code = f"return {', '.join(outputs)}"
                    fb.insert_body(return_code)
                cb.insert_body('')
                if CompileFlag.use_nnfusion and name.startswith('segment'):
                    cb.insert_body('@nnfusion.jit')
                if CompileFlag.use_jit and name.startswith('segment'):
                    cb.insert_body('@torch.jit.script_method')
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

    def init_attributes(self, node: IRCell):
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
                name = ModuleCodeGen.tensor_name(itensor, prefix_attr='self.')
                if isinstance(itensor, IRSubTensor):
                    if itensor.is_attr() and not self.symbols.exist(name):
                        self.symbols.create(name)
                        sign = psign if itensor.is_param() else bsign
                        code = sign.format(
                            name=ModuleCodeGen.tensor_name(itensor),
                            shape=tuple(itensor.shape),
                            dtype=self.dtype_map(itensor.dtype)
                        )
                        self.model_init_statements.append(code)
                        tid = itensor.parent.tid
                        slicers = tuple(slice(start, stop) for (start, stop) in itensor.indmap)
                        val_chunks = itensor.valmap[1]
                        code = map_sign.format(
                            attr=ModuleCodeGen.tensor_name(itensor), tid=tid,
                            slicers=str(slicers), val_chunks=val_chunks
                        )
                        self.model_init_statements.append(code)
                        self.model_init_statements.append('')
                if isinstance(itensor, str):
                    if name.startswith('self.'):
                        if not hasattr(self._ref_module, name[5:]):
                            raise NotImplementedError("member attribute is not added")
            for output in node.outputs():
                self.symbols.create(ModuleCodeGen.tensor_name(output, prefix_attr='self.'))
        else:
            for sub_node in node.nodes():
                self.init_attributes(sub_node)
        return

    def init_reducer(self, node: IRWeightReducer) -> None:
        """
        Emit code to initialize involved reducer objects in `__init__`.

        The fields storing intermediate codes that are populated by this method:
        -   `model_init_statements`
        """
        max_nbytes = CompileFlag.max_reducer_bucket
        # reducer init interface
        reducer_init = '{reducer} = cube.runtime.adapter.Reducer(ranks={ranks}, max_bucket_size_bytes={max_nbytes})'
        reducer_add = 'self.add_reducer({reducer})'
        add_param = '{reducer}.add_param({weight})'
        # create reducer in declare region
        weights = node.inputs()
        reducer_name = f'self.wreducer{node._id}'
        self.model_init_statements.append('')
        init_code = reducer_init.format(reducer=reducer_name, ranks=node.device, max_nbytes=max_nbytes)
        self.model_init_statements.append(init_code)
        weights = [ModuleCodeGen.tensor_name(t, prefix_attr='self.') for t in weights]
        for weight in weights:
            add_param_code = add_param.format(reducer=reducer_name, weight=weight)
            self.model_init_statements.append(add_param_code)
        add_code = reducer_add.format(reducer=reducer_name)
        self.model_init_statements.append(add_code)

    def init_batchsize(self, node: IRDataOperation):
        """
        Emit batch size declare
        """
        signature = 'self.set_batch_size({bs})'
        bs = [t.shape[dim] for t, dim in zip(node.outputs(), node.get_batch_dims()) if dim is not None]
        bs = set(bs)
        if len(bs) > 1:
            warnings.warn(f'Find Heterogenous batch size {bs}. Keep output to be same with semantic dataloder.')
        bs = list(bs)[0] if len(bs) == 1 else None
        assert self.batch_size is None or self.batch_size == bs, f"Not match for batch size: {self.batch_size} != {bs}"
        self.model_init_statements.append(signature.format(bs=bs))
        self.model_init_statements.append('')
        self.batch_size = bs

    @staticmethod
    def emit_segment(segment: IRSegment) -> List[str]:
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
        nodes : List[IRCell] = segment.nodes()
        lifetime = LifeCycle(nodes, segment.inputs(), segment.outputs())
        rc_groups: List[List[IRCell]] = list(
            split_when(nodes, lambda prev, curr: prev.recompute != curr.recompute))

        codes: List[str] = []
        for rc_group in rc_groups:
            assert len(rc_group) > 0
            gid: Optional[int] = rc_group[0].recompute
            if gid is None:
                codes += ModuleCodeGen._emit_nodes(rc_group, lifetime)
            else:
                # get recompute excution code
                rc_segment = segment.create_segment(rc_group)
                rc_codes = ModuleCodeGen._emit_recompute(rc_group,
                    rc_segment.inputs(), rc_segment.outputs(), lifetime)
                codes += rc_codes
                # release input tensors after exiting a RC group:
                last_node = rc_group[-1]
                line = lifetime.get_line(last_node)
                if last_node != nodes[-1]: # skip if it is the last node
                    inputs_to_rel = [t for t in rc_segment.inputs() if lifetime.releasable_after_line(t, line)]
                    if len(inputs_to_rel) > 0:
                        del_stmt = ModuleCodeGen.emit_release(inputs_to_rel)
                        codes.append(del_stmt)

        return codes

    @staticmethod
    def _emit_nodes(nodes: List[IRCell], lifecycle: LifeCycle) -> List[str]:
        """
        Emit code to invoke operations and adapter, 
        e.g. (the lines are split into `List[str]`)

        ```
        tensor_2222 = torch.view(tensor_1111, size=[3,6,9])
        del tensor_1111    # if no more reference
        tensor_3333 = cube.runtime.adapter.allgather_reducescatter(tensor_2222, dim=1, rank=[0,1])
        del tensor_2222    # if no more reference
        ```

        The fields storing intermediate codes that are populated by this method:
        - NONE
        """
        node_codes = []
        for node in nodes:
            # execute
            if isinstance(node, IRFwOperation):
                code = ModuleCodeGen.emit_fnode(node, prefix_attr='self.')
                node_codes += code
            elif isinstance(node, IRAdapter):
                code = ModuleCodeGen.emit_adapter(node)
                node_codes += code
            else:
                raise RuntimeError(f"unexpected type {type(node)} in IRSegment")
            # release
            tensors_to_del = lifecycle.release_tensors_after_node(node)
            if len(tensors_to_del) > 0:
                node_codes.append(FuncEmission.emit_release(tensors_to_del))

        return node_codes

    @staticmethod
    def _emit_recompute(nodes: Tuple[IRCell], inputs: List[IRSubTensor], outputs: List[IRSubTensor],
                        lifecycle: LifeCycle) -> List[str]:
        """
        Emit code to define a Python function for Recomputing and invoke it
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

        @return codes List[str]
        """
        assert len(nodes) > 0

        inputs = [t for t in inputs if not t.is_attr()]
        input_names = [FuncEmission.tensor_name(t) for t in inputs]
        input_names_tuple = ', '.join(input_names)
        output_names = [FuncEmission.tensor_name(t) for t in outputs]
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

            # for ncode in ModuleCodeGen._emit_nodes(nodes, lifecycle):
            #     fb.insert_body(ncode)
            fb.insert_body(ModuleCodeGen._emit_nodes(nodes, lifecycle))
            fb.insert_body(f'return {output_names_tuple}')
        codes = [''] + fb.code + ['']
        codes.append(
            f'{output_names_tuple} = ckpt.checkpoint(recompute, {input_names_tuple})'
        )

        return codes

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
        # batch size
        self.batch_size = None
