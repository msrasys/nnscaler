"""
Generate Pytorch code given the model DAG and the transformation config
"""
import itertools
from typing import Dict, Generator, Iterable, List, Any, Optional, Set, Tuple, Union
import torch
import copy
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


# TODO this could be applied in all codegen: Segments, Adapters, Schedules...
# TODO but Schedules don't have standalone inputs, while Segments and Adapters have and 
#      those inputs may need to be released too.
def calc_tenvars_lifetime(
        nodes:Iterable[IRCell], subgraph_outputs:Iterable[IRTensor]
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
        i.e. the variable for tensor `t` could be released *AFTER* the `i`-th statement 
        in codegen.

        If a tensor is not included in the dict, it means that tensor is never referenced.

        TODO If an input of the subgraph is never used, its corresponding `i` is `-1`.
    
    REMARK:
    
        We cannot `detele O_j` because it may be a variable alias and the tensor object
        behind is still active (e.g. `runtime.multiref`).
        So we just set it to `None`, decrement the reference count 
        and let Python (the codegen target) decide the deletion.
    """

    lifetime : Dict[IRTensor, int] = dict()

    def is_temp_tensor(v):
        return isinstance(v, IRSubTensor) and not v.is_param()

    #lifetime.update((tsin, -1) for tsin in subgraph_inputs if is_temp_tensor(tsin))

    for i, node in enumerate(nodes):
        # aggressively mark all outputs for immediate deletion, namely 'i'
        # TODO should work fine even for IRBpOperation
        lifetime.update((tout, i) for tout in node.outputs() if is_temp_tensor(tout))

        # "fast-forward" all inputs to the current statement, namely 'i'
        inputs : Iterable[IRTensor]

        if isinstance(node, IRSegment):
            if node.forward:
                inputs = node.inputs()
            else:
                # NOTE
                # An 'IRBpOperation' does not explicitly record all tensors that are
                # inputs-and-outputs-to-its-correspondeding-autograd.grad-call after codegen,
                #
                # E.g.
                # IRBpOperation bp_op{ pair=fw_op } has:
                # ```
                # len(bp_op.inputs())==len(fw_op.outputs())
                # len(bp_op.outputs())==len(fw_op.inputs())
                # ````
                #
                # while a call to 'torch.autograd.grad' in Python is like:
                # ```
                # grad_inputs : Tuple[torch.Tensor, ...] = torch.autograd.grad(outputs, inputs, grad_outputs)
                # len(grad_inputs) == len(inputs)
                # len(grad_outputs) == len(outputs)
                # ```
                # in other words, if we simply treat `autograd.grad` as an `g_op:IRCell`, it has:
                # ```
                # len(g_op.inputs()) == len(fw_op.outputs())*2 + len(fw_op.inputs())
                # len(g_op.outputs()) == len(fw_op.inputs())
                # ```
                #
                fw_inputs : tuple = node.mirror.inputs()
                fw_outputs : tuple = node.mirror.outputs()
                grad_inputs : Generator = (t.grad for t in fw_outputs)
                
                inputs = itertools.chain(fw_inputs, fw_outputs, grad_inputs)

        else:
            inputs = node.inputs()

        lifetime.update((tin, i) for tin in inputs if is_temp_tensor(tin))

    i += 1
    lifetime.update((tsout, i) for tsout in subgraph_outputs if is_temp_tensor(tsout))

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
            if prefix_attr is not None and tensor.is_param():
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
            if isinstance(t, IRTensor) and skip_attr and t.is_param():
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
            if isinstance(t, IRTensor) and skip_attr and t.is_param():
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
    """

    def __init__(self, execplan: ExecutionPlan):
        super().__init__(execplan)
        # model full code
        self.init_code: List[str] = [
            '\n\n########## Generated Model Code ###########',
            'import torch', 'import torch.utils.checkpoint as ckpt',
            'import cube', '', '']
        # customized op code
        for _, op_impl in Sign2Op.kOpCodeDef.items():
            self.init_code.append(op_impl)
            self.init_code += ['']
        # module init code
        self.declare_region: List[str] = list()
        # module forward code
        self.forward_region_units: List[List[str]] = list()
        self.forward_region: List[str] = list()
        # module member name
        self.symbols = SymbolTable()
        # ref module to check shared variables
        self._ref_module = torch.nn.Module()

    def init_comm_groups(self):
        """
        Get all communication groups.

        Creating communication group requires all the devices
        enter the same call.
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
        adapters = [n for n in graph.flatten() if isinstance(n, IRAdapter)]
        for adapter in adapters:
            for prim in adapter.prims:
                if isinstance(prim, CollectivePrim):
                    ranks = list(prim.kwargs['ranks'])
                    ranks.sort()
                    ranks = tuple(ranks)
                    if ranks not in comm_groups:
                        comm_groups.append(ranks)
        # create communication group
        self.declare_region.append('# communication groups')
        for ranks in comm_groups:
            code = sign.format(ranks=list(ranks))
            self.declare_region.append(code)
        self.declare_region.append(' ')

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
                if not node.forward: continue  # skip backward segment
                codes = self.emit_segment_code(node)
            elif isinstance(node, IRFwOperation):
                codes = self.emit_op_code(node)
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
            self.forward_region += codes
            # emit node tensor declaration
            self.emit_node_declare(node)
            # emit node code
            self.forward_region_units.append(self.forward_region)
            self.forward_region = list()
            gen_nodes.append(node)
            args = list()
            for t in node.inputs():
                if isinstance(t, IRSubTensor):
                    if not t.is_param():
                        args.append(self.tensor_naming(t))
                else:
                    args.append(self.tensor_naming(t))
            node_args.append(args)

        # generate full code
        with ClassBlock(class_name='GenModel', derived=['cube.runtime.module.CubeModule']) as cb:
            with FunctionBlock(func_name='__init__', args=['self']) as ib:
                ib.insert_body(self.declare_region)
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
                forward_code = self.forward_region_units[idx]
                with FunctionBlock(func_name=name, args=input_args) as fb:
                    fb.insert_body(forward_code)
                    # generate output
                    outputs = [self.tensor_naming(t) for t in node.outputs()]
                    return_code = f"return {', '.join(outputs)}"
                    fb.insert_body(return_code)
                cb.insert_body('')
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

    def emit_node_declare(self, node: IRCell):
        """
        Emit tensor declaration code
        """
        sign = 'torch.nn.Parameter(torch.empty({shape}, dtype={dtype}))'
        map_sign = "self.add_full_map('{attr}', {tid}, {slicers}, {val_chunks})"
        for itensor in node.inputs():
            name = self.tensor_naming(itensor, prefix_attr='self.')
            if isinstance(itensor, IRSubTensor):
                if itensor.is_param() and not self.symbols.exist(name):
                    self.symbols.create(name)
                    code = f'{name} = {sign.format(shape=tuple(itensor.shape), dtype=self.dtype_map(itensor.dtype))}'
                    self.declare_region.append(code)
                    tid = itensor.parent.tid
                    slicers = tuple(slice(start, stop) for (start, stop) in itensor.indmap)
                    val_chunks = itensor.valmap[1]
                    code = map_sign.format(
                        attr=self.tensor_naming(itensor), tid=tid,
                        slicers=str(slicers), val_chunks=val_chunks
                    )
                    self.declare_region.append(code)
            if isinstance(itensor, str):
                if name.startswith('self.'):
                    if not hasattr(self._ref_module, name[5:]):
                        raise NotImplementedError("member attribute is not added")
        for output in node.outputs():
            self.symbols.create(self.tensor_naming(output, prefix_attr='self.'))
        return

    def emit_segment_code(self, node: IRSegment) -> List[str]:
        """
        Emit IRSegment code

        Nodes in the segment will group into recompute region
        """

        codes = []
                    
        def emit_nodes(nodes: List[IRCell]) -> List[str]:
            node_codes = []
            for node in nodes:
                if isinstance(node, IRFwOperation):
                    code = self.emit_op_code(node)
                elif isinstance(node, IRAdapter):
                    code = self.emit_adapter_code(node)
                else:
                    raise RuntimeError(f"unexpected type {type(node)} in IRSegment")
                node_codes += code
            return node_codes

        def emit_rc_nodes(nodes: List[IRCell]) -> List[str]:
            node_codes = []
            subseg = self.execplan.graph.segment(nodes)
            inputs = [self.tensor_naming(t) for t in subseg.inputs() if not t.is_param()]
            inputs_tup = ', '.join(inputs)
            outputs = [self.tensor_naming(t) for t in subseg.outputs()]
            outputs = ', '.join(outputs)
            with FunctionBlock('recompute', inputs, False) as fb:
                for ncode in emit_nodes(nodes):
                    fb.insert_body(ncode)
                fb.insert_body(f'return {outputs}')
            node_codes += [''] + fb.code + ['']
            node_codes.append(
                f'{outputs} = ckpt.checkpoint(recompute, {inputs_tup})'
            )
            return node_codes

        # to recompute region
        curr_recompute_gid = None
        curr_nodes = []
        for node in node.nodes():
            if isinstance(node, IRFwOperation):
                if node.recompute != curr_recompute_gid:
                    if len(curr_nodes) != 0:
                        if curr_recompute_gid is None:
                            codes += emit_nodes(curr_nodes)
                        else:
                            codes += emit_rc_nodes(curr_nodes)
                    curr_recompute_gid = node.recompute
                    curr_nodes = [node]
                else:
                    curr_nodes.append(node)
            elif isinstance(node, IRAdapter):
                # strategy 1: recompute close before adapter communication
                if curr_recompute_gid is None:
                    curr_nodes.append(node)
                else:
                    if len(curr_nodes) != 0:
                        if curr_recompute_gid is None:
                            codes += emit_nodes()
                        else:
                            codes += emit_rc_nodes()
                    else:
                        curr_recompute_gid = None
                        curr_nodes = [node]

        if len(curr_nodes) != 0:
            if curr_recompute_gid is None:
                codes += emit_nodes(curr_nodes)
            else:
                codes += emit_rc_nodes(curr_nodes)

        return codes

    def emit_op_code(self, node: IRFwOperation) -> List[str]:
        """
        Emit op forward code
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
        Emit adapter call
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

    def emit_reducer_init(self, node: IRWeightReducer):
        # reducer init interface
        reducer_init = '{reducer} = cube.runtime.adapter.Reducer(ranks={ranks})'
        reducer_add = 'self.add_reducer({reducer})'
        add_param = '{reducer}.add_param({weight})'
        # create reducer in declare region
        weights = node.inputs()
        reducer_name = f'self.wreducer{node._id}'
        self.declare_region.append('')
        init_code = reducer_init.format(reducer=reducer_name, ranks=node.device)
        self.declare_region.append(init_code)
        weights = [self.tensor_naming(t, prefix_attr='self.') for t in weights]
        for weight in weights:
            add_param_code = add_param.format(reducer=reducer_name, weight=weight)
            self.declare_region.append(add_param_code)
        add_code = reducer_add.format(reducer=reducer_name)
        self.declare_region.append(add_code)

    def emit_reducer_call(self, node: IRWeightReducer):
        reducer_name = f'self.wreducer{node._id}'
        code = f'{reducer_name}.allreduce()'
        return [code]

    def clear(self):
        """
        Clear buffer that used for generating code
        """
        # module init code
        self.declare_region: List[str] = list()
        # module forward code
        self.forward_region_units: List[List[str]] = list()
        self.forward_region: List[str] = list()
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
            elif self.execplan.graph.schedule_plan:
                code = self.emit_schedule_plan(self.execplan.graph.schedule_plan, device)
                fb.insert_body(code)
            else:
                for i, node in enumerate(device_nodes):
                    name = self.node_naming(node)
                    code = self.emit_node(node, name=name)
                    fb.insert_body(code)

                    # decrement reference counts for output tensors that are no longer used
                    tensors : Optional[List[IRTensor]] = lifetime_by_line_id.get(i, None)
                    if tensors is not None: # not necessarily to have one after each line
                        tnames : Generator = (self.tensor_naming(t) for t in tensors)
                        fb.insert_body(', '.join(tnames) + ' = ' + ', '.join(['None'] * len(tensors)))
            
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
            if node.forward:
                code = fsign.format(
                    outputs = outputs,
                    model = f'model.{name}',
                    inputs = inputs,
                    req_grad = req_grad
                )
            # emit backward
            else:
                input_tensors = [t for t in node.mirror.inputs() if \
                    isinstance(t, IRSubTensor) and \
                    t.requires_grad and \
                    not t.is_param()
                ]
                output_tensors = [t for t in node.mirror.outputs() if isinstance(t, IRSubTensor)]
                input_grads = [t.grad for t in input_tensors]
                output_grads = [t.grad for t in output_tensors]
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
