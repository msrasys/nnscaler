#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Iterable, Dict, List, Any
import itertools

from nnscaler.flags import CompileFlag
from nnscaler.ir.cten import IRCell, IRTensor, IRObject
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import IdentityPrim
from nnscaler.graph.segment import IRSegment
from nnscaler.execplan.execplan import ExeReuseCell

from nnscaler.codegen.emit import FuncEmission


class LifeCycle:

    def __init__(self, nodes: List[IRCell], graph_inputs: List[Any], graph_outputs: List[Any]):

        graph_inputs = IRSegment.get_objects_from_complex(graph_inputs)
        graph_outputs = IRSegment.get_objects_from_complex(graph_outputs)
        func_emission = FuncEmission()

        self.nodes: Dict[IRCell, int] = {node: lid for lid, node in enumerate(nodes)}
        # the last line id of consuming or producing a tensor
        self.lifetime: Dict[IRObject, int] = {}
        # the tensors can be released given the finish of line id
        self.release: Dict[int, List[IRObject]] = {}
        # forward segment outputs whose data has left the device after their last
        # forward use (all consumers are sends) but which stay alive only as roots
        # of a later backward call; keyed by that last forward line. Their storage
        # can be dropped there while the tensor object (and autograd graph) is kept.
        self.release_storage: Dict[int, List[IRSubTensor]] = {}
        # the last line id of forward activations
        fw_lifetime: Dict[IRObject, int] = {}
        releasable: List[IRSubTensor] = []
        # tensors consumed by a forward node that may keep or alias them locally,
        # i.e. anything but an adapter without local outputs (a pure send).
        kept_locally = set()

        # FIXME: consider the case of IRObject in the kwargs of IRFwOperation
        # is_activation = lambda t: isinstance(t, IRObject) and not t.is_attr()
        is_activation = lambda t: isinstance(t, IRSubTensor) and not t.is_attr()

        self.lifetime.update((tsin, 0) for tsin in graph_inputs if is_activation(tsin))

        for i, node in enumerate(nodes):

            outputs : Iterable[IRObject]
            inputs : Iterable[IRObject]

            if isinstance(node, (IRSegment, ExeReuseCell)):
                # this is only for scheduler to track the lifetime of tensors
                # for module code generation, no IRSegment/ExeReuseCell will be used.
                # so will not go into this branch.

                # forward segment/adapter/dataloader
                if node.isfw():
                    outputs = node.outputs()
                    inputs = node.inputs()
                    cell = node.cell if isinstance(node, ExeReuseCell) else node
                    if CompileFlag.pipeline_deallocate_outputs and isinstance(cell, IRSegment):
                        # ExeReuseCell outputs are per-micro-batch copies of the segment
                        # outputs with the same nesting, so flattening both aligns them.
                        tensors = IRSegment.get_objects_from_complex(outputs)
                        cell_tensors = IRSegment.get_objects_from_complex(cell.outputs())
                        assert len(tensors) == len(cell_tensors), f'output mismatch: {node} vs {cell}'
                        releasable.extend(
                            t for t, cell_t in zip(tensors, cell_tensors)
                            if is_activation(t) and self._is_releasable_output(cell, cell_t)
                        )
                # backward segment
                else:
                    # in `_train_step`, we will explicitly call backward to generate gradients.
                    # When pipeline is enabled, there will be multiple backward calls in the same segment.
                    # So we also need to track the temporary gradient tensors
                    # and delete them after the backward call to save memory.
                    fw_inputs, fw_outputs, output_grads, input_grads = \
                        func_emission.get_backward_callsite_io_tensors(node)
                    # remove loss gradient
                    output_grads = [t for t in output_grads if not t.is_loss()]

                    outputs = input_grads
                    inputs = list(itertools.chain(fw_inputs, fw_outputs, output_grads))
            else:
                outputs = node.outputs()
                inputs = node.inputs()

            # aggressively mark all outputs for immediate deletion,
            # namely *after* 'i'-th statement, in case it's never used.
            self.lifetime.update((tout, i) for tout in IRSegment.get_objects_from_complex(outputs) if is_activation(tout))

            # "fast-forward" all inputs to the current statement, namely after 'i'-th node.
            self.lifetime.update((tin, i) for tin in IRSegment.get_objects_from_complex(inputs) if is_activation(tin))

            if CompileFlag.pipeline_deallocate_outputs and node.isfw():
                fw_lifetime.update((t, i) for t in IRSegment.get_objects_from_complex(list(inputs) + list(outputs)) if is_activation(t))
                cell = node.cell if isinstance(node, ExeReuseCell) else node
                # A pure send is a generated adapter without local outputs. A custom
                # differentiable adapter runs user autograd code that may save its input.
                # TODO: adapters with local outputs could also qualify when every prim
                # allocates a fresh output (all_gather, all_to_all, reduce_scatter, ...),
                # i.e. no IdentityPrim and not differentiable. Chunk only yields a view,
                # so releasing its input frees nothing. Kept conservative for now.
                is_send = (isinstance(cell, IRAdapter) and len(outputs) == 0
                           and not (cell.differentiable and cell.custom))
                if not is_send:
                    kept_locally.update(t for t in IRSegment.get_objects_from_complex(inputs) if is_activation(t))


        # Here (i+1) is always greater than 'len(nodes)'
        # Generally we don't manually release those tensors since the enclosing function is about to
        # return, all local variables are automatically released.
        # But we do need to update the lifetime of all outputs, to avoid early releasing.
        self.lifetime.update((tsout, len(nodes)) for tsout in graph_outputs if is_activation(tsout))

        for tensor, line_id in self.lifetime.items():
            self.release.setdefault(line_id, []).append(tensor)

        if CompileFlag.pipeline_deallocate_outputs:
            for tensor in releasable:
                if tensor in graph_outputs or tensor in kept_locally:
                    continue
                if fw_lifetime[tensor] < self.lifetime[tensor]:
                    self.release_storage.setdefault(fw_lifetime[tensor], []).append(tensor)

    @staticmethod
    def _is_releasable_output(segment: IRSegment, output: IRSubTensor) -> bool:
        """
        Whether a forward segment output may own its runtime object. identity/
        multiref/IdentityPrim exist to fan a tensor out to several consumers, so
        their outputs are the same Python object as a sibling that may still be
        consumed later in forward; dropping one would drop the other, and the
        the sibling may even be a graph output. Other producers may also return
        their input (in-place ops, no-op `contiguous`/`to`) or a view of it, but
        then any remaining holder of that object shows up as an extra Python
        reference or in its impl use count, which `Executor.release_storage`
        checks before swapping the data; the ops above are excluded here only
        because they are fan-outs by design and never worth the runtime check.
        """
        producers = segment.producers(output.parent)
        if len(producers) != 1:
            return False
        producer = producers[0]
        if isinstance(producer, IRAdapter):
            return not any(isinstance(prim, IdentityPrim) for prim in producer.prims)
        return isinstance(producer, IRFwOperation) and producer.name not in ('identity', 'multiref')

    def release_storage_after_line(self, line_id: int) -> List[IRSubTensor]:
        """
        Get the forward outputs whose storage can be dropped after `line_id`
        while keeping the tensors as backward roots.
        """
        return self.release_storage.get(line_id, [])

    def release_tensors_after_line(self, line_id: int) -> List[IRSubTensor]:
        """
        Get the releasable IRSubTensors after finish of executing of `line_id`.

        @param line_id int

        @return tensors List[IRSubTensors]: tensors that can be released.
        """
        return self.release.get(line_id, [])

    def release_tensors_after_node(self, node: IRCell) -> List[IRSubTensor]:
        """
        Get the releasable IRSubTensors after finish of executing of the node.

        @param line_id int

        @return tensors List[IRSubTensors]: tensors that can be released.
        """
        assert node in self.nodes
        line_id = self.nodes[node]
        return self.release.get(line_id, [])

    def releasable_after_node(self, obj: IRObject, node: IRCell) -> bool:
        """
        Check if the tensor is releasable after executing the node.
        Please note that if it is not a IRSubTensor(is IRObject),
            we will never manually release it.

        Args:
            tensor (IRObject): the tensor to be checked
            node (IRCell): the node to be checked
        Returns:
            releasable (bool): whether the tensor is releasable after executing
        """
        if not isinstance(obj, IRSubTensor):
            return False

        assert node in self.nodes
        assert obj in self.lifetime
        line_id = self.nodes[node]
        return self.lifetime[obj] < line_id

    def releasable_after_line(self, obj: IRObject, line: int) -> bool:
        """
        Check if the tensor is releasable after executing the node.
        Please note that if it is not a IRSubTensor(is IRObject),
            we will never manually release it.

        Args:
            tensor (IRObject): the tensor to be checked
            line (int): the line to be checked
        Returns:
            releasable (bool): whether the tensor is releasable after specific line.
        """
        if not isinstance(obj, IRSubTensor):
            return False

        assert obj in self.lifetime
        return self.lifetime[obj] < line

    def get_line(self, node: IRCell) -> int:
        """
        Get line id of the node
        """
        return self.nodes[node]
