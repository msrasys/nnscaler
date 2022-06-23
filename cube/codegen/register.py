from typing import Any, Dict, List, Union

from cube.ir.cten import IRTensor


class VarManager:
    """
    Tensor naming reuse engine for saving memory
    """

    def __init__(self):
        # the unique id
        self.nid = 0
        self.slots: List[int] = list()
        # original tensor id -> new tensor id mapping
        self.tmap: Dict[int, int] = dict()

    def free(self, tensor: Union[IRTensor, Any]):
        """
        Free a tensor
        """
        if isinstance(tensor, IRTensor):
            assert tensor._id in self.tmap, f"Double free on tensor {tensor}"
            reg = self.tmap[tensor._id]
            del self.tmap[tensor._id]
            self.slots.append(reg)

    def allocate(self, tensor: Union[IRTensor, Any]) -> str:
        """
        Allocate a tensor name for the tensor.
        New tensors will be allocated by available
        unique ids freed by other tensor.
        Existing teensor will get the allocated name.
        """
        if isinstance(tensor, IRTensor):
            ttype = 'g' if tensor.is_grad() else 't'
            # param is graph attribute, don't need allocation
            if tensor.is_param():
                return f'{tensor.name}_{tensor._id}'
            if tensor._id in self.tmap:
                # fetch the original one
                reg = self.tmap[tensor._id]
            else:
                # allocate a new one
                if len(self.slots) == 0:
                    reg = self.nid
                    self.nid += 1
                else:
                    reg = self.slots.pop(-1)
                self.tmap[tensor._id] = reg
            # reg = tensor._id  # => enable this for debug
            return f'{ttype}{reg}'
        else:
            return str(tensor)



