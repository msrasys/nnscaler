import torch


class ValueMapReduceOp:

    def __init__(self, val_map_op, val_reduce_op):
        if not (callable(val_map_op) and callable(val_reduce_op)):
            raise TypeError("Expected val_map_op and val_reduce_o callable")
        self.val_map_op = (val_map_op,)
        self.val_reduce_op = (val_reduce_op,)

    def map(self, tensor, group):
        if not torch.is_tensor(tensor):
            raise RuntimeError("Expected tensor to be torch.Tensor")
        return self.val_map_op[0](tensor, group)

    def reduce(self, tensor, group):
        if not torch.is_tensor(tensor):
            raise RuntimeError("Expected `tensor` to be torch.Tensor")
        return self.val_map_op[0](tensor, group)


def _val_split_map(tensor, group):
    world_size = torch.distributed.get_world_size(group)
    return tensor / world_size


def _val_sum_reduce(tensor, group):
    torch.distributed.all_reduce(tensor, group=group)
    return tensor


PartialSum = ValueMapReduceOp(
    val_map_op = _val_split_map,
    val_reduce_op = _val_sum_reduce
)
