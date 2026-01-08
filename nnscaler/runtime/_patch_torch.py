# The following code is copied from torch.distributed.distributed_c10d in PyTorch 2.4.0
# For copyright, see pytorch/LICENSE
# https://github.com/pytorch/pytorch/blob/main/LICENSE


import torch
import torch.distributed


if torch.__version__ < (2, 4, 0):
    # send_object_list and recv_object_list only available in PyTorch 2.4.0+

    import torch.distributed.distributed_c10d as dist_c10d


    if torch.__version__ < (2, 3, 0):
        def _object_to_tensor(obj, device, group):
            return dist_c10d._object_to_tensor(obj, device)
    else:
        def _object_to_tensor(obj, device, group):
            return dist_c10d._object_to_tensor(obj, device, group)


    if torch.__version__ < (2, 3, 0):
        def _tensor_to_object(tensor, size, group):
            return dist_c10d._tensor_to_object(tensor, size)
    else:
        def _tensor_to_object(tensor, size, group):
            return dist_c10d._tensor_to_object(tensor, size, group)


    def send_object_list(object_list, dst, group=None, device=None):
        if torch.distributed.get_rank() == dst:
            raise ValueError(
                "Invalid destination rank: destination rank should not be the same as "
                "the rank of the current process."
            )

        if dist_c10d._rank_not_in_group(group):
            dist_c10d._warn_not_in_group("send_object_list")
            return

        # Current device selection.
        # To preserve backwards compatibility, ``device`` is default to ``None``
        # in which case we run current logic of device selection, i.e.
        # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
        # case it is not ``None`` we move the size and object tensors to be
        # sent to this device.
        current_device = device or torch.device("cuda", torch.cuda.current_device())
        # Serialize object_list elements to tensors on src rank.
        tensor_list, size_list = zip(*[_object_to_tensor(obj, current_device, group) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)

        # Send object sizes
        torch.distributed.send(object_sizes_tensor, dst=dst, group=group)

        # Concatenate and send serialized object tensors
        # Note: torch.cat will do an extra memory copy to the current device, if the tensor_list
        # has only one element, we can skip the copy.
        if len(tensor_list) == 1:  # type: ignore[possibly-undefined]
            object_tensor = tensor_list[0]
        else:
            object_tensor = torch.cat(tensor_list)

        torch.distributed.send(object_tensor, dst=dst, group=group)


    def recv_object_list(object_list, src=None, group=None, device=None):
        if dist_c10d._rank_not_in_group(group):
            dist_c10d._warn_not_in_group("recv_object_list")
            return -1

        # Current device selection.
        # To preserve backwards compatibility, ``device`` is default to ``None``
        # in which case we run current logic of device selection, i.e.
        # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
        # case it is not ``None`` we move the size and object tensors to be
        # received to this device.
        current_device = device or torch.device("cuda", torch.cuda.current_device())
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long, device=current_device)

        # Receive object sizes
        rank_sizes = torch.distributed.recv(object_sizes_tensor, src=src, group=group)

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device=current_device
        )

        rank_objects = torch.distributed.recv(object_tensor, src=src, group=group)
        assert rank_sizes == rank_objects, "Mismatch in return ranks for object sizes and objects."
        # Deserialize objects using their stored sizes.
        offset = 0
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size, group)
        return rank_objects

    torch.distributed.send_object_list = send_object_list
    torch.distributed.recv_object_list = recv_object_list
