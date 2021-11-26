import torch


class RollDimParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, shifts, dims, group, full_input=False, full_output=False):
        pass

    @staticmethod
    def backward(ctx):
        pass

def roll_dim_parallel(input: torch.Tensor, shift: int, dim: int,
                      group, full_input=False, full_output=False):
    """
    partition torch.roll at shifted dimension

    Inputs:
        input: [B, H, W, C]
        shift: int
        dim: int
    """
    world_size = torch.distributed.get_world_size(group)
    rank = torch.distributed.get_rank(group)
    # halo exchange at H dimension
    if dim == 1:
        assert shift < 0
        shift = 0 - shift
        local = input[:, shift:, :, :]
        remote = input[:, slice(0, shift), :, :].contiguous()
        recv_tensor = torch.empty_like(remote, requires_grad=True)
        # send to next rank and recv from prevous rank
        send_op = torch.distributed.P2POp(
            torch.distributed.isend, remote,
            (rank - 1 + world_size) % world_size, group=group
        )
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv, recv_tensor,
            (rank + 1) % world_size, group=group
        )
        ops = [send_op, recv_op] if rank % 2 == 0 else [recv_op, send_op]
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        tensor = torch.cat((local, recv_tensor), dim=dim).contiguous()
        return tensor
    else:
        raise NotImplementedError


def roll_dim_allgather(input: torch.Tensor, shift: int, dim: int, group,
                       full_input=False, full_output=False):
    """
    partition torch.roll at shifted dimension

    Inputs:
        input: [B, H, W, C]
        shift: int
        dim: int
    """
    world_size = torch.distributed.get_world_size(group)
    rank = torch.distributed.get_rank(group)
    # allgather to have all and select what each rank needed
    if dim == 1:
        tensor_list = [torch.empty_like(input) for _ in range(world_size)]
        tensor_list[rank] = input 
        torch.distributed.all_gather(tensor_list, input, group=group)
        full_tensor = torch.cat(tuple(tensor_list), dim=dim).contiguous()
        full_tensor = torch.roll(full_tensor, shifts=(shift,), dims=(dim,))
        chunk_len = input.shape[dim]
        mytensor = full_tensor[:, rank * chunk_len : (rank + 1) * chunk_len, :, :]
        mytensor = mytensor.contiguous()
        return mytensor
    else:
        raise NotImplementedError
