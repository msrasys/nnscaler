import torch


def _roll_dim_parallel(input: torch.Tensor, shift: int, dim: int, group):
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
    if shift < 0:
        shift = 0 - shift
        if dim == 1:
            local = input[:, shift:, :, :]
            remote = input[:, slice(0, shift), :, :].contiguous()
        elif dim == 2:
            local = input[:, :, shift:, :]
            remote = input[:, :, slice(0, shift), :].contiguous()
        else:
            raise NotImplementedError("Only support on dim 1 and dim 2")
        recv_tensor = torch.empty_like(remote)
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
    elif shift > 0:
        boundary = input.shape[dim] - shift
        if dim == 1:
            local = input[:, slice(0, boundary), :, :]
            remote = input[:, slice(boundary, input.shape[dim]), :, :].contiguous()
        elif dim == 2:
            local = input[:, :, slice(0, boundary), :]
            remote = input[:, :, slice(boundary, input.shape[dim]), :].contiguous()
        else:
            raise NotImplementedError("Only support on dim 1 and dim 2")
        recv_tensor = torch.empty_like(remote)
        send_op = torch.distributed.P2POp(
            torch.distributed.isend, remote,
            (rank + 1) % world_size, group=group
        )
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv, recv_tensor,
            (rank - 1 + world_size) % world_size, group=group
        )
        ops = [send_op, recv_op] if rank % 2 == 0 else [recv_op, send_op]
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        tensor = torch.cat((recv_tensor, local), dim=dim).contiguous()
        return tensor
    else:
        return input


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
    tensor_list = [torch.empty_like(input) for _ in range(world_size)]
    tensor_list[rank] = input 
    torch.distributed.all_gather(tensor_list, input, group=group)
    full_tensor = torch.cat(tuple(tensor_list), dim=dim).contiguous()
    full_tensor = torch.roll(full_tensor, shifts=(shift,), dims=(dim,))
    chunk_len = input.shape[dim]
    if dim == 1:
        mytensor = full_tensor[:, rank * chunk_len : (rank + 1) * chunk_len, :, :]
    elif dim == 2:
        mytensor = full_tensor[:, :, rank * chunk_len : (rank + 1) * chunk_len, :]
    else:
        raise NotImplementedError("Only supported on dim 1 and dim 2")
    mytensor = mytensor.contiguous()
    return mytensor


class RollDimParallel(torch.autograd.Function):
    """
    Halo exchange implementation on partitioning torch.roll
    at shift dimension
    
    """
    @staticmethod
    def forward(ctx, input_, shift: int, dim: int, group=None):
        ctx.shift = shift
        ctx.dim = dim
        ctx.group = group
        output = _roll_dim_parallel(input_, shift, dim, group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shift = ctx.shift
        dim = ctx.dim
        group = ctx.group
        grad = _roll_dim_parallel(grad_output, 0-shift, dim, group)
        return grad, None, None, None


def roll_dim_parallel(input: torch.Tensor, shift: int, dim: int, group):
    """
    partition torch.roll at shifted dimension

    Inputs:
        input: [B, H, W, C]
        shift: int
        dim: int
    """
    return RollDimParallel.apply(input, shift, dim, group)


class GridPartition(torch.autograd.Function):
    """
    Full input
    """
    @staticmethod
    def forward(ctx, input_, nrow: int, ncol: int, group=None):
        """
        input: [B, H, W, C]
        """
        ctx.group = group
        world_size = torch.distributed.get_world_size(group)
        ctx.nrow = nrow
        ctx.ncol = ncol
        assert nrow * ncol == world_size
        rank = torch.distributed.get_rank(group)
        myrow = rank // ncol
        mycol = rank % ncol

        chunk = torch.chunk(input_, nrow, dim=1)[myrow]
        chunk = torch.chunk(chunk, ncol, dim=2)[mycol].contiguous()
        return chunk

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        nrow = ctx.nrow
        ncol = ctx.ncol

        world_size = torch.distributed.get_world_size(group)
        rank = torch.distributed.get_rank(group)
        grad_output = grad_output.contiguous()
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        tensor_list[rank] = grad_output
        torch.distributed.all_gather(tensor_list, grad_output, group=group)

        rows = list()
        for row in range(nrow):
            row_slice = torch.cat(tuple(tensor_list[row*ncol:(row+1)*ncol]), dim=2)
            rows.append(row_slice)
        grad_output = torch.cat(tuple(rows), dim=1).contiguous()
        return grad_output, None, None, None


class GridCollection(torch.autograd.Function):
    """
    Full input
    """
    @staticmethod
    def forward(ctx, input_, nrow: int, ncol: int, group=None):
        """
        input: [B, H, W, C]
        output: [B, nrow * H, ncol * W, C]
        """
        ctx.group = group
        world_size = torch.distributed.get_world_size(group)
        ctx.nrow = nrow
        ctx.ncol = ncol
        assert nrow * ncol == world_size

        world_size = torch.distributed.get_world_size(group)
        rank = torch.distributed.get_rank(group)
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_, group=group)

        rows = list()
        for row in range(nrow):
            row_slice = torch.cat(tuple(tensor_list[row*ncol:(row+1)*ncol]), dim=2)
            rows.append(row_slice)
        output = torch.cat(tuple(rows), dim=1).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        nrow = ctx.nrow
        ncol = ctx.ncol

        rank = torch.distributed.get_rank(group)
        myrow = rank // ncol
        mycol = rank % ncol

        chunk = torch.chunk(grad_output, nrow, dim=1)[myrow]
        chunk = torch.chunk(chunk, ncol, dim=2)[mycol].contiguous()
        return chunk, None, None, None


def grid_partition(input_, nrow, ncol, group=None):
    return GridPartition.apply(input_, nrow, ncol, group)


def grid_collection(input_, nrow, ncol, group=None):
    return GridCollection.apply(input_, nrow, ncol, group)
