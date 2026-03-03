from typing import TYPE_CHECKING, Optional

import torch

from nnscaler.runtime.device import DeviceGroup

if TYPE_CHECKING:
    from nnscaler.runtime.module import AttrMeta, Zero3AttrMeta


class DTensor:
    def __init__(
        self,
        rank: int,
        local_tensor: Optional[torch.Tensor],
        attr_metas: list['AttrMeta'],
        zero3_attr_metas: list['Zero3AttrMeta']
    ):
        """
        Args:
            rank (int): The rank of the current process.
            local_tensor (Optional[torch.Tensor]): The local tensor in the current rank.
            attr_metas (list[AttrMeta]): A list of metadatas for attributes for each rank.
            zero3_attr_metas (list[Zero3AttrMeta]): A list of metadatas for ZeRO3 attributes for each rank.
        """
        self.rank = rank
        self.local_tensor = local_tensor
        self.attr_metas = attr_metas
        self.zero3_attr_metas = zero3_attr_metas

        self.attr_meta = attr_metas[rank]
        self.zero3_attr_meta = zero3_attr_metas[rank] if zero3_attr_metas else None

        self.z3_groups = self._get_zero3_groups()
        self.z3_pg, self.z3_ranks = self._create_pgs(self.z3_groups)

        self.tp_groups, self.partitioned_dim = self._get_tp_groups()
        self.tp_pg, self.tp_ranks = self._create_pgs(self.tp_groups)

        self.pp_groups = self._get_pp_groups()
        self.pp_pg, self.pp_ranks = self._create_pgs(self.pp_groups)

        super().__init__()

    def _create_pgs(self, groups: Optional[list[list[int]]]) -> tuple[Optional[torch.distributed.ProcessGroup], Optional[list[int]]]:
        if not groups:
            return None, None

        ret_pg: Optional[torch.distributed.ProcessGroup] = None
        ret_ranks: Optional[list[int]] = None
        for g in groups:
            pg = DeviceGroup().get_group(g)
            if self.rank in g:
                ret_pg = pg
                ret_ranks = g
        return ret_pg, ret_ranks

    def _get_zero3_groups(self):
        if self.zero3_attr_meta is None:
            return None

        zero3_groups: list[list[int]] = []
        last_end = -1
        chunk_size = -1
        for i, meta in enumerate(self.zero3_attr_metas):
            if meta is None:
                continue

            assert self.attr_meta is not None, "ZeRO3 attributes should also have attr metadata."
            if chunk_size == -1:
                chunk_size = meta.chunk_size
            assert meta.chunk_size == chunk_size, "All ZeRO3 attributes should have the same chunk size."

            if meta.start == 0:
                zero3_groups.append([i])
                last_end = meta.end
            else:
                assert meta.start == last_end, "ZeRO3 groups should be non-overlapping and cover the full tensor."
                zero3_groups[-1].append(i)
                last_end = meta.end

        assert all(len(group) == len(zero3_groups[0]) for group in zero3_groups), "All ZeRO3 groups should have the same number of ranks."
        return zero3_groups

    def _gather_zero3(self, local_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        # For ZeRO3 partitioned parameters, we need to gather the tensor on the current rank using ZeRO3 metadata.
        if self.z3_pg is None:
            return local_tensor

        local_tensor = local_tensor.cuda(non_blocking=True)
        dest_tensor = torch.empty(
            self.zero3_attr_meta.chunk_size * len(self.z3_ranks),
            dtype=self.attr_meta.dtype,
            device='cuda'
        )
        torch.distributed.all_gather_into_tensor(dest_tensor, local_tensor, group=self.z3_pg)
        del local_tensor
        # The gathered tensor may have extra padding elements, so we need to slice it to the full size and reshape it.
        return dest_tensor[:self.attr_meta.get_full_numel()].reshape(self.attr_meta.shape).contiguous()

    def _get_tp_groups(self):
        if self.attr_meta is None:
            return None, None

        partitioned_dims = self.attr_meta.get_partitioned_dims()
        assert len(partitioned_dims) <= 1, "Only support partitioning on one dimension for now."
        partitioned_dim = partitioned_dims[0] if partitioned_dims else None

        if partitioned_dim is None:
            return None, None

        last_end = -1
        tp_groups: list[list[int]] = []

        for i, meta in enumerate(self.attr_metas):
            if meta is None:
                continue
            slicer: slice = meta.slicers[partitioned_dim]
            if slicer.start == 0:
                tp_groups.append([i])
                last_end = slicer.stop
            else:
                assert slicer.start == last_end, "TP groups should be non-overlapping and cover the full tensor."
                tp_groups[-1].append(i)
                last_end = slicer.stop

        return tp_groups, partitioned_dim

    def _gather_tp(self, local_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.tp_pg is None:
            return local_tensor

        local_tensor = local_tensor.cuda(non_blocking=True)
        dest = [
            torch.empty(self.attr_meta.sub_shape, dtype=self.attr_meta.dtype, device='cuda')
            for _ in self.tp_ranks
        ]
        # all_gather_into_tensor is not used here
        # because the partitioned dimension may not be the first dimension,
        # and all_gather_into_tensor only supports gathering on the first dimension.
        torch.distributed.all_gather(dest, local_tensor, group=self.tp_pg)
        del local_tensor
        ret = torch.cat(dest, dim=self.partitioned_dim)
        del dest
        return ret

    def _get_pp_groups(self):
        # collect all param names
        num_replicated = 0
        for meta in self.attr_metas:
            if meta is not None:
                num_replicated += 1

        if num_replicated == len(self.attr_metas):
            # all ranks have the full tensor, no need to gather
            return None

        assert len(self.attr_metas) % num_replicated == 0, "The number of replicated ranks should be divisible by the total number of ranks."
        pp_groups: list[list[int]] = []
        for i in range(0, len(self.attr_metas), num_replicated):
            pp_groups.append(list(range(i, i + num_replicated)))
        return pp_groups

    def _gather_pp(self, local_tensor: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pp_pg is None:
            assert local_tensor is not None, "Full tensor should be available on all ranks when pp_pg is None."
            return local_tensor

        if local_tensor is None:
            local_tensor = torch.empty(self.attr_meta.shape, dtype=self.attr_meta.dtype, device='cuda')
        else:
            local_tensor = local_tensor.cuda(non_blocking=True)

        src_rank = None
        for r in self.pp_ranks:
            if self.attr_metas[r] is not None:
                assert src_rank is None, "There should be only one source rank for PP broadcast."
                src_rank = r

        assert src_rank is not None, "There should be at least one source rank for PP broadcast."
        torch.distributed.broadcast(local_tensor, src=src_rank, group=self.pp_pg)
        return local_tensor

    def full_tensor(self, *, device=None) -> torch.Tensor:
        """
        Gather the full tensor for the current attribute, regardless of whether it is partitioned or replicated.
        Here we take 3 steps to gather the full tensor:
        1. zero3 partitioned tensor: gather the tensor (may be sharded) on the current rank using ZeRO3 metadata.
        2. tp partitioned tensor: gather the tensor on the current rank using TP metadata.
        3. pp: broadcast the full tensor to ranks in the same PP group.
        """
        device = device or torch.cuda.current_device()

        ftensor = self._gather_zero3(self.local_tensor)
        ftensor = self._gather_tp(ftensor)
        ftensor = self._gather_pp(ftensor)
        ftensor = ftensor.to(device, non_blocking=True)

        return ftensor
