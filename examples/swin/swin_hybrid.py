
# --------------------------------------------------------
# Modified from Swin-Transformer Repo
"""
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_hybrid.py \
        --layer0 8 1 1 \
        --layer1 8 1 1 \
        --layer2 8 1 1 \
        --layer3 8 1 1 \
        --gbs 1 --mbs 1

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NID \
    --master_addr=worker-0 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_hybrid.py \
        --layer0 2 8 1 \
        --layer1 2 8 1 \
        --layer2 2 8 1 \
        --layer3 2 8 1 \
        --gbs 8 --mbs 8

# V100-16GB: 8GPU: need checkpoint: 8 micro bs
"""
# --------------------------------------------------------

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


import cube
from cube.profiler import CudaTimer
from cube.runtime.device import DeviceGroup
from cube.profiler.timer import print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.reducer import Reducer

import argparse

from examples.swin.hybrid_schedule import scheduling_1f1b, is_last_stage
from examples.swin.layers import ColumnParallelLinear, RowParallelLinear, DPtoTP, TPtoDP

from examples.swin.pmodule import ParallelModule


_dp_reducer: Dict[Tuple[int], Reducer] = dict()


def setup_device_group(pp: int, dp: int, tp: int, layer_id: int):
    """
    Layer wise device group initialize

    Returns:

    """
    resource = cube.runtime.resource.EnvResource()
    ndevs = resource.ngpus

    if not pp * tp * dp == ndevs:
        raise RuntimeError("Expected same device number")

    # assert tp == 1 or dp == 1, "Currently hybrid not supported"

    devs = cube.runtime.device.DeviceGroup()

    myrank = torch.distributed.get_rank()

    # initialize tensor parallel groups
    for i in range(ndevs // tp):
        ranks = list(range(i * tp, (i + 1) * tp))
        if len(ranks) > 1:
            group = devs.get_group(ranks)
        if myrank in ranks:
            tp_ranks = ranks
    print_each_rank(f'layer {layer_id}: initialzed tensor parallel group: {tp_ranks}', rank_only=myrank)

    # initialize data parallel groups
    for i in range(pp):
        start_rank = i * ndevs // pp
        end_rank = (i+1) * ndevs // pp
        for j in range(tp):
            ranks = list(range(start_rank + j, end_rank, tp))
            if len(ranks) > 1:
                group = devs.get_group(ranks)
            if myrank in ranks:
                dp_ranks = ranks
                _dp_reducer[tuple(dp_ranks)] = Reducer(dp_ranks)
    print_each_rank(f'layer {layer_id}: initialzed data parallel group: {dp_ranks}', rank_only=myrank)

    # initialize pipeline parallel groups
    for i in range(dp * tp):
        ranks = list(range(i, ndevs, tp * dp))
        if len(ranks) > 1:
            group = devs.get_group(ranks)
        if myrank in ranks:
            pp_ranks = ranks
    print_each_rank(f'layer {layer_id}: initialized pipeline parallel group: {pp_ranks}')

    return pp_ranks, dp_ranks, tp_ranks


def drop_path(x, drop_prob: float = 0.):
    if drop_prob == 0.:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class MegatronMlp(ParallelModule):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.,
                 pp_ranks=-1, tp_ranks=-1, dp_ranks=-1):
        super().__init__(
            pp_ranks=pp_ranks, dp_ranks=dp_ranks, tp_ranks=tp_ranks
        )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, in_adapter=True, out_adapter=False, tp_group=self.tp_group)
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = RowParallelLinear(hidden_features, out_features, in_adapter=False, out_adapter=True, tp_group=self.tp_group)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # [B, H_window_num, window_size, W_window_num, window_size, C]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # [B, H_window_num, W_window_num, window_size, window_size, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # [B * H_windows_num * W_window_size, window_size, window_size, C]
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_position_index(window_size_h: int, window_size_w: int):
    coords_h = torch.arange(window_size_h)
    coords_w = torch.arange(window_size_w)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size_w - 1
    relative_coords[:, :, 0] *= 2 * window_size_w - 1
    relative_position_index = relative_coords.sum(-1).view(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index


class MegatronWindowAttention(ParallelModule):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0,
                 pp_ranks=-1, tp_ranks=-1, dp_ranks=-1):

        super().__init__(
            pp_ranks=pp_ranks, dp_ranks=dp_ranks, tp_ranks=tp_ranks
        )
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.global_num_heads = num_heads

        tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        if num_heads % tp_world_size != 0:
            print(f'detecting un-even num head {num_heads} partition to {tp_world_size}')
        self.num_heads = num_heads // tp_world_size
        self.dim_heads = dim // self.global_num_heads
        self.scale = qk_scale or self.dim_heads ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # relative position index
        relative_position_index = window_position_index(self.window_size[0], self.window_size[1])
        self.register_buffer('relative_position_index', relative_position_index)


        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # print(f'qkv embed dim: {dim}')
        self.qkv = ColumnParallelLinear(dim, dim * 3, bias=qkv_bias, in_adapter=True, out_adapter=False, tp_group=self.tp_group)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj = RowParallelLinear(dim, dim, in_adapter=False, out_adapter=True, tp_group=self.tp_group)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.dim_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]
        # [Wh * Ww, Wh * Ww, nH]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads * self.dim_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(ParallelModule):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 pp_ranks=-1, tp_ranks=-1, dp_ranks=-1, fw_bs=-1):
        super().__init__(
            pp_ranks=pp_ranks, dp_ranks=dp_ranks, tp_ranks=tp_ranks
        )

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = MegatronWindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            pp_ranks=pp_ranks, dp_ranks=dp_ranks, tp_ranks=tp_ranks)

        self.drop_path_p = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MegatronMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                               pp_ranks=pp_ranks, dp_ranks=dp_ranks, tp_ranks=tp_ranks)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        H, W = self.input_resolution

        assert fw_bs // len(dp_ranks) != 0
        self.set_in_size([fw_bs // len(dp_ranks), H * W, self.dim])
        self.set_out_size([fw_bs // len(dp_ranks), H * W, self.dim])

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # [B, H, W, C] -> [B * num_windows, window_size_h, windows_size_w, C]
        x_windows = window_partition(shifted_x, self.window_size)
        # -> [B * num_windows, window_size_h * windows_size_w, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # same in/out: [B * num_windows, window_size_h * windows_size_w, C]
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        # [B * num_windows, w_h * w_w, C] -> [B * num_windows, w_h, w_w, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # [B * num_windows, window_size_h, windows_size_w, C] -> [B, H', W', C]
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        # [B, H', W', C] -> [B, H, W, C]
        x = shifted_x
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # [B, H, W, C] -> [B, H * W, C]
        x = x.view(B, H * W, C)
        # [B, H * W, C] -> [B, H * W, C]
        x = shortcut + drop_path(x, self.drop_path_p)
        # FFN
        # [B, H * W, C] -> [B, H * W, C]
        ffn = self.norm2(x)
        # [B, H * W, C] -> [B, H * W, C]
        ffn = self.mlp(ffn)
        # [B, H * W, C] + [B, H * W, C] -> [B, H * W, C]
        x = x + drop_path(ffn, self.drop_path_p)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(ParallelModule):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm,
                 pp_ranks=-1, tp_ranks=-1, dp_ranks=-1, fw_bs=-1):
        super().__init__(
            pp_ranks=pp_ranks, dp_ranks=dp_ranks, tp_ranks=tp_ranks
        )

        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

        H, W = self.input_resolution

        assert fw_bs // len(dp_ranks) != 0
        self.set_in_size([fw_bs // len(dp_ranks), H * W, self.dim])
        self.set_out_size([fw_bs // len(dp_ranks), H // 2 * W // 2, self.dim * 2])

    def forward(self, x):
        """
        x: B, H*W, C
        """
        assert list(x.shape) == self.in_size

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        assert list(x.shape) == self.out_size
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(ParallelModule):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 pp_ranks=-1, tp_ranks=-1, dp_ranks=-1, layer_id=-1, fw_bs=-1):

        super().__init__(
            pp_ranks=pp_ranks, dp_ranks=dp_ranks, tp_ranks=tp_ranks
        )
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            block = SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pp_ranks=pp_ranks, dp_ranks=dp_ranks, tp_ranks=tp_ranks, fw_bs=fw_bs
            )
            self.blocks.append(block)

    def forward(self, x):
        raise RuntimeError("Error call here")

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 pconfigs=None, fw_bs=-1, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        # self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        tp_ranks, dp_ranks, pp_ranks = list(), list(), list()
        for i in range(4):
            pconfig = pconfigs[i]
            layer_pp_ranks, layer_dp_ranks, layer_tp_ranks = setup_device_group(**pconfig)
            tp_ranks.append(layer_tp_ranks)
            dp_ranks.append(layer_dp_ranks)
            pp_ranks.append(layer_pp_ranks)
        
        # build network layers
        layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            pconfig = pconfigs[i_layer]
            layer_tp_ranks, layer_dp_ranks = tp_ranks[i_layer], dp_ranks[i_layer]

            if i_layer != self.num_layers - 1:
                next_layer_tp_ranks = tp_ranks[i_layer + 1]
                next_layer_dp_ranks = dp_ranks[i_layer + 1]
            else:
                next_layer_dp_ranks = list()
                next_layer_tp_ranks = list()

            input_resolution = (
                patches_resolution[0] // (2 ** i_layer),
                patches_resolution[1] // (2 ** i_layer)
            )
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=input_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                pp_ranks=pp_ranks[i_layer], dp_ranks=dp_ranks[i_layer], tp_ranks=tp_ranks[i_layer],
                fw_bs=fw_bs
            )

            for block in layer.blocks:
                layers.append(block)

            if i_layer < self.num_layers - 1:
                merging = PatchMerging(
                    input_resolution, dim=int(embed_dim * 2 ** i_layer),
                    norm_layer=norm_layer,
                    pp_ranks=pp_ranks[i_layer], dp_ranks=dp_ranks[i_layer], tp_ranks=tp_ranks[i_layer],
                    fw_bs = fw_bs,
                )
                layers.append(merging)
            else:
                merging = None

            # adapter
            if len(layer_dp_ranks) == 1 and len(layer_tp_ranks) > 1 \
               and len(next_layer_dp_ranks) > 1 and len(next_layer_tp_ranks) == 1:
                print_each_rank('add tp to dp adapters')
                adapter = TPtoDP(DeviceGroup().get_group(next_layer_dp_ranks))
                adapter.in_size = layers[-1].out_size
                out_size = [size for size in layers[-1].out_size]
                out_size[0] = out_size[0] // len(next_layer_dp_ranks)
                adapter.out_size = out_size
            elif len(layer_tp_ranks) == 1 and len(layer_dp_ranks) > 1 \
                 and len(next_layer_tp_ranks) > 1 and len(next_layer_dp_ranks) == 1:
                print_each_rank('add dp to tp adapters')
                adapter = DPtoTP(DeviceGroup().get_group(next_layer_tp_ranks))
                adapter.in_size = layers[-1].out_size
                out_size = [size for size in layers[-1].out_size]
                out_size[0] = out_size[0] * len(layer_dp_ranks)
                adapter.out_size = out_size
            elif len(layer_tp_ranks) == len(next_layer_tp_ranks) and \
                 len(layer_dp_ranks) == len(next_layer_dp_ranks):
                adapter = torch.nn.Identity()
                adapter.in_size = layers[-1].out_size
                adapter.out_size = layers[-1].out_size
            layers.append(adapter)


        # ================ Pipeline Parallel Region ======================
        self.pp_group = DeviceGroup().get_group(pp_ranks[0])
        pp_rank = torch.distributed.get_rank(self.pp_group)
        pp_size = torch.distributed.get_world_size(self.pp_group)

        assert len(layers) == 31

        for block in layers:
            print_each_rank(f'> block: {type(block).__name__}: in {block.in_size}, out: {block.out_size}', rank_only=0)

        chunk = len(layers) // pp_size
        if len(layers) % pp_size != 0:
            remain = len(layers) % pp_size
            if pp_rank < remain:
                start = pp_rank * (chunk+1)
                chunk = chunk + 1
            else:
                start = remain * (chunk + 1) + (pp_rank - remain) * chunk
        else:
            start = pp_rank * chunk
        stop = start + chunk

        # self.use_checkpoint = [False] * (stop - start)
        self.use_checkpoint = [True] * (stop - start)

        # 8gpu layer assign
        # layer_split = [5, 5, 4, 3, 3, 3, 3, 5] # original
        # layer_split = [3, 3, 3, 3, 3, 4, 4, 4]
        # assert sum(layer_split) == 31
        # start = sum(layer_split[0:pp_rank])
        # stop = sum(layer_split[0:pp_rank+1])
        # self.use_checkpoint = [False] * (stop - start)
        # for idx in range(stop - start):
        #     if pp_rank == 0:
        #         if idx < 1:
        #             self.use_checkpoint[idx] = True

        # 4 stage layer assign
        # layer_split = [8, 8, 7, 8]  # original
        # layer_split = [6, 7, 7, 7]
        
        # assert sum(layer_split) == 31
        # start = sum(layer_split[0:pp_rank])
        # stop = sum(layer_split[0:pp_rank+1])

        print_each_rank(f'layer start -> end: {start} -> {stop}')
        print_each_rank(self.use_checkpoint)
        self.layers = layers[start:stop]

        local_chunk = list()
        for block in self.layers:
            local_chunk.append(f'{type(block).__name__}: in: {block.in_size}; out: {block.out_size}')
        local_chunk = '\n'.join(local_chunk)
        print_each_rank('local chunk:\n' + local_chunk)

        self.in_size = self.layers[0].in_size
        assert isinstance(self.in_size, list)
        self.out_size = self.layers[-1].out_size
        assert isinstance(self.out_size, list)

        self.preprocess = False
        if pp_rank == 0:
            self.preprocess = True
            self.in_size = [in_chans, img_size, img_size]
        self.postprocess = False
        if is_last_stage(self.pp_group):
            self.postprocess = True
            self.out_size = [1,]

        if self.postprocess:
            self.norm = norm_layer(self.num_features)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(self.num_features, num_classes)


        # =================== Data Parallel ========================

        self.split_data = len(dp_ranks[0])

        # preprocess data parallel region
        if self.preprocess and len(dp_ranks[0]) > 1:
            for param in self.patch_embed.parameters():
                _dp_reducer[tuple(dp_ranks[0])].add_param(param)
        
        # block data parallel region 
        for block in self.layers:
            if isinstance(block, ParallelModule):
                if block.use_dp():
                    for param in block.parameters():
                        _dp_reducer[block.dp_ranks].add_param(param)

        # postprocess data parallel region
        if self.postprocess and len(dp_ranks[-1]) > 1:
            for param in self.norm.parameters():
                _dp_reducer[tuple(dp_ranks[-1])].add_param(param)
            for param in self.head.parameters():
                _dp_reducer[tuple(dp_ranks[-1])].add_param(param)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image: torch.Tensor, feature_map=None):
        
        if self.preprocess:
            with torch.no_grad():
                # FIXME: should select corresponding chunk
                image = image.chunk(self.split_data, 0)[0]
            x = self.patch_embed(image)
            x = self.pos_drop(x)
            feature_map = x

        for layer, use_checkpoint in zip(self.layers, self.use_checkpoint):
            if use_checkpoint:
                feature_map = checkpoint.checkpoint(layer, feature_map)
            else:
                feature_map = layer(feature_map)
        x = feature_map

        if self.postprocess:
            x = self.norm(x)  # B L C
            x = self.avgpool(x.transpose(1, 2))  # B C L
            x = torch.flatten(x, 1)
            x = self.head(x)
            # simulate for simplicity
            x = torch.sum(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def train(args, pconfigs):

    # dim_head is always 32

    # img resolution, windows size: 224, 384, 518, 640
    C, H, W, window_size = [3, 224, 224, 7]
    # C, H, W, window_size = [3, 384, 384, 12]
    # C, H, W, window_size = [3, 518, 518, ?]
    # C, H, W, window_size = [3, 640, 640, 20]
    # C, H, W, window_size = [3, 1536, 1536, 48]

    # image batch size
    N = args.gbs

    # Swin-Tiny
    # embed_dim, depths, num_heads = [
    #     96, [2, 2, 6, 2], [3, 6, 12, 24]
    # ]

    # SwinV2-B: 87 M
    # embed_dim, depths, num_heads = [
    #     128, [2, 2, 18, 2], [4, 8, 16, 32]
    # ]

    # SwinV2-L: 196 M
    # embed_dim, depths, num_heads = [
    #     192, [2, 2, 18, 2], [6, 12, 24, 48]
    # ]

    # SwinV2-H: 657 M
    # embed_dim, depths, num_heads = [
    #     352, [2, 2, 18, 2], [11, 22, 44, 88]
    # ]

    # SwinV2-H modified: 782 M
    embed_dim, depths, num_heads = [
        384, [2, 2, 18, 2], [12, 24, 48, 96]
    ]
    # # head dim 32 -> 48
    embed_dim, depths, num_heads = [
        576, [2, 2, 18, 2], [12, 24, 48, 96]
    ]
    # # head dim 32 -> 64 -- too much
    embed_dim, depths, num_heads = [
        768, [2, 2, 18, 2], [12, 24, 48, 96]
    ]
    # # head dim 32 -> 80 
    embed_dim, depths, num_heads = [
        960, [2, 2, 18, 2], [12, 24, 48, 96]
    ]
    # # head dim 32 -> 96
    embed_dim, depths, num_heads = [
        1152, [2, 2, 18, 2], [12, 24, 48, 96]
    ]
    # # head dim 32 -> 112
    embed_dim, depths, num_heads = [
        1344, [2, 2, 18, 2], [12, 24, 48, 96]
    ]
    # head dim 32 -> 128
    embed_dim, depths, num_heads = [
        1536, [2, 2, 18, 2], [12, 24, 48, 96]
    ]
    # head dim 32 -> 144
    embed_dim, depths, num_heads = [
        1728, [2, 2, 18, 2], [12, 24, 48, 96]
    ]
    # head dim 32 -> 160
    embed_dim, depths, num_heads = [
        1920, [2, 2, 18, 2], [12, 24, 48, 96]
    ]

    # SwinV2-G:  2.5B Model
    # embed_dim, depths, num_heads = [
    #     512, [2, 2, 42, 2], [16, 32, 64, 128]
    # ]

    # 895.7 M Model
    # embed_dim, depths, num_heads = [
    #     384, [2, 2, 22, 2], [12, 24, 48, 96]
    # ]


    # 2.01B model
    # embed_dim, depths, num_heads = [
    #     576, [2, 2, 22, 2], [12, 24, 48, 96]
    # ]

    print_each_rank(
        f'config: embed_dim: {embed_dim}, depths: {depths}, num_heads: {num_heads}'
    )


    model = SwinTransformer(img_size = H,
                            embed_dim = embed_dim,
                            depths = depths,
                            num_heads = num_heads,
                            window_size = window_size,
                            pconfigs = pconfigs,
                            fw_bs = args.mbs)
    model = model.cuda()
    memory_summary()

    dataloader = cube.runtime.syndata.SynDataLoader(
        1280, [0], [args.gbs, C, H, W])
    dataloader.set_data_buffer(buffer_num=2)

    def train_iter(model, dataloader):
        img = next(dataloader)
        scheduling_1f1b(model, [img], args.gbs, args.mbs, torch.float, model.pp_group)
        torch.distributed.barrier()
        CudaTimer().start('dp_allreduce')
        for ranks in _dp_reducer:
            reducer = _dp_reducer[ranks]
            reducer.allreduce()
        CudaTimer().stop('dp_allreduce')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # start training
    nparams_million = sum(p.numel() for p in model.parameters()) / 1000 / 1000
    print_each_rank('model has {:.2f} million parameters'.format(nparams_million))

    CudaTimer(enable=False).warmup()
    torch.distributed.barrier()
    iter_num = 20
    for step in range(iter_num):
        if step >= 10:
            CudaTimer(enable=True).start('e2e')
        torch.distributed.barrier()
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        torch.distributed.barrier()
        if step == 1:
            print('> passed on 1st iteration')
            memory_summary()
        if step >= 10:
            CudaTimer().stop('e2e')
        if (step + 1) % 10 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    iter_time = CudaTimer().duration(iter_num-10, field_name='e2e')
    throughput = N / iter_time * 1000
    print_each_rank('e2e time {:.2f} ms/iter. Throughput: {:.2f} samples/sec'.format(
          iter_time, throughput)
    )

    CudaTimer().print_all(times=iter_num-10)
    memory_summary()


if __name__ == '__main__':
    
    # resource allocation
    parser = argparse.ArgumentParser(description='swin')
    parser.add_argument('--layer0', type=int, nargs='+',
                        help='pipeline, data, tensor parallel config')
    parser.add_argument('--layer1', type=int, nargs='+',
                        help='pipeline, data, tensor parallel config')
    parser.add_argument('--layer2', type=int, nargs='+',
                        help='pipeline, data, tensor parallel config')
    parser.add_argument('--layer3', type=int, nargs='+',
                        help='pipeline, data, tensor parallel config')
    parser.add_argument('--gbs', type=int, default=-1)
    parser.add_argument('--mbs', type=int, default=-1)
    args = parser.parse_args()

    cube.init()

    # allocate resource
    resource = cube.runtime.resource.EnvResource()
    ndevs = resource.ngpus

    args.pp = args.layer0[0]

    pconfigs = [
        dict(layer_id=0, pp=args.layer0[0], dp=args.layer0[1], tp=args.layer0[2]), # basic layer 0
        dict(layer_id=1, pp=args.layer0[0], dp=args.layer1[1], tp=args.layer1[2]), # basic layer 1
        dict(layer_id=2, pp=args.layer0[0], dp=args.layer2[1], tp=args.layer2[2]), # basic layer 2
        dict(layer_id=3, pp=args.layer0[0], dp=args.layer3[1], tp=args.layer3[2]), # basic layer 3
    ]

    train(args, pconfigs)
