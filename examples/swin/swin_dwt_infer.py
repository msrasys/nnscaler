
# --------------------------------------------------------
# Modified from Swin-Transformer Repo
"""
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/swin/swin_dwt.py --bs 8 \
        --layer0 1 4 1 \
        --layer1 1 4 1 \
        --layer2 1 1 4 \
        --layer3 1 1 4

"""
# --------------------------------------------------------

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import argparse
import time

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup
from cube.runtime.reducer import Reducer

from examples.swin.layers import ColumnParallelLinear, RowParallelLinear


_wp_reducer: Dict[Tuple[int], Reducer] = dict()
_dp_reducer: Dict[Tuple[int], Reducer] = dict()


def setup_device_group(tp: int, wp: int, dp: int, layer_id: int):
    """
    Layer wise device group initialize

    Returns:

    """
    resource = cube.runtime.resource.EnvResource()
    ndevs = resource.ngpus

    tp_size, tp_group_nums = tp, ndevs // tp
    wp_size, wp_group_nums = wp, ndevs // wp
    dp_size, dp_group_nums = dp, ndevs // dp

    if not tp_size * wp_size * dp_size == ndevs:
        raise RuntimeError("Expected all devices are used")

    devs = cube.runtime.device.DeviceGroup()

    myrank = torch.distributed.get_rank()

    # initialize tensor parallel groups
    for i in range(tp_group_nums):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = devs.get_group(ranks)
        if myrank in ranks:
            tp_ranks = ranks
    print_each_rank(f'layer {layer_id}: initialzed tensor parallel group: {tp_ranks}', rank_only=myrank)

    # initialize wp parallel group
    all_wp_parallel_group_ranks = list()
    for i in range(dp_size):
        start_rank = i * dp_group_nums
        end_rank = (i + 1) * dp_group_nums
        for j in range(tp_size):
            ranks = list(range(start_rank + j, end_rank, tp_size))
            all_wp_parallel_group_ranks.append(ranks)
            # initialize groups
            group = devs.get_group(ranks)
            if myrank in ranks:
                wp_ranks = ranks
                _wp_reducer[tuple(ranks)] = Reducer(ranks)
    print_each_rank(f'layer {layer_id}: initialzed window parallel group: {wp_ranks}', rank_only=myrank)

    # initialize data parallel groups
    start_rank = 0
    end_rank = ndevs
    for i in range(wp_size * tp_size):
        ranks = list(range(i, ndevs, wp_size * tp_size))
        group = devs.get_group(ranks)
        if myrank in ranks:
            dp_ranks = ranks
            _dp_reducer[tuple(ranks)] = Reducer(ranks)
    print_each_rank(f'layer {layer_id}: initialzed data parallel group: {dp_ranks}', rank_only=myrank)
    return tp_ranks, wp_ranks, dp_ranks


def drop_path(x, drop_prob: float = 0.):
    if drop_prob == 0.:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class MegatronMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., tp_group=-1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, in_adapter=True, out_adapter=False, tp_group=tp_group)
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = RowParallelLinear(hidden_features, out_features, in_adapter=False, out_adapter=True, tp_group=tp_group)
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


class MegatronWindowAttention(nn.Module):
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., tp_group=-1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.global_num_heads = num_heads

        tp_world_size = torch.distributed.get_world_size(group=tp_group)
        if num_heads % tp_world_size != 0:
            raise RuntimeError(f'detecting un-even num head {num_heads} partition to {tp_world_size}')
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
        self.qkv = ColumnParallelLinear(dim, dim * 3, bias=qkv_bias, in_adapter=True, out_adapter=False, tp_group=tp_group)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj = RowParallelLinear(dim, dim, in_adapter=False, out_adapter=True, tp_group=tp_group)
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


class SwinTransformerBlock(nn.Module):
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
                 tp_group=-1, wp_plans=-1, layer_id=-1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.wp_group, self.wp_nH_ranks, self.wp_nW_ranks = wp_plans
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        if layer_id == 3:
            print('set shift size to 0')
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = MegatronWindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            tp_group=tp_group)

        self.drop_path_p = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MegatronMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop,
            tp_group=tp_group
        )

        self.use_wp = torch.distributed.get_world_size(self.wp_group) != 1

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

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if self.use_wp:
                shifted_x = cube.runtime.function.roll_grid_parallel(
                    x, (-self.shift_size, -self.shift_size), (1,2),
                    self.wp_nH_ranks, self.wp_nW_ranks, self.wp_group
                )
            else:
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
            if self.use_wp:
                x = cube.runtime.function.roll_grid_parallel(
                    shifted_x, (self.shift_size, self.shift_size), (1,2),
                    self.wp_nH_ranks, self.wp_nW_ranks, self.wp_group
                )
            else:
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


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
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

        return x


class BasicLayer(nn.Module):
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
                 tp=1, wp=1, dp=1, layer_id=-1):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.resource = cube.runtime.resource.EnvResource()
        tp_ranks, wp_ranks, dp_ranks = setup_device_group(tp, wp, dp, layer_id)
        tp_group = DeviceGroup().get_group(tp_ranks)
        wp_group = DeviceGroup().get_group(wp_ranks)
        wp_nH_ranks = [-1]
        wp_nW_ranks = [-1]

        # window parallel
        self.wp_resolution = input_resolution
        if wp > 1:
            H, W = self.input_resolution
            nH = 1
            nW = wp // nH
            while nH <= nW:
                if H % nH != 0 or W % nW != 0 or (H // nH) % window_size != 0 or (W // nW) % window_size != 0:
                    nW = nW // 2
                    nH = int(nH * 2)
                else:
                    break
            if nH > nW:
                raise RuntimeError(f"layer {layer_id}: Cannot window partition plan")
            print_each_rank(f"layer {layer_id}: Find partition plan: H{H} // {nH}, W{W} // {nW}")
            self.wp_resolution = (H // nH, W // nW)
            self.wp_group = wp_group
            # wp_group multi dim shift ranks
            for i in range(nH):
                ranks = list(range(i * nW, (i + 1) * nW))
                if torch.distributed.get_rank(wp_group) in ranks:
                    wp_nW_ranks = ranks
                    break
            for i in range(nW):
                ranks = list(range(i, wp, nW))
                if torch.distributed.get_rank(wp_group) in ranks:
                    wp_nH_ranks = ranks
                    break
            assert wp_nH_ranks != [-1]
            assert wp_nW_ranks != [-1]
            print_each_rank(f'window parallel nH group ranks: {wp_nH_ranks}')
            print_each_rank(f'window parallel nW group ranks: {wp_nW_ranks}')

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinTransformerBlock(
                dim=dim, input_resolution=self.wp_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                tp_group=tp_group, wp_plans=(wp_group, wp_nH_ranks, wp_nW_ranks),
                layer_id = layer_id
            )
            self.blocks.append(block)

        self.wp_preprocess = False
        self.wp_postprocess = False
        if wp > 1:
            for param in self.blocks.parameters():
                _wp_reducer[tuple(wp_ranks)].add_param(param)
            self.wp_preprocess = True
            self.wp_postprocess = True

    def forward(self, x):
        if self.wp_preprocess:
            oH, oW = self.input_resolution
            pH, pW = self.wp_resolution
            x = x.view(-1, oH, oW, self.dim)
            x = cube.runtime.function.grid_partition(x, oH // pH, oW // pW, group=self.wp_group)
            x = x.view(-1, pH * pW, self.dim).contiguous()

        for blk in self.blocks:
            x = blk(x)

        if self.wp_postprocess:
            oH, oW = self.input_resolution
            pH, pW = self.wp_resolution
            x = x.view(-1, pH, pW, self.dim)
            x = cube.runtime.function.grid_collection(x, oH // pH, oW // pW, group=self.wp_group)
            x = x.view(-1, oH * oW, self.dim)
        return x


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
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, pconfigs=None, **kwargs):
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


        # ====================== depth 0 ===========================
        pconfig = pconfigs[0]
        input_resolution = (
            patches_resolution[0] // (2 ** 0), patches_resolution[1] // (2 ** 0)
        )
        self.basic_layer0 = BasicLayer(
            dim=int(embed_dim * 2 ** 0),
            input_resolution=input_resolution,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
            norm_layer=norm_layer,
            **pconfig,
        )

        self.merging0 = PatchMerging(
            input_resolution, dim=int(embed_dim * 2 ** 0), norm_layer=norm_layer
        )

        # ====================== depth 1 ===========================
        pconfig = pconfigs[1]
        input_resolution = (
            patches_resolution[0] // (2 ** 1), patches_resolution[1] // (2 ** 1)
        )
        self.basic_layer1 = BasicLayer(
            dim=int(embed_dim * 2 ** 1),
            input_resolution=input_resolution,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
            norm_layer=norm_layer,
            **pconfig,
        )

        self.merging1 = PatchMerging(
            input_resolution, dim=int(embed_dim * 2 ** 1), norm_layer=norm_layer
        )


        # ====================== depth 2 ===========================
        pconfig = pconfigs[2]
        input_resolution = (
            patches_resolution[0] // (2 ** 2), patches_resolution[1] // (2 ** 2)
        )
        self.basic_layer2 = BasicLayer(
            dim=int(embed_dim * 2 ** 2),
            input_resolution=input_resolution,
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
            norm_layer=norm_layer,
            **pconfig
        )

        self.merging2 = PatchMerging(
            input_resolution, dim=int(embed_dim * 2 ** 2), norm_layer=norm_layer
        )

        # ====================== depth 3 ===========================
        pconfig = pconfigs[3]
        self.basic_layer3 = BasicLayer(
            dim=int(embed_dim * 2 ** 3),
            input_resolution=(patches_resolution[0] // (2 ** 3),
                              patches_resolution[1] // (2 ** 3)),
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
            norm_layer=norm_layer,
            **pconfig
        )

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        CudaTimer().start('basic_layer0')
        x = self.basic_layer0(x)
        CudaTimer().stop('basic_layer0')
        x = self.merging0(x)
        CudaTimer().start('basic_layer1')
        x = self.basic_layer1(x)
        CudaTimer().stop('basic_layer1')
        x = self.merging1(x)
        CudaTimer().start('basic_layer2')
        x = self.basic_layer2(x)
        CudaTimer().stop('basic_layer2')
        x = self.merging2(x)
        CudaTimer().start('basic_layer3')
        x = self.basic_layer3(x)
        CudaTimer().stop('basic_layer3')
        
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C L
        x = torch.flatten(x, 1)

        x = self.head(x)
        return x


def train(args, pconfigs):

    # dim_head is always 32

    # img resolution, windows size: 224, 384, 518, 640
    C, H, W, window_size = [3, 224, 224, 7]
    # C, H, W, window_size = [3, 384, 384, 12]
    # C, H, W, window_size = [3, 518, 518, ?]
    # C, H, W, window_size = [3, 640, 640, 20]
    # C, H, W, window_size = [3, 1536, 1536, 48]

    # image batch size
    N = args.bs

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
        f'Test setting: Resolution {H}, Embed {embed_dim}, depths: {depths}, heads: {num_heads}',
        rank_only=0
    )


    model = SwinTransformer(img_size = H,
                            embed_dim = embed_dim,
                            depths = depths,
                            num_heads = num_heads,
                            window_size = window_size,
                            pconfigs = pconfigs)
    nparams_million = sum(p.numel() for p in model.parameters()) / 1000 / 1000
    print_each_rank('model has {:.2f} million parameters'.format(nparams_million))

    if args.fp16:
        print_each_rank('use half model')
        model = model.half()
    model = model.cuda()
    memory_summary()

    dataloader = cube.runtime.syndata.SynDataLoader(
        1280, [0], [N // args.dp, C, H, W])

    if args.fp16:
        data_buff = [[e.half() for e in data] for data in dataloader.datas]
        dataloader.datas = data_buff

    model.eval()
    def infer_iter(model, dataloader):
        with torch.no_grad():
            img = next(dataloader)
            loss = model(img)

    # start training
    nparams_million = sum(p.numel() for p in model.parameters()) / 1000 / 1000
    print_each_rank('model has {:.2f} million parameters'.format(nparams_million))

    CudaTimer(enable=False).warmup()
    torch.distributed.barrier()
    span = 0
    iter_num = 60
    for step in range(iter_num):
        if step >= 20:
            torch.cuda.synchronize()
            start = time.time()
            CudaTimer(enable=True).start('e2e')
        infer_iter(model, dataloader)
        if step == 1:
            print('> passed on 1st iteration')
            memory_summary()
        if step >= 20:
            torch.cuda.synchronize()
            stop = time.time()
            span += (stop - start) * 1000
            CudaTimer().stop('e2e')
        if (step + 1) % 20 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    iter_time = span / (iter_num-20)
    throughput = N / iter_time * 1000
    print_each_rank('e2e time {:.2f} ms/iter. Throughput: {:.2f} samples/sec'.format(
          iter_time, throughput)
    )
    memory_summary()
    CudaTimer().print_all(times=iter_num-20)


if __name__ == '__main__':

    cube.init()
    
    # resource allocation
    parser = argparse.ArgumentParser(description='swin')
    parser.add_argument('--layer0', type=int, nargs='+',
                        help='data, window tensor parallel config')
    parser.add_argument('--layer1', type=int, nargs='+',
                        help='data, window tensor parallel config')
    parser.add_argument('--layer2', type=int, nargs='+',
                        help='data, window tensor parallel config')
    parser.add_argument('--layer3', type=int, nargs='+',
                        help='data, window tensor parallel config')
    parser.add_argument('--bs', type=int, default=1,
                        help='bs')
    parser.add_argument('--fp16', action='store_true', dest='fp16')
    args = parser.parse_args()

    assert len(args.layer0) == 3
    assert len(args.layer1) == 3
    assert len(args.layer2) == 3
    assert len(args.layer3) == 3

    # data parallel should be same
    assert args.layer0[0] == args.layer1[0] and args.layer1[0] == args.layer2[0] and args.layer2[0] == args.layer3[0]
    args.dp = args.layer0[0]

    pconfigs = [
        dict(layer_id=0, dp=args.layer0[0], wp=args.layer0[1], tp=args.layer0[2]), # basic layer 0
        dict(layer_id=1, dp=args.layer1[0], wp=args.layer1[1], tp=args.layer1[2]), # basic layer 1
        dict(layer_id=2, dp=args.layer2[0], wp=args.layer2[1], tp=args.layer2[2]), # basic layer 2  # prob at 8:1?
        dict(layer_id=3, dp=args.layer3[0], wp=args.layer3[1], tp=args.layer3[2]), # basic layer 3
    ]

    print_each_rank(pconfigs, rank_only=0)
    train(args, pconfigs)
