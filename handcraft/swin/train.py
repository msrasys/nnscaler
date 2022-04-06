"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/swin/train.py \
        --bs 1 --micro-bs 1 --fp16 \
        --dp-size 1 --pp-size 1 --tp-size 1 \
        --layers 10 --dim 128 --heads 4
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary, model_summary
from cube.runtime.adapter.reducer import Reducer
from cube.runtime.device import DeviceGroup
from cube.runtime.adapter.distnn import IdentityAllreduce, AllReduceIdentity, AllGatherSplit
from handcraft.module.schedule import schedule_1f1b
from handcraft.module.stage import PipeStage, layer_division
from handcraft.swin.utils import create_position_bias, create_position_index, trunc_normal_, window_partition, window_reverse, DropPath

import argparse


parser = argparse.ArgumentParser(description='swin')

# model arch
parser.add_argument('--layers', type=int, default=18,
                    help='third stage layer depths. default large')
parser.add_argument('--dim', type=int, default=192,
                    help='input channel of first stage')
parser.add_argument('--heads', type=int, default=6,
                    help='head num of first stage')
# data
parser.add_argument('--img-size', type=int, default=1536,
                    help='image size, can be 224, 640, 1536')
parser.add_argument('--window-size', type=int, default=48,
                    help='image size, can be 7, 40, 48')
# training
parser.add_argument('--bs', type=int, default=256,
                    help='batch size')
parser.add_argument('--micro-bs', type=int, default=1,
                    help='micro batch size')
parser.add_argument('--pp-size', type=int, default=1,
                    help='pipeline parallelism size')
parser.add_argument('--tp-size', type=int, default=1,
                    help='tensor parallelism size')
parser.add_argument('--dp-size', type=int, default=1,
                    help='data parallelism size')
parser.add_argument('--schedule', type=str, default='1f1b', choices=['1f1b'],
                    help='scheduling algorithm')
parser.add_argument('--use-coshard', action='store_true', default=False,
                    help='enable this will split head but co-locate them with re-compute')
parser.add_argument('--use-inner-coshard', action='store_true', default=False,
                    help='enable this will shard bmm in attention of q @ k')
parser.add_argument('--fp16', action='store_true', default=False)

args = parser.parse_args()
print(args)

_tp_group = -1

_dp_group = -1
_dp_reducer = None

_pp_group = -1
_pp_global_ranks = ()
_schedule = schedule_1f1b
_layer_divisions = []

cube.init()
dp_ranks, pp_ranks, tp_ranks= DeviceGroup().create_hybrid(
    [args.dp_size, args.pp_size, args.tp_size]
)

if len(dp_ranks) != 1:
    print_each_rank(f'initializing dp ranks: {dp_ranks}')
    _dp_group = DeviceGroup().get_group(dp_ranks)
    _dp_reducer = Reducer(dp_ranks)

if len(tp_ranks) != 1:
    print_each_rank(f'initializing tp ranks: {tp_ranks}')
    _tp_group = DeviceGroup().get_group(tp_ranks)

if len(pp_ranks) != 1:
    print_each_rank(f'initializing pp ranks: {pp_ranks}')
    _pp_group = DeviceGroup().get_group(pp_ranks)
    _pp_global_ranks = tuple(pp_ranks)

    # layer division
    nlayers = 2 + 2 + args.layers + 2 + 3  # 3 is patch merging layers
    # metrics for V100-32GB-PCIe
    if args.dim == 256:  # OK!
        times = ([109.93] * 2 + [0]) + \
                ([60.34] * 2 + [0]) + \
                ([43.18] * args.layers + [0]) + \
                ([27.51] * 2)
    elif args.dim == 512: # OK!
        times = ([255.10] * 2 + [0]) + \
                ([139.92] * 2 + [0]) + \
                ([90.98] * args.layers + [0]) + \
                ([63.78] * 2)
    elif args.dim == 768: # OK!
        times = ([440.5] * 2 + [0]) + \
                ([241.4] * 2 + [0]) + \
                ([145.7] * args.layers + [0]) + \
                ([108.9] * 2)
    elif args.dim >= 1024:  # TP needed
        times = ([255.10] * 2 + [0]) + \
                ([139.92] * 2 + [0]) + \
                ([90.98] * args.layers + [0]) + \
                ([63.78] * 2)
    else:
        print_each_rank('WARNING: NO Metric Logged!!')
        times = ([1] * 2 + [0]) + \
                ([1] * 2 + [0]) + \
                ([1] * args.layers + [0]) + \
                ([1] * 2)
    num_stages = len(pp_ranks)
    _layer_divisions = layer_division(times, num_stages)
    # specific rules for stage division in order to fit in memory
    if args.dim == 1024 and args.tp_size == 4:
        if _layer_divisions[0][1] > 8:
            remain_times = times[8:]
            _layer_divisions = [(0, 8)] + layer_division(remain_times, num_stages-1, start_id=8)
else:
    _layer_divisions = [(0, 2 + 2 + args.layers + 2 + 3)]
print_each_rank(f'layer divisions: {_layer_divisions}', rank_only=0)


class Config:

    embed_dim = args.dim
    depths = [2, 2, args.layers, 2]
    num_heads = [args.heads, args.heads * 2, args.heads * 4, args.heads * 8]

    mlp_ratio = 4
    qkv_bias = True
    qk_scale = None
    drop_path_rate = 0.2
    drop_rate = 0.2

    img_size = args.img_size
    window_size = args.window_size
    num_classes = 1000


class Mlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self._tp_group = _tp_group
        self._tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features // self._tp_size
        self.fc1 = nn.Linear(in_features, hidden_features // self._tp_size)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features // self._tp_size, out_features)
        self.drop = nn.Dropout(drop)

    def forward_(self, x):
        if self._tp_size > 1:
            x = IdentityAllreduce.apply(x, self._tp_group)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self._tp_size > 1:
            x = AllReduceIdentity.apply(x, self._tp_group)
        return x

    def forward(self, x, recompute=True):
        if recompute:
            x = checkpoint.checkpoint(self.forward_, x)
        else:
            x = self.forward_(x)
        return x

    def flops(self, seqlen: int):
        mlp_flops = dict(
            fc1=seqlen * self.in_features * self.hidden_features,
            act=8 * seqlen * self.hidden_features,
            drop=seqlen * self.hidden_features,
            fc2=seqlen * self.hidden_features * self.in_features,
            final_drop=seqlen * self.in_features,
        )
        return sum(mlp_flops.values())


class SeqMlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.,
                 coshard=1):
        super().__init__()
        self._tp_group = _tp_group
        self._tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)

        self.coshard = coshard
        assert hidden_features is not None
        assert hidden_features % coshard == 0
        self.mlps = torch.nn.ModuleList(
            [Mlp(in_features, hidden_features // coshard, out_features, act_layer, drop) for _ in range(coshard)]
        )
        # remove tp communication inside each mlp as it will be
        # done outside here
        for mlp in self.mlps:
            mlp._tp_size = 1

    def forward(self, x, recompute=True):
        if self._tp_size > 1:
            x = IdentityAllreduce.apply(x, self._tp_group)

        outs = None
        for mlp in self.mlps:
            x_out = mlp(x, recompute=recompute)
            outs = x_out if outs is None else outs + x_out

        if self._tp_size > 1:
            outs = AllReduceIdentity.apply(outs, self._tp_group)
        return outs

    def flops(self, seqlen: int):
        return sum([mlp.flops(seqlen) for mlp in self.mlps])


class WindowAttention(torch.nn.Module):
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

    def __init__(self, dim, inner_dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_index=True):

        super().__init__()
        self._tp_group = _tp_group
        self._tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)

        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.head_dim = inner_dim // num_heads
        assert num_heads % self._tp_size == 0
        self.num_heads = num_heads // self._tp_size
        self.scale = qk_scale or self.head_dim ** -0.5

        # define define a parameter table of relative position bias
        table = create_position_bias(self.window_size, self.num_heads)
        self.relative_position_bias_table = table
        if position_index:
            index = create_position_index(window_size, cuda=False)
            self.register_buffer("relative_position_index", index)
        else:
            self.relative_position_index = None

        self.qkv = nn.Linear(dim, inner_dim // self._tp_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim // self._tp_size, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward_(self, x, mask=None, position_index=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        assert (self.relative_position_index is None) ^ (position_index is None)
        if position_index is not None:
            relative_position_index = position_index
        else:
            relative_position_index = self.relative_position_index

        if self._tp_size > 1:
            x = IdentityAllreduce.apply(x, self._tp_group)

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale

        k = k.transpose(-2, -1)
        # inner coshard by splitting windows
        if args.use_inner_coshard and (B_ == 64 or B_ == 16):
            chunk_num = B_ // 4
            attn = []
            for shard_q, shard_k in zip(torch.chunk(q, chunks=chunk_num, dim=0), torch.chunk(k, chunks=chunk_num, dim=0)):
                attn.append(shard_q @ shard_k)
            attn = torch.concat(tuple(attn), dim=0)
        else:
            attn = (q @ k)

        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
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

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self._tp_size > 1:
            x = AllReduceIdentity.apply(x, self._tp_group)

        return x

    def forward(self, x, mask=None, position_index=None, recompute=True):
        if recompute:
            x = checkpoint.checkpoint(self.forward_, x, mask, position_index)
        else:
            x = self.forward_(x, mask, position_index)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, seqlen: int):
        # calculate flops for one window
        # seqlen is window size * window size
        attn_flops = dict(
            kqv=3 * seqlen * self.dim * self.head_dim * self.num_heads,
            kqv_bias= 3 * seqlen * self.head_dim * self.num_heads,
            q_scale=seqlen * self.num_heads * self.head_dim,
            attn_score=self.num_heads * seqlen * self.head_dim * seqlen, # q @ k
            position_index=self.num_heads * seqlen * seqlen,
            attn_softmax=5 * self.num_heads * seqlen * seqlen,
            attn_dropout=self.num_heads * seqlen * seqlen,
            attn_output=self.num_heads * seqlen * seqlen * self.head_dim, # attn @ v
            out_proj=seqlen * self.num_heads * self.head_dim * self.dim # self.proj(x)
        )
        return sum(attn_flops.values())


class SeqWindowAttention(torch.nn.Module):

    def __init__(self, dim, inner_dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 coshard=1):
        super().__init__()
        self._tp_group = _tp_group
        self._tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)
        assert (num_heads // args.tp_size) % coshard == 0        
        # only coshard num heads of first two stages
        self.coshard = coshard
        self.attns = torch.nn.ModuleList(
            [WindowAttention(
                dim, inner_dim // self.coshard, window_size, num_heads // self.coshard,
                qkv_bias, qk_scale, attn_drop, proj_drop, False) for _ in range(self.coshard)]
        )
        # 1) remove communication inside each attention as it will be
        #    done outside here
        # 2) share same relative position index
        index = create_position_index(window_size, cuda=False)
        self.register_buffer("relative_position_index", index)
        for attn in self.attns:
            attn._tp_size = 1

    def forward(self, x, mask=None, recompute=True):

        # ===> sharding from both window and heads
        # B = x.size(0)
        # if B % 2 == 0:
        #     xs = torch.chunk(x, 2, dim=0)
        #     masks = torch.chunk(mask, 2, dim=0) if mask is not None else (None,) * 2
        # else:
        #     xs = (x,)
        #     masks = (mask,)
        # outs = []
        # for bid, (cx, cmask) in enumerate(zip(xs, masks)):
        #     for attn in self.attns:
        #         cx_out = attn(cx, cmask, recompute)
        #         if len(outs) < bid + 1:
        #             outs.append(cx_out)
        #         else:
        #             outs[bid] = outs[bid] + cx_out
        # outs = torch.concat(tuple(outs), dim=0)
        # return outs

        # ===> sharding only from heads
        if self._tp_size > 1:
            x = IdentityAllreduce.apply(x, self._tp_group)

        outs = None
        for attn in self.attns:
            x_out = attn(x, mask, self.relative_position_index, recompute)
            outs = x_out if outs is None else outs + x_out

        if self._tp_size > 1:
            outs = AllReduceIdentity.apply(outs, self._tp_group)
        return outs

    def flops(self, seqlen: int):
        flops = 0
        for attn in self.attns:
            flops += attn.flops(seqlen)
        return flops


class SwinTransformerBlock(PipeStage):
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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_coshard=False, layer_id=None):
        super().__init__()
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
        if not use_coshard or layer_id in [2,3]:
            self.attn = WindowAttention(
                dim, dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            coshard = num_heads // args.tp_size
            coshard = coshard // 2 if layer_id > 0 else coshard
            print(f'rank [{torch.distributed.get_rank()}]: Swin-stage-{layer_id} using coshard {coshard}')
            self.attn = SeqWindowAttention(
                dim, dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, coshard=coshard)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if not use_coshard or layer_id in [2,3]:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            coshard = num_heads // args.tp_size
            coshard = coshard // 2 if layer_id > 0 else coshard
            self.mlp = SeqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, coshard=coshard)

        H, W = self.input_resolution
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
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
        
        assert args.bs // (args.micro_bs * args.dp_size) != 0
        self.inputs_info = (
            ((args.micro_bs, H * W, self.dim),),
            (torch.float32 if not args.fp16 else torch.float16,)
        )
        self.outputs_info = (
            ((args.micro_bs, H * W, self.dim),),
            (torch.float32 if not args.fp16 else torch.float16,)
        )
        self.layer_id = layer_id
        self.inner_recompute = False if not use_coshard else layer_id in [0,1]

    def forward_(self, x):
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
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask, recompute=self.inner_recompute)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), recompute=self.inner_recompute))
        return x

    def forward(self, x):
        CudaTimer().start(f'layer{self.layer_id}')
        # layer-wise recompute
        if not self.inner_recompute:
            x = checkpoint.checkpoint(self.forward_, x)
        # attention/mlp-wise recompute
        else:
            x = self.forward_(x)
        CudaTimer().stop(f'layer{self.layer_id}')
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        H, W = self.input_resolution
        num_windows = H * W / self.window_size / self.window_size
        block_flops = dict(
            norm1=5 * H * W * self.dim,
            roll1=0, # ignore
            window_partition=0, # ignore
            attn=num_windows * self.attn.flops(self.window_size * self.window_size),
            roll2=0, # ignore
            attn_dropout=H * W * self.dim,
            atnn_residual=H * W * self.dim,
            norm2=5 * H * W * self.dim,
            mlp=self.mlp.flops(H * W),
            mlp_drop=H * W * self.dim,
            mlp_residual=H * W * self.dim,
        )
        return sum(block_flops.values())


class PatchMerging(PipeStage):
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

        H, W = self.input_resolution
        assert args.bs // (args.micro_bs * args.dp_size) != 0
        self.inputs_info = (
            ((args.micro_bs, H * W, self.dim),),
            (torch.float32 if not args.fp16 else torch.float16,)
        )
        self.outputs_info = (
            ((args.micro_bs, (H // 2) * (W // 2), self.dim * 2),),
            (torch.float32 if not args.fp16 else torch.float16,)
        )

    def forward_(self, x):
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

    def forward(self, x):
        x = checkpoint.checkpoint(self.forward_, x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


def create_basic_layter(dim, input_resolution, depth, num_heads, window_size,
                        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                        drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                        layer_id=None, start_id=0):
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
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    # swin transformer layers
    blocks = [SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if ((i + start_id) % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_coshard=args.use_coshard, layer_id=layer_id)
              for i in range(depth)]
    # patch merging layer
    if downsample is not None:
        downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        blocks.append(downsample)
    return blocks


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


class SwinTransformer(PipeStage):
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
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.set_pipeline(_pp_global_ranks)

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        total_layers = [3, 3, depths[2] + 1, 2]
        # pipeline split layers
        start, end = _layer_divisions[self.stage_local_rank]
        layers = []
        for i_layer in range(self.num_layers):
            layer_start = sum(total_layers[:i_layer])
            layer_end = sum(total_layers[:i_layer+1])
            if max(layer_start, start) >= min(layer_end, end):
                continue
            have_downsample = start < layer_end and layer_end <= end and i_layer < self.num_layers - 1
            layer_start_id = max(layer_start, start) - layer_start
            layer_num = min(layer_end, end) - max(layer_start, start)
            layer_num = layer_num if not have_downsample else layer_num - 1
            assert layer_num >= 1
            blocks = create_basic_layter(dim=int(embed_dim * 2 ** i_layer),
                                        input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                          self.patches_resolution[1] // (2 ** i_layer)),
                                        depth=layer_num,
                                        num_heads=num_heads[i_layer],
                                        window_size=window_size,
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=PatchMerging if have_downsample else None,
                                        layer_id=i_layer, start_id=layer_start_id)
            layers += blocks
        assert (end - start) == len(layers), f"layer num not equal, [{start}, {end}) != {len(layers)} "
        torch.distributed.barrier()
        self.layers = torch.nn.ModuleList(layers)
        print_each_rank(f'initialized {len(self.layers)} layers ranging from [{start}, {end})')

        self.inputs_info = self.layers[0].inputs_info
        self.outputs_info = self.layers[-1].outputs_info

        # preprocess
        if self.is_first_stage:
            print(f'rank [{torch.distributed.get_rank()}]: initializing pre-process...')
            # split image into non-overlapping patches
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
            num_patches = self.patch_embed.num_patches
            # absolute position embedding
            if self.ape:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
                trunc_normal_(self.absolute_pos_embed, std=.02)
            # dropout
            self.pos_drop = nn.Dropout(p=drop_rate)

            self.inputs_info = ((), ())

        # post-process
        if self.is_last_stage:
            print(f'rank [{torch.distributed.get_rank()}]: initializing post-process...')
            self.norm = norm_layer(self.num_features)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            self.criterion = nn.CrossEntropyLoss()
            
            self.outputs_info = (
                (1,),
                torch.float32 if args.fp16 else torch.float16
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x = None):
        if self.is_first_stage:
            CudaTimer().start('pre-process')
            x, _ = self.data
            x = self.patch_embed(x)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)
            CudaTimer().stop('pre-process')

        for layer in self.layers:
            x = layer(x)

        if self.is_last_stage:
            CudaTimer().start('post-process')
            _, labels = self.data

            def _post_process(x):
                x = self.norm(x)  # B L C
                x = self.avgpool(x.transpose(1, 2))  # B C 1
                x = torch.flatten(x, 1)
                x = self.head(x)
                return x

            x = checkpoint.checkpoint(_post_process, x)
            x = self.criterion(x, labels)
            CudaTimer().stop('post-process')

        return x

    def flops(self):
        flops = 0
        if self.is_first_stage:
            flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        if self.is_last_stage:
            flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
            flops += self.num_features * self.num_classes
        return flops


class ImageDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int, img_size: int, num_classes: int):

        self.bs = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        super().__init__(
            shapes=([batch_size, 3, img_size, img_size,],
                    [batch_size],
            ),
            dtypes=(torch.float if not args.fp16 else torch.float16, torch.int),
            batch_dims=(0, 0)
        )
        self.samples = [self.random_sample()]
        
    def random_sample(self):
        img = torch.rand(
            *(self.bs, 3, self.img_size, self.img_size),
            dtype=torch.float if not args.fp16 else torch.float16,
            device=torch.cuda.current_device()
        )
        labels = torch.randint(
            0, self.num_classes,
            size=(self.bs,),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        return (img, labels)
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]


def train():

    cfg = Config()
    model = SwinTransformer(img_size=cfg.img_size,
                            patch_size=4,
                            in_chans=3,
                            num_classes=cfg.num_classes,
                            embed_dim=cfg.embed_dim,
                            depths=cfg.depths,
                            num_heads=cfg.num_heads,
                            window_size=cfg.window_size,
                            mlp_ratio=cfg.mlp_ratio,
                            qkv_bias=cfg.qkv_bias,
                            qk_scale=cfg.qk_scale,
                            drop_rate=cfg.drop_rate,
                            drop_path_rate=cfg.drop_path_rate,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False)
    nparams = sum([param.numel() for param in model.parameters()])
    forward_flops = model.flops()
    tflops = forward_flops * 4 / (1e12) # forward + recompute-forward + backward (2x)
    print_each_rank(f'Model Params#: {nparams} | TFlops: {tflops}')
    if args.fp16:
        model = model.half()
    model = model.cuda()
    dataloader = ImageDataLoader(args.micro_bs, cfg.img_size, cfg.num_classes)
    if _dp_reducer is not None:
        for param in model.parameters():
            _dp_reducer.add_param(param)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    print_each_rank('model weight consumpition:', rank_only=0)
    memory_summary()

    def train_iter(model, dataloader):
        num_microbatch = args.bs // (args.dp_size * args.micro_bs)
        if _pp_group != -1:
            _schedule(model, dataloader, num_microbatch)
        else:
            for _ in range(num_microbatch):
                model.data = next(dataloader)
                loss = model()
                loss.backward()
        if _dp_reducer is not None:
            _dp_reducer.allreduce()

    CudaTimer(enable=False)
    iter_num = 6
    for step in range(iter_num):

        # if step == 0:
        #     model.data = next(dataloader)
        #     model_summary(model, (), rank_only=1)

        if step >= 2:
            CudaTimer(enable=True).start('e2e')

        # training
        train_iter(model, dataloader)

        # if step == 0:
        #     print_each_rank('passed first iteration', rank_only=0)
        #     print_each_rank('memory consumption before optimizer:', rank_only=0)
        #     memory_summary()

        optimizer.step()
        optimizer.zero_grad()

        if step >= 2:
            CudaTimer().stop('e2e')

        if step == 0:
            print_each_rank('memory consumption after optimizer:', rank_only=0)
            memory_summary()
        
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-2, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-2)
    memory_summary()

train()
