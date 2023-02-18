import torch
import cube
import torch.distributed
from typing import Tuple

# (N L) emb; emb * exp_num -> (N L), 1(part_idx)
@cube.graph.parser.register('*, * -> *')
@torch.jit.ignore
def gating_func(x, gate_w) -> torch.Tensor:
  # assert top_k == 1
  affinities = torch.matmul(x, gate_w)
  # print(f'affinities = {affinities}')
  dst_pid_list = torch.argmax(affinities, -1)
  # print(f'dst_pid_list = {dst_pid_list}')
  return dst_pid_list

# split tokens into groups by target expert
@cube.graph.parser.register('* -> *')
@torch.jit.ignore
def split_tokens_by_eid(tokens, eids, expert_num):
  print(f"tokens = {tokens}, shape {tokens.size()}")
  print(f"eids = {eids}, shape {eids.size()}")
  reshape_needed = list(tokens.size()) != list(eids.size())
  reshape_feat_dim = list(tokens.size())[-1]
  print("##### reshape_feat_dim = " + str(reshape_feat_dim))
  if reshape_needed:
    vid_part_extend = torch.unsqueeze(eids, 2).repeat(1, 1, reshape_feat_dim)
    print("vid_part_extend = " + str(vid_part_extend))
  else:
    vid_part_extend = eids

  token_lists = []
  for exp_id in range(0, expert_num):
    print("exp_id = " + str(exp_id))
    mask = (vid_part_extend == exp_id)
    print("mask = " + str(mask))
    parted_tokens = torch.masked_select(tokens, mask)
    if reshape_needed:
      parted_tokens = parted_tokens.reshape(-1, reshape_feat_dim)
    print("parted_tokens = " + str(parted_tokens))
    token_lists.append(parted_tokens)
  return token_lists


@cube.graph.parser.register('* -> *')
@torch.jit.ignore
def samesize_all_gather(tensor: torch.Tensor):
  tensor_list = [torch.zeros_like(tensor) for _ in
                 range(torch.distributed.get_world_size())]
  torch.distributed.all_gather(tensor_list, tensor)
  return torch.stack(tensor_list)


@cube.graph.parser.register('* -> *')
@torch.jit.ignore
def nonvarsize_gather(tensor: torch.Tensor, dst):
  tensor_list = [torch.zeros_like(tensor) for _ in
                 range(torch.distributed.get_world_size())] if torch.distributed.get_rank() == dst else None
  torch.distributed.gather(tensor, tensor_list, dst)

  return torch.cat(tensor_list) if torch.distributed.get_rank() == dst else None


@cube.graph.parser.register('* -> *')
@torch.jit.ignore
def varsize_tensor_gather(tensor: torch.Tensor, dst):
  tensor = tensor.contiguous()
  # cuda_device = f'cuda:{torch.distributed.get_rank()}'
  print(f'tensor.get_device() = {tensor.get_device()}')
  size_tens = torch.tensor([tensor.shape[0]], dtype=tensor.dtype, device=f'cuda:{tensor.get_device()}')
  print(f'size_tens.get_device() = {size_tens.get_device()}')
  size_tens = samesize_all_gather(size_tens)
  print(f"size_tens = {size_tens}, tensor.shape[1:] = {tensor.shape[1:]}")

  max_size = size_tens.max().int().item()
  padded = torch.empty(max_size, *tensor.shape[1:], dtype=tensor.dtype, device=f'cuda:{tensor.get_device()}')
  padded[:tensor.shape[0]] = tensor

  ga = nonvarsize_gather(padded, dst)
  print(f" tensor = {tensor}; padded = {padded}; ga = {ga}")

  if torch.distributed.get_rank() != dst:  # not this rank as dst
    return []

  slices = []
  for i, sz in enumerate(size_tens):
    start_idx = i * max_size
    end_idx = start_idx + sz.int().item()
    print("start_idx = " + str(start_idx))
    print("end_idx = " + str(end_idx))

    if end_idx > start_idx:
      print("ga[start_idx:end_idx] = " + str(ga[start_idx:end_idx]))
      slices.append(ga[start_idx:end_idx])
      # print("slices = " + str(slices))
    else:
      slices.append(torch.empty((0, *tensor.shape[1:]), dtype=tensor.dtype, device=f'cuda:{tensor.get_device()}'))
      # slices.append(torch.tensor([], dtype=tensor.dtype).resize(0, 3))
  return slices


@cube.graph.parser.register('* -> *')
@torch.jit.ignore
def all_to_all_token(input_list):
  print(f'***** all_to_all_token.input_list = {input_list}')
  data_type = input_list[0].dtype
  print(data_type)
  ret = []
  for i in range(len(input_list)):
    gather_list = varsize_tensor_gather(input_list[i], i)  # new replacement
    if i == torch.distributed.get_rank(): #TODO check local_rank
      ret = gather_list
  print(f'***** all_to_all_token.output_list = {ret}')
  return ret


# N * 1, N * emb -> M * 1, M * emd
@cube.graph.parser.register('*, * -> *')
@torch.jit.ignore
def send_to_experts(dst_pid_list, x, expert_num: int) -> Tuple[torch.Tensor]:
  # send to remote and recv from remote
  token_lists = split_tokens_by_eid(x, dst_pid_list, expert_num)
  print(f'### token_lists = {token_lists}')
  local_token_lists = all_to_all_token(token_lists)  # exchange idx
  print(f'### local_token_lists = {local_token_lists}')
  return local_token_lists


# M * 1, M * emd -> N * 1, N * emb
@cube.graph.parser.register('*, * -> *')
@torch.jit.ignore
def recv_from_experts(dst_pid_list: torch.Tensor, new_local_token_lists: torch.Tensor, expert_num: int) -> Tuple[torch.Tensor]:
  local_token_lists = all_to_all_token(new_local_token_lists)
  print(f'### [return] local_token_lists = {local_token_lists}')

  vid_part_np = dst_pid_list.detach().flatten().cpu().tolist() #TODO vid_part_np = dst_pid_list.detach().cpu().numpy()
  print("vid_part_np = " + str(vid_part_np))
  # part_count = {}
  # for i in range(expert_num):
  #   part_count[i] = 0
  part_count = [0 for i in range(expert_num)]
  print(f'part_count = {part_count}')

  embed_list = []
  for i in range(len(vid_part_np)):
    pid = vid_part_np[i]
    print(f'pid = {pid}')
    offset = part_count[pid]
    part_count[pid] += 1
    embed_list.append(local_token_lists[pid][offset])

  # print("### embed_list = " + str(embed_list))
  embed = torch.stack(embed_list)
  print("### final embed = " + str(embed))

  return embed


# @cube.graph.parser.register('L^ N E^, H+ E^, H+, E^ H+ -> L^ N E^', name='feedforward_moe')
@cube.graph.parser.register('L^ N E^, H+ E^, H+, E^ H+, E^ K-> L^ N E^', name='feedforward_moe')
@torch.jit.ignore
def feedforward_moe(x: torch.Tensor,
                proj1: torch.Tensor, proj1_bias: torch.Tensor,
                proj2: torch.Tensor,
                gate_w: torch.Tensor,
                dropout: float,
                is_training: bool = True,
                expert_num: int = 1) -> torch.Tensor:
    #gating
    dst_pid_list = gating_func(x, gate_w)
    #shuffle tokens
    # src_pid_list, x_local
    local_token_lists = send_to_experts(dst_pid_list, x, expert_num)

    new_local_token_lists = []
    for x_local in local_token_lists:
      #local expert
      with torch.no_grad():
        print(f'#### checking ####', x_local, proj1, proj1_bias)
        x_local = torch.nn.functional.linear(x_local, proj1, proj1_bias)
        x_local = torch.nn.functional.gelu(x_local)
        #TODO FIXME x_local = torch.nn.functional.dropout(x_local, dropout, is_training, False)
        x_local = torch.nn.functional.linear(x_local, proj2, None)
        new_local_token_lists.append(x_local)

    #shuffle back tokens
    print(f'### new_local_token_lists = {new_local_token_lists}')
    x = recv_from_experts(dst_pid_list, new_local_token_lists, expert_num)
    return x


class MoEMLP(torch.nn.Module):
  def __init__(self, embed_dim: int, hidden_dim: int, dropout: float, expert_num: int = 1):
    super().__init__()
    # self.proj1 = torch.nn.Parameter(torch.ones((hidden_dim // expert_num, embed_dim)))  # TODO fix me empty
    # self.proj1_bias = torch.nn.Parameter(torch.empty((hidden_dim // expert_num,)))
    # self.proj2 = torch.nn.Parameter(torch.ones((embed_dim, hidden_dim // expert_num)))  # TODO fix me empty
    # self.proj2_bias = torch.nn.Parameter(torch.empty((embed_dim,)))
    # self.gate_w = torch.nn.Parameter(torch.rand((embed_dim, expert_num)))
    self.proj1 = torch.nn.Parameter(torch.rand((hidden_dim, embed_dim)))  # TODO fix me empty
    self.proj1_bias = torch.nn.Parameter(torch.rand((hidden_dim,)))
    self.proj2 = torch.nn.Parameter(torch.rand((embed_dim, hidden_dim)))  # TODO fix me empty
    self.proj2_bias = torch.nn.Parameter(torch.rand((embed_dim,)))
    self.gate_w = torch.nn.Parameter(torch.rand((embed_dim, expert_num)))
    self.dropout = dropout
    self.expert_num = expert_num

  def forward(self, x: torch.Tensor):
    x = feedforward_moe(x, self.proj1, self.proj1_bias,
                    self.proj2, self.gate_w, self.dropout, self.training, self.expert_num)
    x = x + self.proj2_bias
    return x