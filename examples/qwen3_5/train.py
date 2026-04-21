# Qwen3.5-4B nnscaler training with MTP (Multi-Token Prediction)
# Usage: torchrun --nproc_per_node=8 train.py
#
# Runs actual training on random data using all 8 GPUs.
# Includes MTP module: predicts token[i+2] from (hidden[i], embed[i+1]).

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Apply modifier BEFORE importing the model (registers ops, patches forward)
import qwen3_5_modifier  # noqa: F401

import nnscaler
import nnscaler.utils
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import (
    ComputeConfig,
    DataloaderConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerArgs,
)
from nnscaler.runtime.f16_optimizer import MixedPrecisionAdamW

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
)
from qwen3_5_modifier import nnscaler_chunked_cross_entropy


# Config (Qwen3.5-4B architecture)
SEQ_LEN = 131072  # 128K
NUM_HIDDEN_LAYERS = 36  # full Qwen3.5-4B

config = Qwen3_5TextConfig(
    hidden_size=2560,
    intermediate_size=9216,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=256,
    vocab_size=248320,
    rms_norm_eps=1e-6,
    hidden_act='silu',
    attention_bias=False,
    attention_dropout=0.0,
    tie_word_embeddings=True,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    max_position_embeddings=131072,
    use_cache=False,
    rope_parameters={
        'mrope_interleaved': True,
        'mrope_section': [11, 11, 10],
        'rope_type': 'default',
        'rope_theta': 10000000,
        'partial_rotary_factor': 0.25,
    },
)
config._attn_implementation = 'flash_attention_2'
# Extend layer_types so MTP decoder layer (layer_idx=36) maps to 'full_attention'
config.layer_types.append('full_attention')


class Qwen3_5MTP(nn.Module):
    """Multi-Token Prediction: 1 decoder layer that predicts token[i+2]."""
    def __init__(self, config):
        super().__init__()
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layer = Qwen3_5DecoderLayer(config, layer_idx=config.num_hidden_layers)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class WrapperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.causal_lm = Qwen3_5ForCausalLM(config)
        self.mtp = Qwen3_5MTP(config)

    def forward(self, data):
        input_ids = data['input_ids']
        labels = data['labels']
        batch_size, seq_len = input_ids.shape

        # 1. Backbone (TextModel) forward — get hidden states
        outputs = self.causal_lm.model(input_ids=input_ids)
        hidden_states = outputs.last_hidden_state  # (B, L, D)

        # 2. Main causal LM loss (predict token[i+1] from hidden[i])
        #    Use chunked CE to avoid materializing full (B, 128K, 248K) logits
        main_loss = nnscaler_chunked_cross_entropy(
            hidden_states[:, :-1, :].contiguous(),
            self.causal_lm.lm_head.weight,
            labels[:, 1:].contiguous(),
        )

        # 3. MTP: predict token[i+2] from (hidden[i], embed[i+1])
        h = hidden_states[:, :-1, :]  # (B, L-1, D)
        shifted_embeds = self.causal_lm.model.embed_tokens(input_ids[:, 1:])  # (B, L-1, D)

        h_norm = self.mtp.pre_fc_norm_hidden(h)
        e_norm = self.mtp.pre_fc_norm_embedding(shifted_embeds)
        x = self.mtp.fc(torch.cat([h_norm, e_norm], dim=-1))  # (B, L-1, D)

        # Position embeddings for MTP attention (3D mRoPE: temporal, height, width)
        mtp_len = seq_len - 1
        pos_ids = torch.arange(mtp_len, device=input_ids.device)
        pos_ids = pos_ids.view(1, 1, -1).expand(3, batch_size, -1)  # (3, B, L-1)
        position_embeddings = self.causal_lm.model.rotary_emb(x, pos_ids)

        # MTP decoder layer (full_attention, causal via module.is_causal)
        x = self.mtp.layer(x, position_embeddings=position_embeddings)
        x = self.mtp.norm(x)

        # MTP loss: logits[i] predicts token[i+2], target is labels[2:]
        mtp_loss = nnscaler_chunked_cross_entropy(
            x[:, :-1, :].contiguous(),
            self.causal_lm.lm_head.weight,
            labels[:, 2:].contiguous(),
        )

        return main_loss + mtp_loss


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size, seq_len, size=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,))}


def collate(samples):
    if not samples:
        return {}
    input_ids = torch.stack([s['input_ids'] for s in samples])
    return {'input_ids': input_ids, 'labels': input_ids.clone()}


def main():
    nnscaler.utils.set_default_logger_level('INFO')

    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    max_train_steps = 10
    print(f'[qwen3.5] world_size={world_size}, layers={NUM_HIDDEN_LAYERS}, seq_len={SEQ_LEN}')

    compute_config = ComputeConfig(
        plan_ngpus=4,
        runtime_ngpus=world_size,
        constant_folding=True,
        use_zero=True,
        use_end2end=True,
        trace_strategy='cuda_run_cpu_offload',
        pas_config={
            'recompute_modules': 'Qwen3_5DecoderLayer',
        },
    )

    model_config = ModelConfig(
        type=WrapperModel,
        args={},
    )

    optimizer_config = OptimizerConfig(
        type=MixedPrecisionAdamW,
        args={'lr': 2e-5, 'fused': True},
        clip_gnorm=1.0,
    )

    dataset_config = DatasetConfig(
        type=RandomTokenDataset,
        train_args={
            'vocab_size': config.vocab_size,
            'seq_len': SEQ_LEN,
            'size': 10000,
        },
    )

    dataloader_config = DataloaderConfig(
        train_args={'collate_fn': collate, 'drop_last': True},
    )

    trainer_args = TrainerArgs(
        compute_config=compute_config,
        pas_policy='autodist',
        model=model_config,
        optimizer=optimizer_config,
        dataset=dataset_config,
        dataloader=dataloader_config,
        precision='bf16',
        grad_accumulation_steps=1,
        max_train_steps=max_train_steps,
        seed=42,
    )

    trainer = Trainer(train_args=trainer_args)
    trainer.run()

    # FLOPS Calculation (rank 0 only)
    if rank == 0:
        # Count model parameters
        temp_model = WrapperModel()
        total_params = sum(p.numel() for p in temp_model.parameters())
        del temp_model

        B = 1  # batch_size per GPU
        S = SEQ_LEN
        N_gpus = world_size

        # 1. Parameter matmul FLOPS (forward): 2 * params * tokens_per_step
        tokens_per_step = B * S * N_gpus
        param_flops = 2 * total_params * tokens_per_step

        # 2. Full-attention FLOPS (forward): 4 * n_layers * B * n_heads * S^2 * head_dim
        #    (2 for QK^T, 2 for attn@V)
        n_full_attn_backbone = sum(1 for t in config.layer_types[:NUM_HIDDEN_LAYERS] if t == 'full_attention')
        n_full_attn_mtp = 1
        n_full_attn_total = n_full_attn_backbone + n_full_attn_mtp
        n_heads = config.num_attention_heads
        head_dim = config.head_dim
        attn_flops = 4 * n_full_attn_total * (B * N_gpus) * n_heads * S * S * head_dim

        # 3. Training = 3x forward (forward + backward)
        forward_flops = param_flops + attn_flops
        train_flops_per_step = 3 * forward_flops

        print(f'\n{"=" * 60}')
        print(f'[FLOPS Summary]')
        print(f'  Total params: {total_params / 1e9:.3f}B')
        print(f'  Seq length: {S}')
        print(f'  Global batch: {B * N_gpus} (B={B} x {N_gpus} GPUs)')
        print(f'  Full-attn layers: {n_full_attn_total} ({n_full_attn_backbone} backbone + {n_full_attn_mtp} MTP)')
        print(f'  Tokens/step: {tokens_per_step:,}')
        print(f'  Forward FLOPS/step: {forward_flops / 1e12:.2f} TFLOPS')
        print(f'  Training FLOPS/step (3x fwd): {train_flops_per_step / 1e12:.2f} TFLOPS')
        print(f'  --> Divide by train_wall (seconds/step) to get TFLOPS throughput')
        print(f'  --> E.g. if train_wall=X: {train_flops_per_step / 1e12:.2f} / X = TFLOPS')
        print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
