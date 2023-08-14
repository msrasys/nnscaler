import torch

from megatron import initialize_megatron
from megatron.training import get_args, ModelType
from megatron.arguments import core_transformer_config_from_args
from megatron.model import GPTModel
from megatron.model.fused_bias_gelu import GeLUFunction
from megatron.core.tensor_parallel import VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear

class GPT2Model(GPTModel):
    def __init__(self, config, num_tokentypes=0, parallel_output=True, pre_process=True, post_process=True):
        super().__init__(config, num_tokentypes, parallel_output, pre_process, post_process)

    def forward(self, src_tokens, target, ntokens):
        position_ids = torch.arange(0, src_tokens.shape[1], 1).unsqueeze(0).expand_as(src_tokens)
        attention_mask = (torch.tril(torch.ones(1, 1, src_tokens.shape[1], src_tokens.shape[1])) < 0.5).bool()
        res = super().forward(src_tokens, position_ids, attention_mask, labels=target)
        return res, ntokens, {'loss': res, 'ntokens': ntokens, 'nsentences': src_tokens.shape[0], 'sample_size': ntokens}


def build_model() -> GPT2Model:
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    get_args().model_type = ModelType.encoder_or_decoder
    config = core_transformer_config_from_args(get_args())
    model = GPT2Model(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True
    )

    return model
