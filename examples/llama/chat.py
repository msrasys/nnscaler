"""
pip install fire sentencepiece

PYTHONPATH=.:$PYTHONPATH torchrun \
   --nproc_per_node=1 \
   examples/llama/chat.py \
       --ckpt_dir=/home/t-zhiqilin/llama/llama-2-7b-chat \
       --tokenizer_path=/home/t-zhiqilin/llama/tokenizer.model \
       --max_seq_len 512 --max_batch_size 8 --temperature 0 \
       --use-cube
"""

from typing import Optional

import fire
import logging

from examples.llama.generation import Llama

import nnscaler
from nnscaler.utils import set_default_logger_level

nnscaler.init()
set_default_logger_level(level=logging.WARNING)
logging.getLogger('nnscaler.compiler').setLevel(logging.INFO)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    use_cube: bool = False,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        use_cube=use_cube,
    )

    dialog = [
        {"role": "system", "content":
          "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature."},
    ]

    print('Assistant: Hello, this is Llama 2')
    while True:
        user_content = input("Prompt >> ")
        dialog.append({"role": "user", "content": user_content})
        result = generator.chat_completion(
            [dialog],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        assit_content = result['generation']['content']
        print(f"{result['generation']['role'].capitalize()}: {assit_content}")
        dialog.append({"role": "assistant", "content": assit_content})


if __name__ == "__main__":
    fire.Fire(main)