# coding=utf-8
# Copyright 2022 The rinna Co. Ltd. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference with prefix-tuning weights"""

import argparse

from transformers import AutoTokenizer
import torch

from util import step_decode, generate_causal_mask
from model.gpt.modeling_gpt2 import GPT2LMHeadModel
from model.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM


def generate(prompt, model, prefix_weights, tokenizer, prefix_seq_len, max_output_len, decode_method, top_p, top_k):
    if prompt == "":
        input_ids = [tokenizer.bos_token_id]
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    position_ids = list(range(0, len(input_ids)))
    
    input_ids = [input_ids]
    position_ids = [position_ids]
        
    batch_size = len(input_ids)
    seq_lens = [len(seq) for seq in input_ids]
    max_seq_len = max(seq_lens)

    # padding input sequences
    input_ids = [[tokenizer.pad_token_id]*(max_seq_len-len(seq)) + seq for seq in input_ids]
    position_ids = [[0]*(max_seq_len-len(seq)) + seq for seq in position_ids]
    
    if prefix_weights is not None:
        # get past_key_values
        past_key_values_prompt = prefix_weights.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        # [match_n_layer*2, batch_size, match_n_head, prefix_seq_len, match_n_embd]
        past_key_values_prompt = past_key_values_prompt.permute([2, 0, 3, 1, 4]).split(2)

    # unroll
    symbols = [[] for _ in range(batch_size)]
    early_stoppings = [0 for _ in range(batch_size)]
    # prepare init inputs
    step_input_ids = torch.LongTensor(input_ids).to(model.device)
    step_position_ids = torch.LongTensor(position_ids).to(model.device)
    if prefix_weights is not None:
        step_past = past_key_values_prompt
        input2prompt_attn_mask = torch.ones(batch_size, max_seq_len, prefix_seq_len).bool().to(model.device)
        input2input_attn_mask = (
            (step_input_ids != tokenizer.pad_token_id).unsqueeze(1).repeat(1, max_seq_len, 1)
            & generate_causal_mask(max_seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(model.device)
        )
        step_attn_mask = torch.cat([input2prompt_attn_mask, input2input_attn_mask], dim=2)
    else:
        step_past = None
        step_attn_mask = (step_input_ids != tokenizer.pad_token_id).unsqueeze(1).repeat(1, max_seq_len, 1)

    for step in range(max_output_len):
        # forward
        step_outputs = model.forward(
            input_ids=step_input_ids,
            position_ids=step_position_ids,
            attention_mask=step_attn_mask,
            past_key_values=step_past
        )

        # decode
        step_logits = step_outputs["logits"][:, -1, :]
        step_symbol = step_decode(
            logits=step_logits,
            decode_method=decode_method,
            top_p=top_p,
            top_k=top_k
        )

        # collect step outputs
        for batch_idx in range(batch_size):
            if step_symbol[batch_idx] == tokenizer.eos_token_id:
                early_stoppings[batch_idx] = 1
            else:
                if early_stoppings[batch_idx] == 0:
                    symbols[batch_idx].append(step_symbol[batch_idx].item())
        if sum(early_stoppings) == batch_size:
            break

        # inputs at next step
        step_position_ids = torch.LongTensor([seq_len+step+1 for seq_len in seq_lens]).to(model.device)
        step_input_ids = step_symbol
        next_step_attn_mask = torch.ones(batch_size, 1, 1).bool().to(model.device)
        step_attn_mask = torch.cat([step_attn_mask[:, -1:, :], next_step_attn_mask], dim=-1)
        step_past = step_outputs["past_key_values"]

    text = tokenizer.decode(symbols[0])
        
    return text
    

def inference(config):
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # build model and tokenizer
    print("Building model...")
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_dir, use_fast=False)
    if config.model_type == "gpt":
        model = GPT2LMHeadModel.from_pretrained(config.pretrained_model_dir)
        model_n_embd = model.config.n_embd
        model_n_layer = model.config.n_layer
        model_n_head = model.config.n_head
    elif config.model_type == "gpt-neox":
        model = GPTNeoXForCausalLM.from_pretrained(config.pretrained_model_dir)
        model_n_embd = model.config.hidden_size
        model_n_layer = model.config.num_hidden_layers
        model_n_head = model.config.num_attention_heads
    model = model.eval()
    model = model.to(DEVICE)

    # build prefix
    if config.prefix_checkpoint_path:
        print("Building prefix-tuning model...")
        prefix_weights = torch.load(
            config.prefix_checkpoint_path,
            map_location=lambda storage, loc: storage
        )
        prefix_weights = prefix_weights.view(
            config.prefix_seq_len,
            model_n_layer*2,
            model_n_head,
            model_n_embd // model_n_head
        ).to(DEVICE)
        prefix_seq_len = prefix_weights.size(0)
    else:
        prefix_weights = None
        prefix_seq_len = None

    with torch.no_grad():
        while True:
            try:
                prompt = input("Prompt: (Use Ctrl+D or Ctrl+C to exit)")
                generation = generate(
                    prompt,
                    model,
                    prefix_weights,
                    tokenizer,
                    prefix_seq_len,
                    config.max_output_len,
                    config.decode_method,
                    config.top_p,
                    config.top_k
                )
                print(f"{prompt}{generation}")
            except BaseException as e:
                if isinstance(e, EOFError):
                    exit(0)
                elif isinstance(e, KeyboardInterrupt):
                    exit(0)
                else:
                    print(e)
                    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt-neox", choices=["gpt-neox", "gpt"])
    parser.add_argument("--pretrained_model_dir", type=str, required=True)
    parser.add_argument("--prefix_checkpoint_path", help="path to saved prefix checkpoint files")
    parser.add_argument("--prefix_seq_len", type=int, default=10)

    parser.add_argument("--max_output_len", type=int, default=128)
    parser.add_argument("--decode_method", type=str, default="nucleus")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)

    config = parser.parse_args()

    inference(config)
