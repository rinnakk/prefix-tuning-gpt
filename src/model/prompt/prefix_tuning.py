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
from typing import List, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class PrefixWrapper(nn.Module):
    def __init__(self, base_model, prefix_encoder):
        super().__init__()
        self.base_model = base_model
        self.prefix_encoder = prefix_encoder


class PrefixEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        self.prefix_seq_len = config.prefix_seq_len
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.prefix_dropout_rate = getattr(config, "prefix_dropout_rate", 0.0)

        # Model
        self.input_tokens = torch.arange(self.prefix_seq_len).long()
        self.prefix_wte = nn.Embedding(self.prefix_seq_len, config.input_dim)
        # Since prefix-tuning append prefix to each layer, the shape is prefix_seq_len, n_layer, 2(query,key), n_embd
        self.prefix_mlp = nn.Sequential(
            nn.Linear(config.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, config.n_layer * 2 * config.n_embd),
        )
        self.prefix_dropout = nn.Dropout(self.prefix_dropout_rate)

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head

    def forward(
            self,
            batch_size: int,
            device: Union[str, torch.device],
    ):
        """
        Return query & key values from prefix
        """
        input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        # Forward
        input_embd = self.prefix_wte(input_tokens)
        past_key_values = self.prefix_mlp(input_embd)

        # Resize
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_seq_len,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        # Dropout
        past_key_values = self.prefix_dropout(past_key_values)
        
        # Transpose -> [match_n_layer*2, batch_size, match_n_head, prefix_seq_len, match_n_embd]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


def init_prefix(local_rank, config, model, prefix_config, prefix_encoder, device, n_epochs=50, learning_rate=0.0001, weight_decay=0.01):
    if config.world_size > 1:
        to_train_prefix_encoder = DDP(
            prefix_encoder,
            device_ids=[local_rank]
        )
    else:
        to_train_prefix_encoder = prefix_encoder

    token_ids = [list(range(prefix_config.prefix_seq_len))]
    token_ids = torch.LongTensor(token_ids).to(device)
    
    labels = []
    with torch.no_grad():
        outputs = model.forward(input_ids=token_ids, return_dict=True, use_cache=True)
        past_key_values = outputs.past_key_values
        for p in past_key_values:
            labels.append(torch.stack(p))
        labels = torch.cat(tuple(labels), dim=0)
    labels = labels.to(device)

    param_optimizer = list(to_train_prefix_encoder.named_parameters())
    no_decay = ["bias"]  # no decay for bias
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer_prefix = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    to_train_prefix_encoder.train()
    for _ in range(n_epochs):
        optimizer_prefix.zero_grad()

        prompt_outputs = to_train_prefix_encoder.forward(
            batch_size=1, device=device
        )
        prompt_outputs = torch.cat(prompt_outputs, dim=0)

        loss_metrics = torch.nn.MSELoss()
        loss = loss_metrics(prompt_outputs, labels)
        loss.backward()
        optimizer_prefix.step()