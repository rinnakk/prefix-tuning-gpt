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
"""Utilities"""

import collections

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_causal_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    return mask

    
def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_rank_0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def step_decode(logits, decode_method, top_k=0, top_p=0.0, temp=1.0, bad_token_ids=[]):
    logits = logits/temp

    if len(bad_token_ids) > 0:
        bad_token_ids = torch.LongTensor(bad_token_ids).to(logits.device)
        logits[:, bad_token_ids] = -float("inf")

    if decode_method == "greedy":
        symbol = logits.topk(1)[1]
    elif decode_method == "sample":
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        symbol = dist.sample().unsqueeze(1)
    elif decode_method == "nucleus":
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("inf")
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # keep the first token above the threshold
            sorted_indices_to_remove[..., 0] = 0

            for batch_idx in range(logits.size(0)):
                indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                logits[batch_idx, indices_to_remove] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        symbol = dist.sample().unsqueeze(1)
    else:
        raise Exception("unsupported generation type {}".format(decode_method))

    return symbol


class StatisticsReporter:
    def __init__(self):
        self.statistics = collections.defaultdict(list)

    def update_data(self, d):
        for k, v in d.items():
            if isinstance(v, (int, float)):
                self.statistics[k] += [v]

    def clear(self):
        self.statistics = collections.defaultdict(list)

    def to_string(self):
        string_list = []
        for k, v in sorted(list(self.statistics.items()), key=lambda x: x[0]):
            mean = np.mean(v)
            string_list.append("{}: {:.5g}".format(k, mean))
        return ", ".join(string_list)

    def get_value(self, key):
        if key in self.statistics:
            value = np.mean(self.statistics[key])
            return value
        else:
            return None

    def items(self):
        for k, v in self.statistics.items():
            yield k, v
