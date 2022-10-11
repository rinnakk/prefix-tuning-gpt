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
"""Example code for training prefix-tuning weights that generate a smileface emoji suffix"""

import json
import random
import time
import argparse
import os

import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import T5Tokenizer
import deepspeed
from deepspeed.ops.adam import FusedAdam as Adam

from data_source import DataSource, collate_fn
from model.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from model.gpt.modeling_gpt2 import GPT2LMHeadModel
from model.prompt.prefix_tuning import PrefixEncoder, PrefixWrapper
from util import StatisticsReporter, get_linear_schedule_with_warmup, print_rank_0, is_rank_0, count_parameters


def load_data_from_filepath(filepath, tokenizer, max_seq_len=64):
    with open(filepath, encoding="utf-8") as f:
        data = []
        for line in f:
            sample = json.loads(line.strip())
            _text = sample["text"]
            
            _sents = _text.split("\n")

            tmp_text = ""
            for sent in _sents:
                new_tmp_text = f"{tmp_text}\n{sent} 😃"
                new_tmp_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_tmp_text))
                if len(new_tmp_token_ids) > max_seq_len:
                    tmp_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tmp_text))
                    if len(tmp_token_ids) > 0:
                        data.append(([], tmp_token_ids))
                    tmp_text = f"{sent} 😃"
                else:
                    tmp_text = new_tmp_text
        return data


def init_prefix(local_rank, config, model, prefix_config, prefix_encoder, device, n_epochs=50, learning_rate=0.0001, weight_decay=0.01):
    if config.world_size > 1:
        to_train_prefix_encoder = DDP(
            prefix_encoder,
            device_ids=[local_rank]
        )
    else:
        to_train_prefix_encoder = prefix_encoder

    token_ids = list(range(prefix_config.prefix_seq_len))
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
    tqdm_bar = tqdm(range(n_epochs), desc="Epoch")
    for _ in tqdm_bar:
        optimizer_prefix.zero_grad()

        prompt_outputs = to_train_prefix_encoder.forward(
            batch_size=1, device=device
        )
        prompt_outputs = torch.cat(prompt_outputs, dim=0)

        loss_metrics = torch.nn.MSELoss()
        loss = loss_metrics(prompt_outputs, labels)
        tqdm_bar.desc = "Epoch loss: {:.2e}".format(loss.item())
        loss.backward()
        optimizer_prefix.step()


def forward_step(model, prefix_encoder, tokenizer, batch_data):
    input_ids = []
    labels = []
    for prompt, target in batch_data:
        if len(prompt) != 0:
            input_ids.append(prompt + target)
            labels.append([tokenizer.pad_token_id]*(len(prompt)-1) + target + [tokenizer.eos_token_id])
        else:
            input_ids.append([tokenizer.bos_token_id] + target)
            labels.append(target + [tokenizer.eos_token_id])
            input_ids.append(target)
            labels.append(target[1:] + [tokenizer.eos_token_id])
        
    batch_size = len(input_ids)
    max_seq_len = max([len(seq) for seq in input_ids])

    # padding input sequences
    input_ids = [seq + [tokenizer.pad_token_id]*(max_seq_len-len(seq)) for seq in input_ids]
    labels = [seq + [tokenizer.pad_token_id]*(max_seq_len-len(seq)) for seq in labels]

    # convert to tensors
    input_ids = torch.LongTensor(input_ids).to(model.device)
    labels = torch.LongTensor(labels).to(model.device)
    
    # forward
    past_key_values_prompt = prefix_encoder(
        batch_size=batch_size, device=model.device
    )
    model_outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values_prompt,
        return_dict=True
    )
    loss = F.cross_entropy(
        model_outputs["logits"].view(-1, model_outputs["logits"].size(-1)),
        labels.view(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction="mean"
    )
    with torch.no_grad():
        ppl = loss.exp()

    return loss, ppl


def training(local_rank, config):
    global_rank = config.rank
    print(f"local rank: {[local_rank]}, global_rank: {[global_rank]}")

    # set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # multi-gpu init
    if config.deepspeed:
        deepspeed.init_distributed()
    elif torch.cuda.is_available() and config.world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=config.world_size,
            rank=global_rank
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(config.local_rank)
        DEVICE = torch.device("cuda", local_rank)
    else:
        DEVICE = torch.device("cpu")

    # build tokenizer
    if config.model_type == "gpt":
        tokenizer = T5Tokenizer.from_pretrained(config.pretrained_model_dir)
    elif config.model_type == "gpt-neox":
        tokenizer = T5Tokenizer.from_pretrained(config.pretrained_model_dir)
    
    # build data source and reporters
    trn_reporter = StatisticsReporter()
    dev_reporter = StatisticsReporter()

    # get data filepaths)
    data_filepath = config.data_filepath
    data = load_data_from_filepath(data_filepath, tokenizer)
    print_rank_0(f"Origianl data size: {len(data)}")
    assert len(data) >= config.train_data_size + config.dev_data_size
    train_data = data[:config.train_data_size]
    dev_data = data[-config.dev_data_size:]

    train_data_source = DataSource(config, tokenizer, train_data, "train")
    print_rank_0(str(train_data_source.statistics))
    # single gpu or cpu
    if config.world_size == 1 or not torch.cuda.is_available():
        train_data_sampler = RandomSampler(
            train_data_source,
            replacement=False
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_data_source,
            batch_size=config.batch_size,
            sampler=train_data_sampler,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True
        )
    # multi gpus
    else:
        train_data_sampler = DistributedSampler(
            train_data_source,
            num_replicas=config.world_size,
            rank=global_rank
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_data_source,
            batch_size=config.batch_size,
            sampler=train_data_sampler,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False
        )

    print_rank_0("----- Loading dev data -----")
    dev_data_source = DataSource(config, tokenizer, dev_data, "dev")
    print_rank_0(str(dev_data_source.statistics))
    dev_dataloader = torch.utils.data.DataLoader(
        dev_data_source,
        batch_size=config.batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )

    # build model
    print_rank_0("Building model...")
    if config.model_type == "gpt":
        base_model = GPT2LMHeadModel.from_pretrained(config.pretrained_model_dir)
    elif config.model_type == "gpt-neox":
        base_model = GPTNeoXForCausalLM.from_pretrained(config.pretrained_model_dir)
    base_model = base_model.to(DEVICE)

    # freeze model parameters
    for param in base_model.parameters():
        param.requires_grad = False
    
    # build prefix-tuning model
    print_rank_0("Building prefix-tuning model...")
    class PrefixConfig:
        def __init__(self, prefix_seq_len, input_dim, hidden_dim, n_embd, n_layer, n_head):
            # Length of prefix
            self.prefix_seq_len = prefix_seq_len
            # prefix MLP size
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            # dropout of prefix MLP
            self.prefix_dropout_rate = 0.0
            # output size
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
    prefix_config = PrefixConfig(
        config.prefix_seq_len,
        config.prefix_input_dim,
        config.prefix_hidden_dim,
        base_model.config.n_embd,
        base_model.config.n_layer,
        base_model.config.n_head
    )
    prefix_encoder = PrefixEncoder(prefix_config)
    prefix_encoder = prefix_encoder.to(DEVICE)

    print_rank_0(f"Trainable backbone model parameters: {count_parameters(base_model)}")
    print_rank_0(f"Trainable prefix encoder parameters: {count_parameters(prefix_encoder)}")

    # load prefix encoder from checkpoint
    if config.checkpoint_path:
        print_rank_0("Loading prefix encoder from checkpoint...")
        print_rank_0("checkpoint path: {}".format(config.checkpoint_path))
        checkpoint = torch.load(config.checkpoint_path, map_location=DEVICE)
        prefix_encoder.load_state_dict(checkpoint["model"])
    # or init prefix-tuning model
    else:
        print_rank_0("Initializing prefix encoder parameters...")
        init_prefix(
            local_rank,
            config,
            base_model,
            prefix_config,
            prefix_encoder,
            DEVICE,
            n_epochs=200,
            learning_rate=0.0001,
            weight_decay=0.01
        )

    model = PrefixWrapper(base_model, prefix_encoder)

    # non-deepspeed
    if not config.deepspeed:
        # use mixed precision
        scaler = amp.GradScaler()

        # use multi gpus
        if config.world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank]
            )

    # build optimizer
    print_rank_0("Building optimizer...")
    param_optimizer = list(model.prefix_encoder.named_parameters())
    no_decay = ['bias', "ln", 'LayerNorm']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay},
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0
        }
    ]
    if config.deepspeed:
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=config.max_lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.max_lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay
        )

    # build lr scheduler
    print_rank_0("Building scheduler...")
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.n_warmup_steps,
        num_training_steps=config.n_training_steps,
    )

    # deepspeed
    if config.deepspeed:
        print_rank_0("Initializing deepspeed...")
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=config,
            lr_scheduler=lr_scheduler,
            dist_init_required=False
        )
        print_rank_0("Finished initializing deepspeed.")
    
    # init environment
    n_step = 0
    start_n_epoch = 0
    best_loss = float("inf")

    # names
    OUTPUT_FILEID = "{}_prefix_encoder.seed{}.{}".format(
        config.save_name,
        config.seed,
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    )
    if config.filename_note:
        OUTPUT_FILEID += f".{config.filename_note}"
    
    # define logger
    def log_rank_0(s):
        if is_rank_0():
            if config.save_log:
                os.makedirs(f"../log", exist_ok=True)
                with open(f"../log/{OUTPUT_FILEID}.log", "a+", encoding="utf-8") as log_f:
                    log_f.write(s+"\n")
            print_rank_0(s)

    if config.save_log:
        if is_rank_0():
            tb_writer = SummaryWriter(
                log_dir=f"../log/{OUTPUT_FILEID}",
                max_queue=5
            )

    # log hyper parameters
    start_time = time.time()
    log_rank_0("----- Hyper-parameters -----")
    for k, v in sorted(dict(config.__dict__).items()):
        log_rank_0("{}: {}".format(k, v))

    for epoch_idx in range(start_n_epoch, config.n_epochs):
        if isinstance(train_data_sampler, DistributedSampler):
            train_data_sampler.set_epoch(epoch_idx)

        train_data_iterator = iter(train_dataloader)

        while n_step < config.n_training_steps:
            n_step += 1

            for _ in range(config.n_accum_steps):
                try:
                    batch_data = next(train_data_iterator)
                except StopIteration:
                    batch_data = None
                    break

                # forward
                base_model.eval()
                prefix_encoder.train()
                if config.deepspeed:
                    loss, ppl = forward_step(base_model, prefix_encoder, tokenizer, batch_data)
                else:
                    with amp.autocast():
                        loss, ppl = forward_step(base_model, prefix_encoder, tokenizer, batch_data)

                # update statisitcs
                trn_reporter.update_data({
                    "loss": loss.item(),
                    "ppl": ppl.item()
                })

                # backward
                if config.deepspeed:
                    model.backward(loss)
                else:
                    scaler.scale(loss).backward()
                del loss

                # update model parameters (deepspeed)
                if config.deepspeed:
                    model.step()

            # update model parameters (non-deepspeed)
            if not config.deepspeed:
                if config.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(prefix_encoder.parameters(), config.grad_clip)

                # update model parameters
                scaler.step(optimizer)
                scaler.update()

                # zero gradients
                optimizer.zero_grad()

            if batch_data is None:
                break
                
            # check loss
            if n_step > 0 and n_step % config.log_every_n_steps == 0:
                lr = list(lr_scheduler.optimizer.param_groups)[0]["lr"]
                log_s = f"{time.time()-start_time:.2f}s Epoch {epoch_idx}, step {n_step}, lr {lr:.5g} - "
                log_s += trn_reporter.to_string()
                log_rank_0(log_s)

                if config.save_log and global_rank == 0:
                    for k, v in trn_reporter.items():
                        tb_writer.add_scalar(f"{k}/train", np.mean(v), n_step)

                trn_reporter.clear()
            
            # decay learning rate
            lr_scheduler.step()

        # evaluation on dev dataset
        if global_rank == 0:
            
            # forward
            with torch.no_grad():
                base_model.eval()
                prefix_encoder.eval()

                for eval_batch_idx, eval_batch_data in enumerate(dev_dataloader):
                    with amp.autocast():
                        loss, ppl = forward_step(base_model, prefix_encoder, tokenizer, eval_batch_data)

                    dev_reporter.update_data({"monitor": loss.item(), "ppl": ppl.item()})

                    if eval_batch_idx == len(dev_dataloader) - 1:
                        break

            log_s = f"\n<Dev> - Epoch {epoch_idx} - {time.time()-start_time:.3f}s - "
            log_s += dev_reporter.to_string()
            log_rank_0(log_s)

            # Save model if it has better monitor measurement
            if config.save_model:
                os.makedirs(f"../data/model", exist_ok=True)

                prefix_encoder.eval()
                model_to_save = prefix_encoder
                weights = model_to_save.prefix_mlp(model_to_save.prefix_wte.weight)

                checkpoint = weights

                # save current model
                torch.save(
                    checkpoint,
                    f"../data/model/{OUTPUT_FILEID}.checkpoint"
                )
                log_rank_0(f"checkpoint saved to data/model/{OUTPUT_FILEID}.checkpoint")

                # save best model
                cur_loss = dev_reporter.get_value("monitor")
                if cur_loss < best_loss:
                    best_loss = cur_loss

                    torch.save(
                        checkpoint,
                        f"../data/model/{OUTPUT_FILEID}.best.checkpoint"
                    )
                    log_rank_0(f"best checkpoint saved to data/model/{OUTPUT_FILEID}.best.checkpoint")

            if config.save_log:
                for k, v in dev_reporter.items():
                    tb_writer.add_scalar(f"{k}/dev", np.mean(v), n_step)

            dev_reporter.clear()
            torch.cuda.empty_cache()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # launcher
    parser.add_argument("--local_rank", type=int, help="necessary argument for distributed launcher")

    # modeling
    parser.add_argument("--model_type", type=str, default="gpt-neox", choices=["gpt-neox", "gpt"])
    parser.add_argument("--pretrained_model_dir", type=str, required=True, help="link or directory path to pretrained model")
    parser.add_argument("--prefix_seq_len", type=int, default=10, help="prefix length")
    parser.add_argument("--prefix_input_dim", type=int, default=10, help="input dimension of prefix encoder MLP")
    parser.add_argument("--prefix_hidden_dim", type=int, default=10, help="hidden dimension of prefix encoder MLP")
    
    # training
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for training")
    parser.add_argument("--n_training_steps", type=int, default=5e4, help="number of maximum training steps")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of maximum training epochs")
    parser.add_argument("--n_warmup_steps", type=int, default=0, help="number of warmup steps")
    parser.add_argument("--n_accum_steps", type=int, default=1, help="number of gradient accumulation steps")

    # optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam parameter")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam parameter")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="l2 penalty")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="min learning rate")
    parser.add_argument("--max_lr", type=float, default=1e-4, help="peak learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping threshold")

    # data
    parser.add_argument("--data_filepath", type=str, required=True, help="path to data")
    parser.add_argument("--train_data_size", type=int, default=1000, help="number of datapoints for training")
    parser.add_argument("--dev_data_size", type=int, default=10, help="number of datapoints for validation")

    # management
    parser.add_argument("--checkpoint_path", help="path to saved checkpoint file")
    parser.add_argument("--resume_training", action="store_true", default=False, help="resume training from checkpoint or not")
    parser.add_argument("--save_log", action="store_true", default=False, help="save training log or not")
    parser.add_argument("--save_model", action="store_true", default=False, help="save model to checkpoint or not")
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--log_every_n_steps", type=int, default=1e2, help="print loss after every this number of steps")
    parser.add_argument("--filename_note", type=str, help="suffix of saved files' names")

    # deepspeed
    parser.add_argument("--fp16", action="store_true", help="use fp16 for deepspeed")
    parser.add_argument("--bf16", action="store_true", help="use bf16 for deepspeed")
    parser = deepspeed.add_config_arguments(parser)

    config = parser.parse_args()
    config.rank = int(os.environ['RANK'])
    config.world_size = int(os.environ['WORLD_SIZE'])

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": config.batch_size,
        "gradient_accumulation_steps": config.n_accum_steps,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": config.max_lr,
                "betas": [
                    config.adam_beta1,
                    config.adam_beta2
                ],
                "eps": config.adam_eps,
                "weight_decay": config.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": config.min_lr,
                "warmup_max_lr": config.max_lr,
                "warmup_num_steps": config.n_warmup_steps*config.n_accum_steps,
                "total_num_steps": config.n_training_steps
            }
        },
        "fp16": {
            "enabled": config.fp16,
            "initial_scale_power": 10
        },
        "bf16": {
            "enabled": config.bf16
        },
        "gradient_clipping": config.grad_clip,
        "zero_optimization": {
            "stage": 0,
            "overlap_comm": True
        },
        "activation_checkpointing": {
            "partition_activations": True,
        },
        "steps_per_print": config.log_every_n_steps
    }
    if config.deepspeed and config.local_rank == 0:
        if os.path.exists(config.deepspeed_config):
            os.remove(config.deepspeed_config)
        with open(config.deepspeed_config, "w+") as f:
            json.dump(deepspeed_config, f, indent=2)

    import logging
    logging.basicConfig(level=logging.WARNING)

    training(config.local_rank, config)