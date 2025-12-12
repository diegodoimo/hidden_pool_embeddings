import torch
import os
import numpy as np
import time
import json
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.distributed as dist
from argparse import ArgumentParser

from datasets import load_dataset
from transformers import GemmaTokenizerFast
from transformers import AutoConfig
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP


from utils.arguments import parse_args
from utils.helpers import print_memory_consumed, save_model, get_cpt_steps
from utils.model import get_model
from utils.optimizer import get_scheduler_optimizer
from utils.contrastive_datasets import (
    msmarco_dataset,
    prepare_msmarco,
    collate_fn_with_padding,
    collate_fn_with_padding_joint,
    LengthBalancedDistributedSampler,
)
from utils.losses import EmbeddingGemmaLossDistributed, EmbeddingGemmaLoss
import mteb
from typing import Callable
from utils.model import ContrastiveLossEmbedding


class Trainer:
    def __init__(self, args, model_config, len_dataloader, loss_fn):

        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size()
        self.rng = np.random.default_rng(args.seed)
        self.device = torch.device(self.local_rank)

        assert self.rank == RANK
        assert self.world_size == WORLD_SIZE

        if self.rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)

        self.model, task_type, lora_modules = get_model(args, model_config, loss_fn)

        if args.activation_checkpointing:
            # Disable cache first
            self.model.encoder.config.use_cache = False

            # Enable PyTorch gradient checkpointing
            self.model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if args.use_lora:
            peft_config = LoraConfig(
                task_type=task_type,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=lora_modules,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        print_memory_consumed(message="memory consumed before loading model")

        # self.model = ContrastiveLossEmbedding(model = self.model, loss_fn=loss_fn)

        # 3. Move your model to the device
        self.model = self.model.to(self.device)

        self.model = DDP(self.model, device_ids=[self.local_rank])
        # self.model = torch.compile(self.model)
        self.model.compile(mode="reduce-overhead")

        # self.model.compile(
        #     mode="default",
        #     dynamic=True,
        #     fullgraph=False  # Allow graph breaks
        # )
        print_memory_consumed(message="memory consumed after loading model")

        self.optimizer, self.lr_scheduler = get_scheduler_optimizer(
            self.model,
            args,
            len_dataloader,
        )

    def train(
        self,
        args,
        tokenizer,
    ):

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad and RANK ==0:
        #         print("requires_grad:", name)
        text = "The quick brown fox jumps over the lazy dog"
        repeat_count = 2000  # This gives about 2400+ characters
        text = (text * repeat_count).strip()

        loss_fn = EmbeddingGemmaLossDistributed(temperature=0.07)
        text_batches_d = [text for i in range(args.per_device_train_batch_size)]

        text_batches_q = [text for i in range(2 * args.per_device_train_batch_size)]
        doc_ids = torch.tensor(
            [
                int(i + args.per_device_train_batch_size * RANK)
                for i in range(args.per_device_train_batch_size)
            ],
            dtype=torch.long,
            device=self.model.device,
        )

        args.max_seq_len = 1024
        # Tokenize as a batch of sequences

        inputs_q = tokenizer(
            text_batches_q,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs_d = tokenizer(
            text_batches_d,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        )

        print("input_ids shape", inputs_q["input_ids"].shape)
        toks_batch = torch.numel(inputs_q["input_ids"])
        toks_batch_q = torch.numel(inputs_q["input_ids"])
        print("toks batch", toks_batch)

        self.model.train()
        # warmup
        for _ in range(args.gradient_accumulation_steps * 2):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_q = {k: v.to(self.model.device) for k, v in inputs_q.items()}
                # batch_d = {k: v.to(self.model.device) for k, v in inputs.items()}
                # print(batch_q, batch_d, doc_ids)

                # loss = self.model(batch_q, batch_d, doc_ids)

                # batch = {k: v.to(self.model.device) for k, v in inputs_q.items()}
                out = self.model(**batch_q)
                out_q = out[: args.args.per_device_train_batch_size]
                out_d = out[args.args.per_device_train_batch_size :]
                # batch = {k: v.to(self.model.device) for k, v in inputs.items()}
                # out_d = self.model(**batch_d)

                # loss = (outputs_q**2).mean()
                loss = self.loss_fn(
                    query_embeddings=out_q,
                    doc_embeddings=out_d,
                    doc_ids=doc_ids,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad_thresh)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        local_total_tokens = 0
        dist.barrier()
        start = time.time()
        for i in range(10):

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                # batch_q = {k: v.to(self.model.device) for k, v in inputs_q.items()}
                batch_d = {k: v.to(self.model.device) for k, v in inputs.items()}
                local_total_tokens += toks_batch_q  # + toks_batch

                # loss = self.model(batch_q, batch_d, doc_ids)

                # batch = {k: v.to(self.model.device) for k, v in inputs_q.items()}
                out_q = self.model(**batch_q)

                # batch = {k: v.to(self.model.device) for k, v in inputs.items()}
                # out_d = self.model(**batch_d)

                # loss = (outputs_q**2).mean()
                loss = self.loss_fn(
                    query_embeddings=out_q,
                    doc_embeddings=out_d,
                    doc_ids=doc_ids,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad_thresh)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        total_time = time.time() - start

        if WORLD_SIZE > 1:
            num_tokens = torch.tensor([local_total_tokens]).to("cuda")
            dist.all_reduce(num_tokens)
            num_tokens = num_tokens.item()
        else:
            num_tokens = local_total_tokens

        if RANK == 0:
            throughput = num_tokens / total_time / WORLD_SIZE
            print(f"processed {throughput/10**6: .2f} million token/sec/gpu")

        assert False, "benchmarking finished"


def main():
    args = parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

    args.batch_size = WORLD_SIZE * args.per_device_train_batch_size
    args.gradient_accumulation_steps = 1

    # load embeddinggemma tokenizer. The following should be alredy implemented as defaults
    tokenizer = GemmaTokenizerFast.from_pretrained(
        args.model_name_or_path,
        add_bos_token=True,
        add_eos_token=True,
        padding_side="left",
    )

    if RANK == 0:
        print("model setup")

    loss_fn = EmbeddingGemmaLoss(temperature=0.07)
    if WORLD_SIZE > 1 and args.distributed_loss:
        loss_fn = EmbeddingGemmaLossDistributed(temperature=0.07)

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    dist.barrier()
    trainer = Trainer(len_dataloader=1000, model_config=model_config, args=args, loss_fn=loss_fn)

    dist.barrier()
    trainer.train(args, tokenizer)


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])

    main()
