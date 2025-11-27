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
    LengthBalancedDistributedSampler,
)
from utils.losses import EmbeddingGemmaLossDistributed, EmbeddingGemmaLoss
import mteb
from typing import Callable


class Trainer:
    def __init__(
        self,
        args,
        model_config,
        len_dataloader,
    ):

        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size()
        self.rng = np.random.default_rng(args.seed)
        self.device = torch.device(self.local_rank)

        assert self.rank == RANK
        assert self.world_size == WORLD_SIZE

        if self.rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)

        self.model, task_type, lora_modules = get_model(args, model_config)

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

        # Prepare everything with `accelerator` model must be prepared before giving it to the optimizer.
        print_memory_consumed(message="memory consumed before loading model")

        # 3. Move your model to the device
        self.model = self.model.to(self.device)

        if args.activation_checkpointing:
            assert self.model.config.use_cache == False
            self.model.gradient_checkpointing_enable()

        self.model = DDP(self.model, device_ids=[self.local_rank])
        # self.model = torch.compile(self.model)
        self.model.compile(mode="reduce-overhead")
        print_memory_consumed(message="memory consumed after loading model")

        self.optimizer, self.scheduler = get_scheduler_optimizer(
            self.model,
            args,
            len_dataloader,
        )

    # def train(
    #     self,
    #     args: ArgumentParser,
    #     train_loader: DataLoader,
    #     loss_fn: Callable,
    # ):
    def train(
        self,
        args,
        tokenizer,
    ):
        text = "The quick brown fox jumps over the lazy dog"
        repeat_count = 2000  # This gives about 2400+ characters
        text = (text * repeat_count).strip()

        text_batches = [text for i in range(args.per_device_train_batch_size)]

        # Tokenize as a batch of sequences
        inputs = tokenizer(
            text_batches,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        )
        # Add labels for autoregressive training (shifted prediction)
        # inputs["labels"] = inputs["input_ids"].clone()

        # Move inputs to GPU
        # batch = {k: v.to(self.model.device) for k, v in inputs.items()}

        print(
            "input_ids shape", inputs["input_ids"].shape
        )  # Should be (batch_size, sequence_length)
        toks_batch = torch.numel(inputs["input_ids"])
        print("toks batch", toks_batch)

        # warmup
        for _ in range(args.gradient_accumulation_steps * 5):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch = {k: v.to(self.model.device) for k, v in inputs.items()}

                print(batch)
                outputs = self.model(**batch)
                loss = (emb**2).mean()

            loss.backward()
            total_loss += loss.detach().float()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad_thresh)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        local_total_tokens = 0
        start = time.time()
        for i in range(200):

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**batch)
                loss = (emb**2).mean()

            loss.backward()
            total_loss += loss.detach().float()
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
            print(f"processed {throughput: .2f} token/sec/gpu")

        assert False, "benchmarking finished"

        filename = ""
        if args.out_filename != "":
            filename = "_" + args.out_filename

        eval_steps, _ = get_cpt_steps(int(args.eval_steps), args.max_train_steps, logspace=False)
        checkpointing_steps, _ = get_cpt_steps(
            args.checkpointing_steps, args.max_train_steps, logspace=False
        )
        log_steps, log_interval = get_cpt_steps(
            int(args.logging_steps), args.max_train_steps, logspace=False
        )

        stats = defaultdict(dict)
        stats["train_params"] = {
            "num_epochs": args.num_train_epochs,
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }

        if RANK == 0:
            print("log_steps:", log_steps)
            print("eval_steps", eval_steps)
            print("***** Running training *****")
            print(f"  Num Epochs = {args.num_train_epochs}")
            print(f"  Learning rate = {args.learning_rate}")
            print(f"  Weight Decay = {args.weight_decay}")
            print(f"  Lora Rank = {args.lora_rank}")
            print(f"  Lora Alpha = {args.lora_alpha}")
            print(f"  Lora Dropout = {args.lora_dropout}")
            print(f"  Batch size per device = {args.per_device_train_batch_size}")
            print(
                f"  Total batch size (w. parallel, distributed & accumulation) = {args.batch_size}"
            )
            print(f"  world size = {WORLD_SIZE}")
            print(f"  len_dataloader = {len(train_loader)}")
            print(f"  Total optimization steps = {args.max_train_steps}")
            print(f"  Log steps number = {len(log_steps)}")

            print("memory before train run")
            print_memory_consumed(rank=RANK)
            print("\nstart training...")

        completed_steps = 0
        total_loss = 0
        total_time = 0

        start = time.time()
        for epoch in range(args.num_train_epochs):

            self.model.train()
            # gradient accumulation step may not finish with a proper update at the end of the epoch so we call zero grad here
            self.optimizer.zero_grad()

            # if WORLD_SIZE > 1:
            #     sampler.set_epoch(epoch)

            for index, batch in enumerate(train_loader):

                batch = {key: val.to(self.model.device) for key, val in batch.items()}

                query_inputs = batch["query_token_ids"].to(self.model.device)
                query_mask = batch["query_attention_mask"].to(self.model.device)
                doc_inputs = batch["pos_token_ids"].to(self.model.device)
                doc_mask = batch["pos_attention_mask"].to(self.model.device)

                doc_ids = batch["pos_ids"].to(self.model.device)

                # same as before but the gradients will be sync
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                    queries_embeddings = self.model(query_inputs, query_mask)
                    document_embeddings = self.model(doc_inputs, doc_mask)

                    # gradients are averaged across gpus by DDP, even if loss is not. / WORLD SIZE is not necessary.
                    loss = loss_fn(
                        queries_embeddings=queries_embeddings,
                        document_embeddings=document_embeddings,
                        doc_ids=doc_ids,
                    )

                loss.backward()
                total_loss += loss.detach().float()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad_thresh)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                completed_steps += 1

                if completed_steps in log_steps:

                    if WORLD_SIZE > 1:
                        total_loss = total_loss.reshape(1)
                        dist.all_reduce(total_loss)

                    avg_loss = total_loss.item() / WORLD_SIZE / log_interval
                    total_loss = 0

                    if RANK == 0:
                        stats["loss"][completed_steps] = avg_loss
                        print(f"log step: {completed_steps}/{log_steps[-1]}")
                        print_memory_consumed(rank=RANK)

                        total_time = time.time() - start

                        print(
                            f"LR: {self.lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, \
                                Time: {int(total_time//3600)} h {(total_time%3600)/60: .2f} min"
                        )

                        with open(f"{args.output_dir}/train_logs{filename}.json", "w") as f:
                            json.dump(stats, f, indent=4)

                if completed_steps in eval_steps:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        evaluate(self.model, args)

                    if RANK == 0:
                        print(f"iter {completed_steps}.")

                        with open(f"{args.output_dir}/train_logs{filename}.json", "w") as f:
                            json.dump(stats, f, indent=4)

                if completed_steps in checkpointing_steps and args.save_checkpoint:
                    if RANK == 0:
                        print("saving checkpoint")

                    output_dir = f"{len(checkpointing_steps)}ckpts{filename}/step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    save_model(self.model, output_dir, RANK=RANK, dist_type=args.dist_type)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                stats = evaluate()

            if RANK == 0:
                print(f"iter {completed_steps}. vqa accuracy {stats['vqa_acc']}")
                print(f"iter {completed_steps}. coco cider {stats['coco_cider']}")
                with open(f"{args.output_dir}/train_logs{filename}.json", "w") as f:
                    json.dump(stats, f, indent=4)

            output_dir = f"epoch_{epoch+1}{filename}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_model(self.model, output_dir, RANK=RANK, dist_type=args.dist_type)


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

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    dist.barrier()
    trainer = Trainer(
        len_dataloader=1000,
        model_config=model_config,
        args=args,
    )

    dist.barrier()
    trainer.train(args, tokenizer)
    #     args=args,
    #     train_dataloader=train_loader,
    #     tokenizer=tokenizer,
    #     loss_fn=loss_fn,
    # )

    assert False

    if RANK == 0:
        print("loading msmarco")
        start = time.time()
    hf_corpus = load_dataset("mteb/msmarco", name="corpus")
    hf_queries = load_dataset("mteb/msmarco", name="queries")
    hf_qrels = load_dataset("mteb/msmarco", name="default", split="train")

    if RANK == 0:
        print(f"msmarco loaded in {time.time()-start}")
        print("matching query and positives")
        start = time.time()
    train_queries, train_docs = prepare_msmarco(hf_queries, hf_corpus, hf_qrels)
    dist.barrier()

    if RANK == 0:
        print(f"msmarco prepared in {time.time()-start}")
        start = time.time()
        print("tokenizing dataset")

    tokenized_dataset = msmarco_dataset(
        queries_dataset=train_queries,
        pos_passages_dataset=train_docs,
        tokenizer=tokenizer,
        max_query_len=1024,
        max_passage_len=4096,
        sort_by_length=True,
        query_task="query",
        document_task="document",
        batch_size=1000,
        rank=RANK,
    )

    dist.barrier()
    if RANK == 0:
        print(f"msmarco tokenized in {time.time()-start}")
        start = time.time()
        print("dataloader preparation")
    # 3. Create length-balanced sampler
    sampler = LengthBalancedDistributedSampler(
        tokenized_dataset,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=False,
        seed=42,
    )

    # 4. Create DataLoader with correct pad_token_id
    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=args.per_device_train_batch_size,  # Per-GPU batch size
        sampler=sampler,
        collate_fn=lambda batch: collate_fn_with_padding(
            batch, pad_token_id=tokenizer.pad_token_id
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize loss and optimizer
    loss_fn = EmbeddingGemmaLoss(temperature=0.07)
    if WORLD_SIZE > 1 and args.distributed_loss:
        loss_fn = EmbeddingGemmaLossDistributed(temperature=0.07)

    if RANK == 0:
        print("model setup")

    # model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    # dist.barrier()
    # trainer = Trainer(
    #     len_dataloader=len(train_loader),
    #     model_config=model_config,
    #     args=args,
    # )

    dist.barrier()
    trainer.train(
        args=args,
        train_dataloader=train_loader,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
    )
    dist.destroy_process_group()


def evaluate(model, args):

    # benchmark = mteb.get_benchmark(bench_dict[args.benchmark])

    # res = mteb.evaluate(model, tasks=task, overwrite_strategy="always")

    # instance = res.task_results[0]
    # splits = set(instance.scores.keys())

    # for split in splits:
    #     results[instance.task_name] = {
    #         "main_score": instance.scores[split][0]["main_score"],
    #         "task_type": instance.task_type,
    #         "split": split,
    #     }

    # print({np.mean([score["main_score"] for score in results.values()])})
    return


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])

    main()
