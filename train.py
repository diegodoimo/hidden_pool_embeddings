import torch
import os
import numpy as np
import time
import json
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.distributed as dist
from argparse import ArgumentParser

from datasets import load_dataset, load_from_disk, Dataset
from transformers import GemmaTokenizerFast
from transformers import AutoConfig
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.arguments import parse_args
from utils.helpers import print_memory_consumed, save_model, get_cpt_steps
from utils.gemma3model import get_model
from utils.optimizer import get_scheduler_optimizer
from utils.contrastive_datasets import (
    collate_fn_with_hard_negatives,
    load_hard_negatives_datasets,
    QWEN3_600M_10DATASET_SUBSET,
)
from utils.dataloader_helpers import (
    collate_fn_with_padding,
    collate_fn_with_hard_negatives,
)
from utils.losses import EmbeddingGemmaLossDistributed, EmbeddingGemmaLossHardNegatives
from typing import Callable
from functools import partial

import mteb
from inference.test_retrieval_ddp_update import evaluate_retrieval
from inference.create_datasets import (
    instruction_template_qwen3,
    instruction_template_embeddinggemma,
)
from inference.helpers import last_token_pool, mean_pool

# MTEB 20-task subset (mteb_20task_subset_selection.md) - minimizes eval time while preserving category averages
TASK_DICT = {
    "mteb_eng_v2_20": [
        "SCIDOCS",
        "CQADupstackGamingRetrieval",
        "CQADupstackUnixRetrieval",
        "HotpotQAHardNegatives",
        "TRECCOVID",
        "TwentyNewsgroupsClustering.v2",
        "BiorxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2",
        "StackExchangeClustering.v2",
        "AskUbuntuDupQuestions",
        "BIOSSES",
        "STS17",
        "STS12",
        "AmazonCounterfactualClassification",
        "MassiveScenarioClassification",
        "TweetSentimentExtractionClassification",
        "MTOPDomainClassification",
        "TwitterSemEval2015",
        "SprintDuplicateQuestions",
        "SummEvalSummarization.v2",
    ],
}


def get_eval_tasks(eval_set):
    """Return list of MTEB task objects for evaluation."""
    if eval_set == "mteb_multilingual_v2":
        benchmark = mteb.get_benchmark("MTEB(Multilingual, v2)")
        tasks = [task for task in benchmark.tasks]
    elif eval_set == "mteb_eng_v2":
        benchmark = mteb.get_benchmark("MTEB(eng, v2)")
        tasks = [task for task in benchmark.tasks]
    elif eval_set == "mteb_eng_v2_20":
        task_names = TASK_DICT["mteb_eng_v2_20"]
        tasks = [mteb.get_task(name) for name in task_names]
    else:
        raise ValueError(f"Unknown eval_set: {eval_set}")
    return tasks


class Trainer:
    def __init__(self, args, model_config, len_dataloader):

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
        self.model = torch.compile(self.model)
        # self.model.compile(mode="reduce-overhead")

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
        # print(self.model)

    def train(
        self,
        args: ArgumentParser,
        train_loader: DataLoader,
        loss_fn: Callable,
        evaluator,
    ):

        filename = ""
        if args.out_filename != "":
            filename = "_" + args.out_filename

        eval_steps, _ = get_cpt_steps(
            int(args.eval_steps), args.max_train_steps, logspace=False
        )
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

                query_inputs = batch["query_token_ids"]
                query_mask = batch["query_attention_mask"]
                doc_inputs = batch["pos_token_ids"]
                doc_mask = batch["pos_attention_mask"]
                doc_ids = batch["pos_ids"]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                    query_embeddings = self.model(
                        input_ids=query_inputs, attention_mask=query_mask
                    )
                    doc_embeddings = self.model(
                        input_ids=doc_inputs, attention_mask=doc_mask
                    )

                    if "neg_token_ids" in batch and isinstance(
                        loss_fn, EmbeddingGemmaLossHardNegatives
                    ):
                        neg_inputs = batch["neg_token_ids"]
                        neg_mask = batch["neg_attention_mask"]
                        B, num_neg, seq_len_neg = neg_inputs.shape

                        neg_embeddings = self.model(
                            input_ids=neg_inputs.view(B * num_neg, seq_len_neg),
                            attention_mask=neg_mask.view(B * num_neg, seq_len_neg),
                        ).view(B, num_neg, -1)

                        loss = loss_fn(
                            query_embeddings=query_embeddings,
                            doc_embeddings=doc_embeddings,
                            hard_neg_embeddings=neg_embeddings,
                            doc_ids=doc_ids,
                        )
                    else:
                        loss = loss_fn(
                            query_embeddings=query_embeddings,
                            doc_embeddings=doc_embeddings,
                            doc_ids=doc_ids,
                        )

                loss.backward()
                total_loss += loss.detach().float()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), args.clip_grad_thresh
                )
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

                        with open(
                            f"{args.output_dir}/train_logs{filename}.json", "w"
                        ) as f:
                            json.dump(stats, f, indent=4)

                if completed_steps in eval_steps:
                    results, summary = evaluator.evaluate(self.model, batch_size=64)

                    if RANK == 0:
                        print(f"iter {completed_steps}.")
                        stats["test_perf"][completed_steps] = summary
                        with open(
                            f"{args.output_dir}/train_logs{filename}.json", "w"
                        ) as f:
                            json.dump(stats, f, indent=4)

                if completed_steps in checkpointing_steps and args.save_checkpoint:
                    if RANK == 0:
                        print("saving checkpoint")

                    output_dir = f"{len(checkpointing_steps)}ckpts{filename}/step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    save_model(
                        self.model, output_dir, RANK=RANK, dist_type=args.dist_type
                    )

            if RANK == 0:
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
        print("loading train set ")
        start = time.time()

    instruction_template = instruction_template_qwen3
    if args.instruction_template == "embeddinggemma":
        instruction_template = instruction_template_embeddinggemma

    train_dataset = load_hard_negatives_datasets(
        base_dir=args.negatives_dir,
        num_hard_negatives=args.num_hard_negatives,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        max_query_len=args.max_query_len,
        max_passage_len=args.max_passage_len,
        rank=RANK,
        datasets_subset=args.datasets_subset,
    )

    dist.barrier()
    if RANK == 0:
        print(f"datasets tokenized in {time.time()-start:.1f}s")
        start = time.time()
        print("dataloader preparation")

    sampler = LengthBalancedDistributedSampler(
        train_dataset,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=False,
        seed=42,
    )

    collate_fn = partial(
        collate_fn_with_hard_negatives,
        pad_token_id=tokenizer.pad_token_id,
        num_hard_negatives=args.num_hard_negatives,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # **************************************

    eval_tasks = get_eval_tasks(args.eval_set)

    if args.instruction_template == "embeddinggemma":
        pool_fn = mean_pool
        add_special_tokens = True
        eot_id = None
    else:
        pool_fn = last_token_pool
        add_special_tokens = False
        eot_id = tokenizer.pad_token_id

    evaluator = evaluate_retrieval(
        tasks=eval_tasks,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        padding_side="right",
        new_inference_mode=True,
        pool_fn=pool_fn,
        add_special_tokens=add_special_tokens,
        eot_id=eot_id,
    )

    # Initialize loss and optimizer
    loss_fn = EmbeddingGemmaLossHardNegatives(
        temperature=0.07, num_hard_negatives=args.num_hard_negatives
    )
    if WORLD_SIZE > 1 and args.distributed_loss:
        loss_fn = EmbeddingGemmaLossDistributed(temperature=0.07)

    if RANK == 0:
        print("model setup")

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    # dist.barrier()
    trainer = Trainer(
        len_dataloader=len(train_loader),
        model_config=model_config,
        args=args,
    )

    if args.eval_only:
        results, summary = evaluator.evaluate(trainer.model, batch_size=32)
        print(results)
        print(summary)
    else:
        dist.barrier()
        trainer.train(
            args=args,
            train_loader=train_loader,
            loss_fn=loss_fn,
            evaluator=evaluator,
        )
        dist.destroy_process_group()


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])
    main()
