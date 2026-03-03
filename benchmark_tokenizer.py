import torch
import os

# Tokenizer parallelism: safe to enable because the DataLoader uses
# multiprocessing_context="spawn" (no fork → no thread-pool deadlock).
# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # OLD: disabled for fork safety
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import time
import json
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.arguments import parse_args
from utils.helpers import print_memory_consumed, save_model, get_cpt_steps
from models.t5gemma2model import get_model_t5gemma2_model
from utils.optimizer import get_scheduler_optimizer
from utils.create_datasets import (
    create_pretokenized_hard_negatives_datasets,
    create_hard_negatives_datasets,
    QWEN3_600M_DATASET_SUBSET,
    get_eval_tasks,
)
from utils.dataloader_helpers import (
    collate_fn_with_hard_negatives,
    collate_fn_with_hard_negatives_v2,
    collate_fn_pretokenized,
    DatasetAwareSampler,
)
from utils.losses import EmbeddingGemmaLossDistributed, EmbeddingGemmaLossHardNegatives
from typing import Callable
from functools import partial

from huggingface_hub import login as hf_login

from inference.test_retrieval_ddp_update import evaluate_retrieval
from utils.create_datasets import (
    instruction_template_qwen3,
    instruction_template_embeddinggemma,
)
from models.modules import last_token_pool, add_pooling_layers, mean_pool
from datetime import timedelta


def main():

    args = parse_args()
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])

    dist.init_process_group(
        "nccl",
        device_id=torch.device("cuda", LOCAL_RANK),
        timeout=timedelta(minutes=30),
    )
    rank = dist.get_rank()
    torch.cuda.set_device(LOCAL_RANK)

    # Login to Hugging Face for gated models (read token from .hf_token, gitignored)
    _hf_token_path = os.path.join(os.path.dirname(__file__), ".hf_token")
    if os.path.isfile(_hf_token_path):
        with open(_hf_token_path, "r") as f:
            token = f.read().strip()
        if token:
            hf_login(token=token)

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", LOCAL_RANK)


    args.batch_size = WORLD_SIZE * args.per_device_train_batch_size
    args.gradient_accumulation_steps = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    instruction_template = instruction_template_embeddinggemma
    add_special_tokens = True
    eot_id = None

    train_list = None  # defaults to all
    if args.train_subset == "reduced":
        train_list = QWEN3_600M_DATASET_SUBSET

    # train_dataset = create_pretokenized_hard_negatives_datasets(
    #     base_dir=args.negatives_dir,
    #     num_hard_negatives=args.num_hard_negatives,
    #     tokenizer=tokenizer,
    #     instruction_template=instruction_template,
    #     add_special_tokens=add_special_tokens,
    #     rank=RANK,
    #     datasets_subset=train_list,
    #     max_seq_len=args.max_seq_len if args.length_strategy == "filter" else None,
    # )

    train_dataset = create_hard_negatives_datasets(
        base_dir=args.negatives_dir,
        num_hard_negatives=args.num_hard_negatives,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        rank=RANK,
        datasets_subset=train_list,
        max_seq_len=args.max_seq_len if args.length_strategy == "filter" else None,
    )

    # dataset collection is already sorted by length dataset / specific
    sampler = DatasetAwareSampler(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        strategy="grouped",
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=True,
        seed=42,
    )

    # ------------------------------------------------------------------ #
    #  Collate variants to benchmark                                      #
    # ------------------------------------------------------------------ #
    STEP_KEYS = [
        "prompt_extract",
        "tokenize_parallel",
        "id_build",
        "query_pad",
        "doc_pad",
        "total",
    ]
    NUM_BENCH_BATCHES = 500

    collate_variants = {
        "v1_thread_pool": collate_fn_with_hard_negatives,
        "v2_rust_encode_batch": collate_fn_with_hard_negatives_v2,
    }

    all_results: dict[str, dict] = {}  # variant_name -> {duration, timing_stats}

    for variant_name, collate_func in collate_variants.items():
        timing_stats: dict[str, float] = defaultdict(float)

        collate_fn = partial(
            collate_func,
            pad_token_id=tokenizer.pad_token_id,
            num_hard_negatives=args.num_hard_negatives,
            padding_side="right",
            tokenizer=tokenizer,
            eot_id=eot_id,
            add_special_tokens=add_special_tokens,
            max_seq_len=(
                args.max_seq_len if args.length_strategy == "truncate" else None
            ),
            timing_stats=timing_stats,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            # num_workers MUST be 0 for collate_fn step-timing to work.
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None,
            multiprocessing_context=None,
        )

        # Reset sampler so both variants iterate the same batches
        sampler.set_epoch(0)

        start = time.time()
        for index, batch in enumerate(train_loader):
            batch = {
                key: val.to(device) if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
            if index >= NUM_BENCH_BATCHES:
                break

        duration = time.time() - start
        all_results[variant_name] = {
            "duration": duration,
            "timing_stats": dict(timing_stats),
        }

        # Per-variant report
        n_calls = int(timing_stats.get("_calls", 0))
        total_acc = timing_stats.get("total", 1e-9)
        print(f"\n{'='*60}")
        print(f"  {variant_name}  ({n_calls} calls, {duration:.3f}s wall)")
        print(f"{'='*60}")
        if n_calls > 0:
            print(f"  {'step':<20}  {'total_s':>10}  {'avg_ms':>10}  {'pct':>7}")
            print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*7}")
            for key in STEP_KEYS:
                val = timing_stats.get(key, 0.0)
                avg_ms = val / n_calls * 1000
                pct = val / total_acc * 100 if key != "total" else 100.0
                print(f"  {key:<20}  {val:>10.3f}  {avg_ms:>10.2f}  {pct:>6.1f}%")

    # ------------------------------------------------------------------ #
    #  Side-by-side comparison                                            #
    # ------------------------------------------------------------------ #
    names = list(all_results.keys())
    if len(names) == 2 and RANK == 0:
        a, b = names
        sa = all_results[a]["timing_stats"]
        sb = all_results[b]["timing_stats"]
        na = int(sa.get("_calls", 1))
        nb = int(sb.get("_calls", 1))
        print(f"\n{'='*72}")
        print(f"  Comparison: {a} vs {b}")
        print(f"{'='*72}")
        header = f"  {'step':<20}  {'v1 avg_ms':>10}  {'v2 avg_ms':>10}  {'speedup':>8}"
        print(header)
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")
        for key in STEP_KEYS:
            va = sa.get(key, 0.0) / na * 1000
            vb = sb.get(key, 0.0) / nb * 1000
            speedup = va / vb if vb > 0 else float("inf")
            print(f"  {key:<20}  {va:>10.2f}  {vb:>10.2f}  {speedup:>7.2f}x")
        da = all_results[a]["duration"]
        db = all_results[b]["duration"]
        print(f"\n  Wall time: {a}={da:.3f}s  {b}={db:.3f}s  speedup={da/db:.2f}x")


if __name__ == "__main__":
    main()
