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
    create_hard_negatives_datasets,
    QWEN3_600M_DATASET_SUBSET,
    get_eval_tasks,
)
from utils.dataloader_helpers import (
    collate_fn_with_hard_negatives,
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


args = parse_args()

# Login to Hugging Face for gated models (read token from .hf_token, gitignored)
_hf_token_path = os.path.join(os.path.dirname(__file__), ".hf_token")
if os.path.isfile(_hf_token_path):
    with open(_hf_token_path, "r") as f:
        token = f.read().strip()
    if token:
        hf_login(token=token)


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

# timing_stats is updated in-process only; num_workers MUST stay 0 here.
# (Worker processes run in separate address spaces and cannot share the dict.)
timing_stats: dict[str, float] = defaultdict(float)

collate_fn = partial(
    collate_fn_with_hard_negatives,
    pad_token_id=tokenizer.pad_token_id,
    num_hard_negatives=args.num_hard_negatives,
    padding_side="right",
    tokenizer=tokenizer,
    eot_id=eot_id,
    add_special_tokens=add_special_tokens,
    max_seq_len=args.max_seq_len if args.length_strategy == "truncate" else None,
    timing_stats=timing_stats,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.per_device_train_batch_size,
    sampler=sampler,
    collate_fn=collate_fn,
    # num_workers MUST be 0 for collate_fn step-timing to work: workers run in
    # separate processes and cannot share the timing_stats dict.
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=None,
    multiprocessing_context=None,
)


start = time.time()
for index, batch in enumerate(train_loader):
    batch = {
        key: val.to(device) if isinstance(val, torch.Tensor) else val
        for key, val in batch.items()
    }
    if index > 100:
        break


duration = time.time() - start
print(f"Total dataloader loop duration: {duration:.3f}s")

# Print per-step timing report
n_calls = int(timing_stats.get("_calls", 0))
if n_calls > 0:
    STEP_KEYS = [
        "query_tokenize",
        "query_to_tensor",
        "pos_tokenize",
        "pos_to_tensor",
        "neg_tokenize",
        "neg_to_tensor",
        "id_build",
        "query_pad",
        "doc_pad",
        "total",
    ]
    total_acc = timing_stats.get("total", 1e-9)
    print(f"\ncollate_fn step timings ({n_calls} calls):")
    print(f"  {'step':<20}  {'total_s':>10}  {'avg_ms':>10}  {'pct':>7}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*7}")
    for key in STEP_KEYS:
        val = timing_stats.get(key, 0.0)
        avg_ms = val / n_calls * 1000
        pct = val / total_acc * 100 if key != "total" else 100.0
        print(f"  {key:<20}  {val:>10.3f}  {avg_ms:>10.2f}  {pct:>6.1f}%")
else:
    print("No collate_fn timing data collected (was num_workers > 0?).")
