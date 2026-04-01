"""Shared helpers for all evaluation modules.

These functions were extracted from the ``evaluate_retrieval`` class so that
every ``eval_*.py`` module can import them without depending on a class
instance.
"""

from dataclasses import dataclass
from typing import Any, Optional

import os
import torch
import torch.distributed as dist
from functools import partial
from torch.utils.data import DataLoader
from mteb.types import PromptType
from datasets import Dataset as HFDataset

from inference.helpers import encode
from utils.create_datasets import create_dataset
from utils.dataloader_helpers import collate_fn_with_padding, LenghtSortedSampler


# ---------------------------------------------------------------------------
# Eval context
# ---------------------------------------------------------------------------


@dataclass
class EvalContext:
    """Context passed to evaluate_one_* functions instead of self.

    Carries tokenizer, encoding params, and distributed info needed by
    encode_dataset and collate_fn construction.
    """

    tokenizer: Any
    padding_side: str = "right"
    eot_id: Optional[int] = None
    add_special_tokens: bool = False
    world_size: int = 1
    rank: int = 0
    new_inference_mode: bool = True


# ---------------------------------------------------------------------------
# Collate / encoding helpers
# ---------------------------------------------------------------------------


def make_collate_fn(
    tokenizer,
    padding_side,
    eot_id,
    add_special_tokens,
    truncation_max_length=None,
):
    """Build a reusable collate function from tokenizer params."""
    return partial(
        collate_fn_with_padding,
        pad_token_id=tokenizer.pad_token_id,
        padding_side=padding_side,
        tokenizer=tokenizer,
        eot_id=eot_id,
        add_special_tokens=add_special_tokens,
        truncation_max_length=truncation_max_length,
    )


def encode_dataset(
    model,
    dataset,
    batch_size,
    collate_fn,
    prompt_type=PromptType.query,
):
    """Encode a prepared dataset using the DDP-aware pipeline.

    *prompt_type* is passed through to ``encode()`` (e.g. document chunking);
    prompts are already baked into *dataset* via ``create_dataset``.
    """
    sampler = LenghtSortedSampler(dataset)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=max(1, len(os.sched_getaffinity(0)) // 2 - 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dist.barrier()
    world_size = dist.get_world_size()
    embeddings = encode(
        model,
        loader,
        prompt_type=prompt_type,
        world_size=world_size,
    )
    dist.barrier()
    return embeddings


# ---------------------------------------------------------------------------
# Dataset preparation helpers
# ---------------------------------------------------------------------------


def prepare_text_dataset(
    texts,
    task_metadata,
    instruction_template,
    tokenizer,
    rank,
    max_length=8192,
    prompt_type=PromptType.query,
    skip_length_filter: bool = False,
):
    """Create a prompt-augmented HF dataset from raw texts for encoding.

    *prompt_type* should match MTEB per-column behavior (e.g. STS column1 vs
    column2, or document for plain passage encoding).

    Returns
    -------
    (ds, removed_indices) : tuple
        The filtered dataset and the set of original integer positions that
        were removed (too long or empty).
    """
    dataset = HFDataset.from_dict(
        {"id": [str(i) for i in range(len(texts))], "text": texts}
    )
    ds = create_dataset(
        dataset=dataset,
        task_metadata=task_metadata,
        instruction_template=instruction_template,
        tokenizer=tokenizer,
        prompt_type=prompt_type,
        max_length=max_length,
        skip_length_filter=skip_length_filter,
    )
    removed_indices = (
        set(int(x) for x in ds.removed_ids) if ds.removed_ids else set()
    )
    if removed_indices and rank == 0:
        if skip_length_filter:
            print(
                f"  WARNING: {len(removed_indices)}/{len(texts)} empty texts removed"
            )
        else:
            print(
                f"  WARNING: {len(removed_indices)}/{len(texts)} texts filtered "
                f"(>{max_length} tokens or empty)"
            )
    return ds, removed_indices


def build_index_remap(n_original, removed_set):
    """Build old-index → new-index mapping after removing items."""
    old_to_new = {}
    new_idx = 0
    for old_idx in range(n_original):
        if old_idx not in removed_set:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    return old_to_new
