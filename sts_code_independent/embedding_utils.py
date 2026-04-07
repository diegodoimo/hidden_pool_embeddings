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
from datasets import Dataset

from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math
from torch.utils.data.sampler import Sampler
from typing import TypeVar, Iterator


import os
import json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F


import torch
import numpy as np


_T_co = TypeVar("_T_co", covariant=True)


class LenghtSortedSampler(Sampler[_T_co]):

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
    ) -> None:

        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.

        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

        lengths = [len(instance) for instance in self.dataset["prompt"]]
        indices = list(np.argsort(lengths))[::-1]
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)

        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        if len(indices) != self.total_size:
            raise AssertionError(
                f"Number of indices ({len(indices)}) does not match total_size ({self.total_size})"
            )

        # same seed in all
        rng = np.random.RandomState(self.seed)

        # subsample
        self.indices = []
        for i in range(0, self.total_size, self.num_replicas):
            # Get the next batch of num_replicas consecutive indices
            batch_end = min(self.total_size, i + self.num_replicas)
            batch = indices[i:batch_end]

            if self.total_size - i > self.num_replicas:
                # do not shuffle the last batch
                rng.shuffle(batch)

            if self.rank < len(batch):
                self.indices.append(batch[self.rank])

        # self.indices = indices[self.rank : self.total_size : self.num_replicas]

    def __iter__(self) -> Iterator[_T_co]:

        if len(self.indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(self.indices)}) does not match num_samples ({self.num_samples})"
            )

        # pyrefly: ignore [bad-return]
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples


def collate_fn_with_padding(
    batch,
    pad_token_id=0,
    padding_side="right",
    tokenizer=None,
    eot_id=None,
    add_special_tokens=False,
):

    input_text = [item["prompt"] for item in batch]
    tokens = tokenizer(
        input_text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
    )["input_ids"]

    if eot_id is not None:
        query_token_ids = [torch.tensor(tok + [eot_id]) for tok in tokens]
    else:
        query_token_ids = [torch.tensor(tok) for tok in tokens]

    query_attention_mask = [torch.ones_like(input_ids) for input_ids in query_token_ids]

    # Pad queries and create attention masks
    query_token_ids_padded = pad_sequence(
        query_token_ids,
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )

    query_attention_mask = pad_sequence(
        query_attention_mask,
        batch_first=True,
        padding_value=0,
        padding_side=padding_side,
    )

    assert query_token_ids_padded.dtype == torch.int64, batch
    return {
        "input_ids": query_token_ids_padded,
        "attention_mask": query_attention_mask,
    }


def _build_prompt_text(
    rows,
    instruction_template,
    prompt_type,
    task_metadata,
):

    titles = rows.get("title", None)
    if titles:
        text_prompts = [
            instruction_template(prompt_type, task_metadata, text, title)
            for text, title in zip(rows["text"], rows["title"])
        ]
    else:
        text_prompts = [
            instruction_template(prompt_type, task_metadata, text)
            for text in rows["text"]
        ]

    new_rows = {
        "id": rows["id"],
        "prompt": text_prompts,
        "text": rows["text"],
    }
    return new_rows


def create_dataset(
    dataset,
    task_metadata,
    instruction_template,
    prompt_type,
):
    if "text" not in dataset.column_names:
        raise ValueError("Column 'text' not found in dataset")

    if isinstance(dataset["text"][0], list):
        raise ValueError("Can't handle queries type queries for conversation")

    input_to_dict = partial(
        _build_prompt_text,
        instruction_template=instruction_template,
        prompt_type=prompt_type,
        task_metadata=task_metadata,
    )
    new_ds = dataset.map(input_to_dict, batched=True, batch_size=10000)
    return new_ds


@torch.inference_mode()
def encode(
    model,
    loader,
    world_size,
    prompt_type,
    divided_by_chunks=False,
):

    # distributed sampler will duplicate examples at the end
    indices = None
    if hasattr(loader.sampler, "indices"):
        indices = loader.sampler.indices
        assert isinstance(indices, list)

    num_samples = len(loader.dataset)
    embeddings = []

    for batch in loader:
        batch = {key: val.to(model.device) for key, val in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batch_embeddings = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0).float()
    indices = torch.tensor(indices, device=embeddings.device)

    if prompt_type == PromptType.document and divided_by_chunks:
        # if we are processing documents divided in chunks, we postpone the allgather
        # (in the distributed setup) and return the (local) indices
        return embeddings, indices

    if world_size > 1:
        gathered = [torch.zeros_like(embeddings) for _ in range(world_size)]
        dist.all_gather(gathered, embeddings)
        embeddings = torch.cat(gathered, dim=0)[:num_samples]

        if indices is not None:
            gathered_indices = [torch.zeros_like(indices) for _ in range(world_size)]
            dist.all_gather(gathered_indices, indices)
            indices = torch.cat(gathered_indices, dim=0)[:num_samples]

    # Restore original order
    sorted_positions = torch.argsort(indices)
    embeddings = embeddings[sorted_positions]
    return embeddings


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


def make_collate_fn(tokenizer, padding_side, eot_id, add_special_tokens):
    """Build a reusable collate function from tokenizer params."""
    return partial(
        collate_fn_with_padding,
        pad_token_id=tokenizer.pad_token_id,
        padding_side=padding_side,
        tokenizer=tokenizer,
        eot_id=eot_id,
        add_special_tokens=add_special_tokens,
    )


def encode_dataset(model, dataset, batch_size, collate_fn):
    """Encode a prepared dataset using the DDP-aware pipeline.

    This is a standalone replacement for the former
    ``evaluate_retrieval._encode_dataset`` method.
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
        prompt_type=PromptType.query,
        world_size=world_size,
    )
    dist.barrier()
    return embeddings


# ---------------------------------------------------------------------------
# Dataset preparation helpers
# ---------------------------------------------------------------------------


def prepare_text_dataset(
    texts, task_metadata, instruction_template, tokenizer, rank, max_length=8192
):
    """Create a prompt-augmented HF dataset from raw texts for encoding.

    Returns
    -------
    (ds, removed_indices) : tuple
        The filtered dataset and the set of original integer positions that
        were removed (too long or empty).
    """
    dataset = Dataset.from_dict(
        {"id": [str(i) for i in range(len(texts))], "text": texts}
    )
    ds = create_dataset(
        dataset=dataset,
        task_metadata=task_metadata,
        instruction_template=instruction_template,
        prompt_type=PromptType.query,
    )
    return ds


def build_index_remap(n_original, removed_set):
    """Build old-index → new-index mapping after removing items."""
    old_to_new = {}
    new_idx = 0
    for old_idx in range(n_original):
        if old_idx not in removed_set:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    return old_to_new


# --------------------------------------------------------------------
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def mean_pool(last_hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
    masked_sum = (last_hidden_states * mask).sum(dim=1)  # (B, H)
    mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    return masked_sum / mask_sum


def add_pooling_layers(model, pool_fn, projection_layers=None):
    """Wrap a model so its forward pass includes pooling and L2 normalization.

    *projection_layers* (optional ``nn.Module``) is inserted between pooling
    and L2-norm — use :func:`load_st_dense_layers` to build one for models
    that ship Dense projections in their SentenceTransformer checkpoint.
    """
    return EncoderWithPooling(model, pool_fn, projection_layers=projection_layers)


class EncoderWithPooling(nn.Module):
    """
    Wraps an encoder model to perform pooling and L2 normalization in the forward pass.
    The wrapped model's forward should return an object with `last_hidden_state` (e.g. BaseModelOutput).

    If *projection_layers* is provided it is applied between pooling and
    normalization.  This is needed for models like ``embeddinggemma-300m``
    whose SentenceTransformer checkpoint ships two Dense layers (768→3072→768)
    on top of mean-pooling.
    """

    def __init__(self, model, pool_fn, projection_layers=None):
        super().__init__()
        self.model = model
        self.pool_fn = pool_fn
        self.projection = projection_layers

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        pooled = self.pool_fn(output.last_hidden_state, attention_mask)
        if self.projection is not None:
            pooled = self.projection(pooled)
        return F.normalize(pooled, p=2, dim=1)


_ABSTASK_FALLBACK_PROMPTS = {
    "STS": "Retrieve semantically similar text.",
    "Summarization": "Given a news summary, retrieve other semantically similar summaries.",
    "Retrieval": "Retrieve text based on user query.",
    "Reranking": "Retrieve text based on user query.",
    "Classification": "Classify user passages.",
    "Clustering": "Identify categories in user passages.",
    "PairClassification": "Retrieve text that are semantically similar to the given text.",
    "BitextMining": "Retrieve parallel sentences.",
}


def instruction_template_qwen3(prompt_type, task_metadata, text, title="") -> str:

    if prompt_type == PromptType.query:
        instruction = _ABSTASK_FALLBACK_PROMPTS["STS"]
        prompt = f"Instruct: {instruction.strip()}\nQuery:{text.strip()}"

    elif prompt_type == PromptType.document:

        if len(title) > 0:
            prompt = f"{title} {text.strip()}"
        else:
            prompt = text.strip()

    return prompt


_EMBEDDINGGEMMA_STS_PROMPT = "task: sentence similarity | query: "


def instruction_template_embeddinggemma(prompt_type, task_metadata, text, title="") -> str:

    if prompt_type == PromptType.query:
        prompt = f"{_EMBEDDINGGEMMA_STS_PROMPT}{text.strip()}"

    elif prompt_type == PromptType.document:

        if len(title) > 0:
            prompt = f"title: {title} | text: {text.strip()}"
        else:
            prompt = f"title: none | text: {text.strip()}"

    return prompt
