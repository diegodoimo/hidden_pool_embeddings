import math
from collections.abc import Iterator
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
from torch.nn.utils.rnn import pad_sequence


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
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
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


class DatasetAwareSampler(Sampler[_T_co]):
    """DDP-aware sampler ensuring each batch contains items from a single dataset.

    Two strategies:
    - "sequential": All batches from dataset A first, then B, then C, etc.
      The model fully processes one dataset before moving on.
    - "grouped": Batches from different datasets interleaved (round-robin),
      but each individual batch contains items from only one dataset.
      The model alternates between datasets throughout the epoch.

    Both strategies guarantee every datapoint is processed exactly once per epoch
    (modulo minimal DDP padding).  Requires the dataset to have a "dataset_name"
    column.

    Args:
        dataset: HF Dataset with a "dataset_name" column.
        batch_size: Per-device batch size (must match DataLoader's batch_size).
        strategy: "sequential" or "grouped".
        num_replicas: Number of DDP processes (defaults to world_size).
        rank: Current DDP rank (defaults to dist.get_rank()).
        shuffle: If True, shuffle dataset/chunk order per epoch via set_epoch.
        seed: Base random seed (same across all ranks for determinism).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        strategy: str = "sequential",
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval "
                f"[0, {num_replicas - 1}]"
            )
        if strategy not in ("sequential", "grouped"):
            raise ValueError(
                f"strategy must be 'sequential' or 'grouped', got '{strategy}'"
            )

        self.batch_size = batch_size
        self.strategy = strategy
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        names = dataset["dataset_name"]
        groups: dict[str, list[int]] = {}
        for idx, name in enumerate(names):
            groups.setdefault(name, []).append(idx)

        self.dataset_order = sorted(groups.keys())

        # Pad each dataset so its size is divisible by (batch_size * num_replicas).
        # This guarantees every batch is fully homogeneous after DDP sharding.
        effective = batch_size * num_replicas
        self._per_rank: dict[str, list[int]] = {}
        self.num_samples = 0
        for name in self.dataset_order:
            indices = groups[name]
            remainder = len(indices) % effective
            if remainder:
                pad_n = effective - remainder
                full_repeats = pad_n // len(indices)
                leftover = pad_n % len(indices)
                indices = indices + indices * full_repeats + indices[:leftover]

            # Interleave across ranks (preserves relative length ordering
            # when the dataset is pre-sorted by length)
            rank_indices = indices[rank::num_replicas]
            self._per_rank[name] = rank_indices
            self.num_samples += len(rank_indices)

    def __iter__(self) -> Iterator[_T_co]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.strategy == "sequential":
            order = list(self.dataset_order)
            if self.shuffle:
                perm = torch.randperm(len(order), generator=g).tolist()
                order = [order[i] for i in perm]

            indices: list[int] = []
            for name in order:
                indices.extend(self._per_rank[name])
            return iter(indices)

        # --- grouped: round-robin interleaving of batch-sized chunks ---
        per_dataset_chunks: dict[str, list[list[int]]] = {}
        for name in self.dataset_order:
            rank_indices = self._per_rank[name]
            per_dataset_chunks[name] = [
                rank_indices[i : i + self.batch_size]
                for i in range(0, len(rank_indices), self.batch_size)
            ]

        order = list(self.dataset_order)
        if self.shuffle:
            perm = torch.randperm(len(order), generator=g).tolist()
            order = [order[i] for i in perm]

        indices = []
        iterators = {name: iter(per_dataset_chunks[name]) for name in order}
        active = list(order)
        while active:
            next_active = []
            for name in active:
                chunk = next(iterators[name], None)
                if chunk is not None:
                    indices.extend(chunk)
                    next_active.append(name)
            active = next_active

        # pyrefly: ignore [bad-return]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


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


def _str_to_int_id(s: str) -> int:
    """Deterministic hash of a string to a positive 63-bit integer."""
    import hashlib

    return int(hashlib.md5(s.encode()).hexdigest()[:15], 16)


def collate_fn_with_hard_negatives(
    batch,
    pad_token_id=0,
    num_hard_negatives=7,
    padding_side="right",
    tokenizer=None,
    eot_id=None,
    add_special_tokens=False,
    max_seq_len=None,
):
    """Collate function for batches that include hard negatives.

    Tokenizes prompts in the collate (like collate_fn_with_padding), then returns
    padded tensors for queries and all docs (positives + negatives concatenated)
    for a single forward pass.

    When max_seq_len is set (option 1 / truncation strategy), every tokenizer call
    uses truncation=True so that no sequence exceeds max_seq_len tokens.  When
    eot_id is also appended the effective content budget is max_seq_len-1 tokens
    so that the final sequence (content + eot) stays within the limit.
    """

    # Reserve one slot for eot_id when it is appended after tokenisation.
    _max_content = (max_seq_len - 1) if (max_seq_len is not None and eot_id is not None) else max_seq_len
    _trunc_kwargs = (
        {"truncation": True, "max_length": _max_content}
        if max_seq_len is not None
        else {}
    )

    # Tokenize queries (like collate_fn_with_padding)
    query_prompts = [item["query_prompt"] for item in batch]
    query_encs = tokenizer(
        query_prompts,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
        **_trunc_kwargs,
    )["input_ids"]

    if eot_id is not None:
        query_token_ids = [torch.tensor(tok + [eot_id]) for tok in query_encs]
    else:
        query_token_ids = [torch.tensor(tok) for tok in query_encs]

    # Tokenize positives
    pos_prompts = [item["positive_prompt"] for item in batch]
    pos_encs = tokenizer(
        pos_prompts,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
        **_trunc_kwargs,
    )["input_ids"]
    if eot_id is not None:
        pos_token_ids = [torch.tensor(tok + [eot_id]) for tok in pos_encs]
    else:
        pos_token_ids = [torch.tensor(tok) for tok in pos_encs]

    # Tokenize negatives per item
    all_neg_token_ids = []
    for i, item in enumerate(batch):
        neg_prompts = item["negative_prompts"][:num_hard_negatives]

        neg_encs = tokenizer(
            neg_prompts,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
            **_trunc_kwargs,
        )["input_ids"]

        if eot_id is not None:
            neg_ids = [tok + [eot_id] for tok in neg_encs]
        else:
            neg_ids = neg_encs

        all_neg_token_ids.extend([torch.tensor(n) for n in neg_ids])

    # pos_ids from dataset_name and positive_id
    pos_ids = torch.tensor(
        [
            _str_to_int_id(f"{item['dataset_name']}/{item['positive_id']}")
            for item in batch
        ],
        dtype=torch.long,
    )

    # query_ids from dataset_name and query_id
    q_ids = torch.tensor(
        [
            _str_to_int_id(f"{item['dataset_name']}/{item['query_id']}")
            for item in batch
        ],
        dtype=torch.long,
    )

    # Query attention mask: ones for content, then pad (like collate_fn_with_padding)
    query_attention_mask = [torch.ones_like(ids) for ids in query_token_ids]
    query_padded = pad_sequence(
        query_token_ids,
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )
    query_mask = pad_sequence(
        query_attention_mask,
        batch_first=True,
        padding_value=0,
        padding_side=padding_side,
    )

    # Build all_doc_seqs: [pos_0, ..., pos_{B-1}, neg_0_0, ..., neg_{B-1}_{num_neg-1}]
    all_doc_seqs = pos_token_ids + all_neg_token_ids
    all_doc_attention_mask = [torch.ones_like(ids) for ids in all_doc_seqs]

    all_doc_padded = pad_sequence(
        all_doc_seqs,
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )
    all_doc_mask = pad_sequence(
        all_doc_attention_mask,
        batch_first=True,
        padding_value=0,
        padding_side=padding_side,
    )

    return {
        "query_token_ids": query_padded,
        "query_attention_mask": query_mask,
        "all_doc_token_ids": all_doc_padded,
        "all_doc_attention_mask": all_doc_mask,
        "pos_ids": pos_ids,
        "query_ids": q_ids,
        "num_hard_negatives": num_hard_negatives,
    }
