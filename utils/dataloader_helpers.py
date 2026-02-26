import math
from collections.abc import Iterator
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
from torch.nn.utils.rnn import pad_sequence

__all__ = ["DistributedSampler"]

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


def collate_fn_with_hard_negatives(batch, pad_token_id=0, num_hard_negatives=8):
    """Collate function for batches that include hard negatives.

    Returns padded tensors for queries, positives, and negatives with their
    attention masks.
    """
    query_token_ids = [torch.tensor(item["query_token_ids"]) for item in batch]
    pos_token_ids = [torch.tensor(item["pos_token_ids"]) for item in batch]
    pos_ids = torch.tensor([item["pos_ids"] for item in batch], dtype=torch.long)

    # Pad queries
    query_padded = pad_sequence(
        query_token_ids, batch_first=True, padding_value=pad_token_id
    )
    query_mask = (query_padded != pad_token_id).long()

    # Pad positives
    pos_padded = pad_sequence(
        pos_token_ids, batch_first=True, padding_value=pad_token_id
    )
    pos_mask = (pos_padded != pad_token_id).long()

    # Flatten all negatives across the batch, pad, then reshape
    all_neg_seqs = []
    for item in batch:
        for neg in item["neg_token_ids"]:
            all_neg_seqs.append(torch.tensor(neg))

    neg_padded = pad_sequence(
        all_neg_seqs, batch_first=True, padding_value=pad_token_id
    )
    neg_mask = (neg_padded != pad_token_id).long()

    batch_size = len(batch)
    neg_seq_len = neg_padded.size(1)
    neg_padded = neg_padded.view(batch_size, num_hard_negatives, neg_seq_len)
    neg_mask = neg_mask.view(batch_size, num_hard_negatives, neg_seq_len)

    return {
        "query_token_ids": query_padded,
        "query_attention_mask": query_mask,
        "pos_token_ids": pos_padded,
        "pos_attention_mask": pos_mask,
        "neg_token_ids": neg_padded,
        "neg_attention_mask": neg_mask,
        "pos_ids": pos_ids,
    }
