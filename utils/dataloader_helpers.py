import math
import random
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
from torch.nn.utils.rnn import pad_sequence


_T_co = TypeVar("_T_co", covariant=True)

# Persistent thread pool used by collate_fn_with_hard_negatives to overlap the
# (query + pos) and neg tokeniser calls.  HF fast tokenisers (Rust/rayon)
# release the GIL, so two Python threads give genuine CPU parallelism.
# With spawn workers the module is re-imported in each worker, so each worker
# gets its own 2-thread pool – that is intentional and correct.
_COLLATE_TOKEN_POOL = _ThreadPoolExecutor(max_workers=2)


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


# ---------------------------------------------------------------------------
# MultiDatasetLoader – faithful reimplementation of the ``MultiLoader``
# from CodeFuse-Embeddings/F2LLM/run.py, adapted for PyTorch DDP.
#
# Each dataset gets its own DataLoader (with its own DistributedSampler if
# running multi-GPU).  At every epoch a new random order is created; a
# dataset is selected with probability proportional to its remaining
# batches.  The epoch ends when every DataLoader is exhausted.
# ---------------------------------------------------------------------------


class MultiDatasetLoader:
    """Weighted-random multi-dataset batch iterator (F2LLM ``MultiLoader``).

    Wraps a ``{name: DataLoader}`` dict and at each iteration randomly
    selects a dataset weighted by the number of remaining batches,
    yielding one batch at a time.  The epoch ends when every DataLoader
    has been exhausted.

    Compatible with the ``train.py`` training loop:

    * ``len(loader)`` returns the total number of batches across all datasets.
    * ``loader.sampler.set_epoch(e)`` propagates the epoch to every
      underlying ``DistributedSampler`` (if present) and seeds the
      random selection order.
    * Iterating (``for batch in loader``) creates fresh iterators for each
      underlying DataLoader every time.

    Usage example::

        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        loaders = {}
        for name, ds in dataset_dict.items():
            sampler = DistributedSampler(ds, shuffle=True, seed=42)
            loaders[name] = DataLoader(ds, batch_size=bs,
                                       sampler=sampler, collate_fn=collate_fn)

        multi_loader = MultiDatasetLoader(loaders)

        for epoch in range(num_epochs):
            multi_loader.sampler.set_epoch(epoch)
            for batch in multi_loader:
                ...
    """

    # ------------------------------------------------------------------
    # Lightweight proxy so ``loader.sampler.set_epoch(e)`` works, which
    # is what train.py calls at the top of each epoch.
    # ------------------------------------------------------------------
    class _EpochProxy:
        """Proxy object exposing a ``set_epoch`` method on ``MultiDatasetLoader``."""

        def __init__(self, parent: "MultiDatasetLoader"):
            self._parent = parent

        def set_epoch(self, epoch: int) -> None:
            self._parent._epoch = epoch
            # Propagate to every underlying sampler that supports set_epoch
            for loader in self._parent.loader_dict.values():
                if hasattr(loader.sampler, "set_epoch"):
                    loader.sampler.set_epoch(epoch)

    def __init__(self, loader_dict: dict[str, DataLoader]):
        self.loader_dict = loader_dict
        self.sampler = self._EpochProxy(self)
        self._epoch = 0

    def __len__(self) -> int:
        return sum(len(v) for v in self.loader_dict.values())

    def __iter__(self):
        rng = random.Random(self._epoch)
        iters: dict[str, Iterator] = {k: iter(v) for k, v in self.loader_dict.items()}
        names = list(iters.keys())
        weights = [len(self.loader_dict[k]) for k in names]

        while names:
            name = rng.choices(names, weights=weights)[0]
            try:
                batch = next(iters[name])
                yield batch
            except StopIteration:
                idx = names.index(name)
                names.pop(idx)
                weights.pop(idx)


def _fast_pad_v2(
    token_lists: list[list[int]],
    pad_id: int,
    eot_id: int | None = None,
    padding_side: str = "right",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimised drop-in replacement for ``_fast_pad``.

    Two additional improvements over ``_fast_pad``:

    1. **``np.ones`` + zero the padding tail** – initialising to ones and
       then zeroing only the trailing (or leading) padding writes
       ``max_len - L`` elements per row instead of ``L`` elements.  For
       length-sorted batches (the common training case) the tail is nearly
       empty, so mask construction costs almost nothing.

    2. **``eot_id`` branch hoisted outside the loop** – the conditional is
       evaluated once per call rather than once per sequence.

    Correctness guarantee (``pad_id == eot_id``) is inherited from
    ``_fast_pad``: the mask is positional, not value-based.

    Returns:
        padded  – LongTensor of shape (N, max_len)
        mask    – LongTensor of shape (N, max_len), 1 for real tokens, 0 for pad
    """
    n = len(token_lists)
    extra = 1 if eot_id is not None else 0
    max_len = max(len(s) for s in token_lists) + extra
    arr = np.full((n, max_len), pad_id, dtype=np.int64)
    mask_arr = np.ones((n, max_len), dtype=np.int64)

    if padding_side == "right":
        if eot_id is not None:
            for i, s in enumerate(token_lists):
                L = len(s)
                arr[i, :L] = s
                arr[i, L] = eot_id
                if L + extra < max_len:
                    mask_arr[i, L + extra :] = 0
        else:
            for i, s in enumerate(token_lists):
                L = len(s)
                arr[i, :L] = s
                if L < max_len:
                    mask_arr[i, L:] = 0
    else:  # left padding
        if eot_id is not None:
            for i, s in enumerate(token_lists):
                L = len(s)
                start = max_len - L - extra
                arr[i, start : start + L] = s
                arr[i, start + L] = eot_id
                if start:
                    mask_arr[i, :start] = 0
        else:
            for i, s in enumerate(token_lists):
                L = len(s)
                start = max_len - L
                arr[i, start:] = s
                if start:
                    mask_arr[i, :start] = 0

    padded = torch.from_numpy(arr)
    mask = torch.from_numpy(mask_arr)
    return padded, mask


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
    timing_stats=None,
):
    """Collate function for batches that include hard negatives.

    Tokenizes prompts in the collate (like collate_fn_with_padding), then returns
    padded tensors for queries and all docs (positives + negatives concatenated)
    for a single forward pass.

    When max_seq_len is set (option 1 / truncation strategy), every tokenizer call
    uses truncation=True so that no sequence exceeds max_seq_len tokens.  When
    eot_id is also appended the effective content budget is max_seq_len-1 tokens
    so that the final sequence (content + eot) stays within the limit.

    Args:
        timing_stats: optional dict-like (e.g. ``collections.defaultdict(float)``)
            that accumulates per-step wall-clock seconds.  Key ``"_calls"`` counts
            invocations so the caller can compute per-batch averages.  Only works
            with ``num_workers=0`` in the DataLoader (workers are separate processes).
    """

    # Reserve one slot for eot_id when it is appended after tokenisation.
    _max_content = (
        (max_seq_len - 1)
        if (max_seq_len is not None and eot_id is not None)
        else max_seq_len
    )

    B = len(batch)

    # --- Build all prompt lists before launching threads ---
    # t0 = _tick()
    query_prompts = [item["query_prompt"] for item in batch]
    pos_prompts = [item["positive_prompt"] for item in batch]
    flat_neg_prompts: list[str] = []
    for item in batch:
        flat_neg_prompts.extend(item["negative_prompts"][:num_hard_negatives])
    # _record("prompt_extract", t0)

    # --- Parallel tokenisation ---
    # HF fast tokenisers (Rust/rayon) release the GIL, so two Python threads
    # give genuine CPU overlap:
    _trunc_kwargs = (
        {"truncation": True, "max_length": _max_content}
        if max_seq_len is not None
        else {}
    )

    def _tok(texts: list[str]) -> list[list[int]]:
        return tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
            **_trunc_kwargs,
        )["input_ids"]

    f_qpos = _COLLATE_TOKEN_POOL.submit(_tok, query_prompts + pos_prompts)
    f_neg = _COLLATE_TOKEN_POOL.submit(_tok, flat_neg_prompts)
    qpos_encs = f_qpos.result()
    flat_neg_encs = f_neg.result()

    query_encs = qpos_encs[:B]
    pos_encs = qpos_encs[B:]

    pos_ids = torch.tensor(
        [
            _str_to_int_id(f"{item['dataset_name']}/{item['positive_id']}")
            for item in batch
        ],
        dtype=torch.long,
    )
    q_ids = torch.tensor(
        [
            _str_to_int_id(f"{item['dataset_name']}/{item['query_id']}")
            for item in batch
        ],
        dtype=torch.long,
    )

    query_padded, query_mask = _fast_pad_v2(
        query_encs, pad_id=pad_token_id, eot_id=eot_id, padding_side=padding_side
    )

    all_doc_padded, all_doc_mask = _fast_pad_v2(
        pos_encs + flat_neg_encs,
        pad_id=pad_token_id,
        eot_id=eot_id,
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


def collate_fn_pretokenized_fast_pad_v2(
    batch,
    pad_token_id=0,
    num_hard_negatives=7,
    padding_side="right",
    eot_id=None,
):
    """Pre-tokenized collate using ``_fast_pad_v2`` for padding.

    Drop-in replacement for ``collate_fn_pretokenized_fast_pad`` that uses
    the further-optimised ``_fast_pad_v2`` (hoisted branch + np.ones tail-
    zeroing strategy).  Return dict is identical.

    The batch dict includes ``dataset_name`` (first item's name, valid when
    using ``DatasetAwareSampler`` or ``MultiDatasetLoader`` which guarantee
    intra-batch homogeneity) so the training loop can make per-task loss
    decisions.
    """
    query_tokens = [item["query_token_ids"] for item in batch]
    all_docs = [item["positive_token_ids"] for item in batch] + [
        neg for item in batch for neg in item["negative_token_ids"][:num_hard_negatives]
    ]
    query_padded, query_mask = _fast_pad_v2(
        query_tokens,
        pad_id=pad_token_id,
        eot_id=eot_id,
        padding_side=padding_side,
    )
    all_doc_padded, all_doc_mask = _fast_pad_v2(
        all_docs,
        pad_id=pad_token_id,
        eot_id=eot_id,
        padding_side=padding_side,
    )
    pos_ids = torch.tensor(
        [
            _str_to_int_id(f"{item['dataset_name']}/{item['positive_id']}")
            for item in batch
        ],
        dtype=torch.long,
    )
    q_ids = torch.tensor(
        [
            _str_to_int_id(f"{item['dataset_name']}/{item['query_id']}")
            for item in batch
        ],
        dtype=torch.long,
    )
    return {
        "query_token_ids": query_padded,
        "query_attention_mask": query_mask,
        "all_doc_token_ids": all_doc_padded,
        "all_doc_attention_mask": all_doc_mask,
        "pos_ids": pos_ids,
        "query_ids": q_ids,
        "num_hard_negatives": num_hard_negatives,
        "dataset_name": batch[0]["dataset_name"],
    }
