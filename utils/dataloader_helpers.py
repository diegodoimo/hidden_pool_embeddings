import math
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from typing import TypeVar

import torch
import torch.distributed as dist
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

# ---------------------------------------------------------------------------
# Characters-per-token safety factor for pre-text truncation.  A generous
# factor (8) ensures we never cut too aggressively – worst-case BPE tokens
# are 1-2 chars (CJK), so 8 covers all scripts with ample margin.
# ---------------------------------------------------------------------------
_CHARS_PER_TOKEN_BUDGET = 8


def _encode_batch_fast(
    tokenizer,
    texts: list[str],
    add_special_tokens: bool,
    max_token_len: int | None,
) -> list[list[int]]:
    """Tokenise *texts* via the low-level Rust ``encode_batch``.

    Three combined optimisations over ``PreTrainedTokenizerFast.__call__``:

    1. **Pre-truncate text** to ``max_token_len * 8`` characters so the
       tokeniser never processes throwaway trailing text for very long
       documents.
    2. **Direct ``encode_batch``** – bypasses the Python-level
       ``BatchEncoding`` wrapper, padding / attention-mask construction,
       and other overhead that is redundant here (padding is handled later
       by ``_fast_pad``).
    3. **Single batched call** for all texts (queries + positives +
       negatives) lets the Rust rayon pool distribute work optimally across
       all CPU cores, instead of splitting into two thread-pool futures
       with imbalanced load.
    """
    # --- (1) character pre-truncation ---
    if max_token_len is not None:
        budget = max_token_len * _CHARS_PER_TOKEN_BUDGET
        texts = [t[:budget] if len(t) > budget else t for t in texts]

    # --- (2) Rust encode_batch (releases GIL, uses rayon) ---
    encodings = tokenizer._tokenizer.encode_batch(
        texts,
        add_special_tokens=add_special_tokens,
    )

    # --- (3) extract IDs + token-level truncation ---
    if max_token_len is not None:
        return [enc.ids[:max_token_len] for enc in encodings]
    return [enc.ids for enc in encodings]


def _fast_pad(
    token_lists: list[list[int]],
    pad_id: int,
    eot_id: int | None = None,
    padding_side: str = "right",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of token-id lists into (N, L) tensors without creating N
    intermediate 1-D torch.Tensors.

    Steps:
    1. Allocate a numpy int64 array pre-filled with pad_id  (one alloc).
    2. Copy each sequence via a numpy slice (much faster than torch.tensor
       per element and avoids the Python-object overhead of torch.tensor).
    3. Convert to torch via zero-copy from_numpy.
    4. Build the attention mask with a single vectorised comparison instead
       of creating N ones_like tensors and padding them separately.

    Returns:
        padded  – LongTensor of shape (N, max_len)
        mask    – LongTensor of shape (N, max_len), 1 for real tokens, 0 for pad
    """
    n = len(token_lists)
    extra = 1 if eot_id is not None else 0
    max_len = max(len(s) for s in token_lists) + extra
    arr = np.full((n, max_len), pad_id, dtype=np.int64)
    if padding_side == "right":
        for i, s in enumerate(token_lists):
            L = len(s)
            arr[i, :L] = s
            if eot_id is not None:
                arr[i, L] = eot_id
    else:  # left padding
        for i, s in enumerate(token_lists):
            L = len(s)
            start = max_len - L - extra
            arr[i, start : start + L] = s
            if eot_id is not None:
                arr[i, start + L] = eot_id
    padded = torch.from_numpy(arr)
    mask = (padded != pad_id).to(dtype=torch.long)
    return padded, mask


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


# def collate_fn_with_hard_negatives_v0(
#     batch,
#     pad_token_id=0,
#     num_hard_negatives=7,
#     padding_side="right",
#     tokenizer=None,
#     eot_id=None,
#     add_special_tokens=False,
#     max_seq_len=None,
#     timing_stats=None,
# ):
#     """Collate function for batches that include hard negatives.

#     Tokenizes prompts in the collate (like collate_fn_with_padding), then returns
#     padded tensors for queries and all docs (positives + negatives concatenated)
#     for a single forward pass.

#     When max_seq_len is set (option 1 / truncation strategy), every tokenizer call
#     uses truncation=True so that no sequence exceeds max_seq_len tokens.  When
#     eot_id is also appended the effective content budget is max_seq_len-1 tokens
#     so that the final sequence (content + eot) stays within the limit.
#     """
#     import time as _time

#     _bench = timing_stats is not None

#     def _tick() -> float:
#         return _time.perf_counter() if _bench else 0.0

#     def _record(key: str, t0: float) -> None:
#         if _bench:
#             timing_stats[key] = timing_stats.get(key, 0.0) + (_time.perf_counter() - t0)

#     t_total = _tick()

#     # Reserve one slot for eot_id when it is appended after tokenisation.
#     _max_content = (
#         (max_seq_len - 1)
#         if (max_seq_len is not None and eot_id is not None)
#         else max_seq_len
#     )
#     _trunc_kwargs = (
#         {"truncation": True, "max_length": _max_content}
#         if max_seq_len is not None
#         else {}
#     )

#     # --- Extract prompts ---
#     t0 = _tick()
#     query_prompts = [item["query_prompt"] for item in batch]
#     pos_prompts = [item["positive_prompt"] for item in batch]
#     _record("prompt_extract", t0)

#     # --- Tokenize queries ---
#     t0 = _tick()
#     query_encs = tokenizer(
#         query_prompts,
#         add_special_tokens=add_special_tokens,
#         return_attention_mask=False,
#         **_trunc_kwargs,
#     )["input_ids"]

#     if eot_id is not None:
#         query_token_ids = [torch.tensor(tok + [eot_id]) for tok in query_encs]
#     else:
#         query_token_ids = [torch.tensor(tok) for tok in query_encs]

#     # Tokenize positives
#     pos_encs = tokenizer(
#         pos_prompts,
#         add_special_tokens=add_special_tokens,
#         return_attention_mask=False,
#         **_trunc_kwargs,
#     )["input_ids"]
#     if eot_id is not None:
#         pos_token_ids = [torch.tensor(tok + [eot_id]) for tok in pos_encs]
#     else:
#         pos_token_ids = [torch.tensor(tok) for tok in pos_encs]

#     # Tokenize negatives per item (one tokenizer call per sample)
#     all_neg_token_ids = []
#     for item in batch:
#         neg_prompts = item["negative_prompts"][:num_hard_negatives]

#         neg_encs = tokenizer(
#             neg_prompts,
#             add_special_tokens=add_special_tokens,
#             return_attention_mask=False,
#             **_trunc_kwargs,
#         )["input_ids"]

#         if eot_id is not None:
#             neg_ids = [tok + [eot_id] for tok in neg_encs]
#         else:
#             neg_ids = neg_encs

#         all_neg_token_ids.extend([torch.tensor(n) for n in neg_ids])
#     _record("tokenize_parallel", t0)

#     # --- Build sample-ID tensors ---
#     t0 = _tick()
#     pos_ids = torch.tensor(
#         [
#             _str_to_int_id(f"{item['dataset_name']}/{item['positive_id']}")
#             for item in batch
#         ],
#         dtype=torch.long,
#     )
#     q_ids = torch.tensor(
#         [
#             _str_to_int_id(f"{item['dataset_name']}/{item['query_id']}")
#             for item in batch
#         ],
#         dtype=torch.long,
#     )
#     _record("id_build", t0)

#     # --- Pad queries ---
#     t0 = _tick()
#     query_attention_mask = [torch.ones_like(ids) for ids in query_token_ids]
#     query_padded = pad_sequence(
#         query_token_ids,
#         batch_first=True,
#         padding_value=pad_token_id,
#         padding_side=padding_side,
#     )
#     query_mask = pad_sequence(
#         query_attention_mask,
#         batch_first=True,
#         padding_value=0,
#         padding_side=padding_side,
#     )
#     _record("query_pad", t0)

#     # --- Pad docs ---
#     t0 = _tick()
#     all_doc_seqs = pos_token_ids + all_neg_token_ids
#     all_doc_attention_mask = [torch.ones_like(ids) for ids in all_doc_seqs]

#     all_doc_padded = pad_sequence(
#         all_doc_seqs,
#         batch_first=True,
#         padding_value=pad_token_id,
#         padding_side=padding_side,
#     )
#     all_doc_mask = pad_sequence(
#         all_doc_attention_mask,
#         batch_first=True,
#         padding_value=0,
#         padding_side=padding_side,
#     )
#     _record("doc_pad", t0)

#     _record("total", t_total)
#     if _bench:
#         timing_stats["_calls"] = timing_stats.get("_calls", 0) + 1

#     return {
#         "query_token_ids": query_padded,
#         "query_attention_mask": query_mask,
#         "all_doc_token_ids": all_doc_padded,
#         "all_doc_attention_mask": all_doc_mask,
#         "pos_ids": pos_ids,
#         "query_ids": q_ids,
#         "num_hard_negatives": num_hard_negatives,
#     }


# def collate_fn_with_hard_negatives_v01(
#     batch,
#     pad_token_id=0,
#     num_hard_negatives=7,
#     padding_side="right",
#     tokenizer=None,
#     eot_id=None,
#     add_special_tokens=False,
#     max_seq_len=None,
#     timing_stats=None,
# ):
#     """Collate function for batches that include hard negatives.

#     Tokenizes prompts in the collate (like collate_fn_with_padding), then returns
#     padded tensors for queries and all docs (positives + negatives concatenated)
#     for a single forward pass.

#     When max_seq_len is set (option 1 / truncation strategy), every tokenizer call
#     uses truncation=True so that no sequence exceeds max_seq_len tokens.  When
#     eot_id is also appended the effective content budget is max_seq_len-1 tokens
#     so that the final sequence (content + eot) stays within the limit.
#     """
#     import time as _time

#     _bench = timing_stats is not None

#     def _tick() -> float:
#         return _time.perf_counter() if _bench else 0.0

#     def _record(key: str, t0: float) -> None:
#         if _bench:
#             timing_stats[key] = timing_stats.get(key, 0.0) + (_time.perf_counter() - t0)

#     t_total = _tick()

#     # Reserve one slot for eot_id when it is appended after tokenisation.
#     _max_content = (
#         (max_seq_len - 1)
#         if (max_seq_len is not None and eot_id is not None)
#         else max_seq_len
#     )
#     _trunc_kwargs = (
#         {"truncation": True, "max_length": _max_content}
#         if max_seq_len is not None
#         else {}
#     )

#     # --- Extract prompts ---
#     t0 = _tick()
#     query_prompts = [item["query_prompt"] for item in batch]
#     pos_prompts = [item["positive_prompt"] for item in batch]
#     flat_neg_prompts: list[str] = []
#     for item in batch:
#         flat_neg_prompts.extend(item["negative_prompts"][:num_hard_negatives])
#     _record("prompt_extract", t0)

#     # --- Tokenize: 3 separate batched calls (query, pos, all negatives) ---
#     t0 = _tick()
#     query_encs = tokenizer(
#         query_prompts,
#         add_special_tokens=add_special_tokens,
#         return_attention_mask=False,
#         **_trunc_kwargs,
#     )["input_ids"]

#     if eot_id is not None:
#         query_token_ids = [torch.tensor(tok + [eot_id]) for tok in query_encs]
#     else:
#         query_token_ids = [torch.tensor(tok) for tok in query_encs]

#     pos_encs = tokenizer(
#         pos_prompts,
#         add_special_tokens=add_special_tokens,
#         return_attention_mask=False,
#         **_trunc_kwargs,
#     )["input_ids"]
#     if eot_id is not None:
#         pos_token_ids = [torch.tensor(tok + [eot_id]) for tok in pos_encs]
#     else:
#         pos_token_ids = [torch.tensor(tok) for tok in pos_encs]

#     # --- OLD: Tokenize negatives per item (one tokenizer call per sample) ---
#     # all_neg_token_ids = []
#     # for i, item in enumerate(batch):
#     #     neg_prompts = item["negative_prompts"][:num_hard_negatives]
#     #
#     #     neg_encs = tokenizer(
#     #         neg_prompts,
#     #         add_special_tokens=add_special_tokens,
#     #         return_attention_mask=False,
#     #         **_trunc_kwargs,
#     #     )["input_ids"]
#     #
#     #     if eot_id is not None:
#     #         neg_ids = [tok + [eot_id] for tok in neg_encs]
#     #     else:
#     #         neg_ids = neg_encs
#     #
#     #     all_neg_token_ids.extend([torch.tensor(n) for n in neg_ids])
#     # --- END OLD ---

#     # Tokenize negatives – batched across the entire batch for one tokenizer call
#     flat_neg_encs = tokenizer(
#         flat_neg_prompts,
#         add_special_tokens=add_special_tokens,
#         return_attention_mask=False,
#         **_trunc_kwargs,
#     )["input_ids"]

#     if eot_id is not None:
#         all_neg_token_ids = [torch.tensor(tok + [eot_id]) for tok in flat_neg_encs]
#     else:
#         all_neg_token_ids = [torch.tensor(tok) for tok in flat_neg_encs]
#     _record("tokenize_parallel", t0)

#     # --- Build sample-ID tensors ---
#     t0 = _tick()
#     pos_ids = torch.tensor(
#         [
#             _str_to_int_id(f"{item['dataset_name']}/{item['positive_id']}")
#             for item in batch
#         ],
#         dtype=torch.long,
#     )
#     q_ids = torch.tensor(
#         [
#             _str_to_int_id(f"{item['dataset_name']}/{item['query_id']}")
#             for item in batch
#         ],
#         dtype=torch.long,
#     )
#     _record("id_build", t0)

#     # --- Pad queries ---
#     t0 = _tick()
#     query_attention_mask = [torch.ones_like(ids) for ids in query_token_ids]
#     query_padded = pad_sequence(
#         query_token_ids,
#         batch_first=True,
#         padding_value=pad_token_id,
#         padding_side=padding_side,
#     )
#     query_mask = pad_sequence(
#         query_attention_mask,
#         batch_first=True,
#         padding_value=0,
#         padding_side=padding_side,
#     )
#     _record("query_pad", t0)

#     # --- Pad docs ---
#     t0 = _tick()
#     all_doc_seqs = pos_token_ids + all_neg_token_ids
#     all_doc_attention_mask = [torch.ones_like(ids) for ids in all_doc_seqs]

#     all_doc_padded = pad_sequence(
#         all_doc_seqs,
#         batch_first=True,
#         padding_value=pad_token_id,
#         padding_side=padding_side,
#     )
#     all_doc_mask = pad_sequence(
#         all_doc_attention_mask,
#         batch_first=True,
#         padding_value=0,
#         padding_side=padding_side,
#     )
#     _record("doc_pad", t0)

#     _record("total", t_total)
#     if _bench:
#         timing_stats["_calls"] = timing_stats.get("_calls", 0) + 1

#     return {
#         "query_token_ids": query_padded,
#         "query_attention_mask": query_mask,
#         "all_doc_token_ids": all_doc_padded,
#         "all_doc_attention_mask": all_doc_mask,
#         "pos_ids": pos_ids,
#         "query_ids": q_ids,
#         "num_hard_negatives": num_hard_negatives,
#     }


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
    # import time as _time

    # _bench = timing_stats is not None

    # def _tick() -> float:
    #     return _time.perf_counter() if _bench else 0.0

    # def _record(key: str, t0: float) -> None:
    #     if _bench:
    #         timing_stats[key] = timing_stats.get(key, 0.0) + (_time.perf_counter() - t0)

    # t_total = _tick()

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
    #   Thread 0 tokenises query + pos in one batched call  (~10 ms)
    #   Thread 1 tokenises all negatives                    (~59 ms)
    # Wall time ≈ max(10, 59) instead of 10 + 59.
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

    # t0 = _tick()
    f_qpos = _COLLATE_TOKEN_POOL.submit(_tok, query_prompts + pos_prompts)
    f_neg = _COLLATE_TOKEN_POOL.submit(_tok, flat_neg_prompts)
    qpos_encs = f_qpos.result()
    flat_neg_encs = f_neg.result()
    # _record("tokenize_parallel", t0)

    query_encs = qpos_encs[:B]
    pos_encs = qpos_encs[B:]

    # --- Build sample-ID tensors (pos_ids / query_ids) ---
    # t0 = _tick()
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
    # _record("id_build", t0)

    # --- Pad queries via _fast_pad (numpy fill + zero-copy from_numpy) ---
    # This replaces: creating B individual tensors, creating B ones_like tensors,
    # two pad_sequence calls.
    # t0 = _tick()
    query_padded, query_mask = _fast_pad(
        query_encs, pad_id=pad_token_id, eot_id=eot_id, padding_side=padding_side
    )
    # _record("query_pad", t0)

    # --- Pad positives and negatives ---
    # Positives and negatives are padded together (same semantic role, same
    # downstream usage) so all_doc_padded has shape
    # (B + B*num_hard_negatives, max_doc_len).
    # t0 = _tick()
    all_doc_padded, all_doc_mask = _fast_pad(
        pos_encs + flat_neg_encs,
        pad_id=pad_token_id,
        eot_id=eot_id,
        padding_side=padding_side,
    )
    # _record("doc_pad", t0)

    # _record("total", t_total)
    # if _bench:
    #     timing_stats["_calls"] = timing_stats.get("_calls", 0) + 1

    return {
        "query_token_ids": query_padded,
        "query_attention_mask": query_mask,
        "all_doc_token_ids": all_doc_padded,
        "all_doc_attention_mask": all_doc_mask,
        "pos_ids": pos_ids,
        "query_ids": q_ids,
        "num_hard_negatives": num_hard_negatives,
    }


# def collate_fn_with_hard_negatives_v2(
#     batch,
#     pad_token_id=0,
#     num_hard_negatives=7,
#     padding_side="right",
#     tokenizer=None,
#     eot_id=None,
#     add_special_tokens=False,
#     max_seq_len=None,
#     timing_stats=None,
# ):
#     """Optimised collate function – drop-in replacement for
#     ``collate_fn_with_hard_negatives``.

#     Three improvements over the original:

#     1. **Single Rust ``encode_batch``** for all texts (queries + positives +
#        negatives) instead of two thread-pool futures with imbalanced load.
#        The Rust rayon pool distributes work optimally across all CPU cores.
#     2. **Direct ``tokenizer._tokenizer.encode_batch``** bypasses the
#        Python-level ``BatchEncoding`` wrapper, padding / attention-mask
#        construction, and other overhead that is redundant here.
#     3. **Pre-text character truncation** clips long documents to
#        ``max_token_len * 8`` chars *before* tokenisation so the encoder
#        never processes throwaway trailing text.

#     Return dict is identical to ``collate_fn_with_hard_negatives``.
#     """
#     import time as _time

#     _bench = timing_stats is not None

#     def _tick() -> float:
#         return _time.perf_counter() if _bench else 0.0

#     def _record(key: str, t0: float) -> None:
#         if _bench:
#             timing_stats[key] = timing_stats.get(key, 0.0) + (_time.perf_counter() - t0)

#     t_total = _tick()

#     # Reserve one slot for eot_id when it is appended after tokenisation.
#     _max_content = (
#         (max_seq_len - 1)
#         if (max_seq_len is not None and eot_id is not None)
#         else max_seq_len
#     )

#     B = len(batch)

#     # --- Build all prompt lists ---
#     t0 = _tick()
#     query_prompts = [item["query_prompt"] for item in batch]
#     pos_prompts = [item["positive_prompt"] for item in batch]
#     flat_neg_prompts: list[str] = []
#     for item in batch:
#         flat_neg_prompts.extend(item["negative_prompts"][:num_hard_negatives])
#     _record("prompt_extract", t0)

#     # --- Tokenisation (single Rust encode_batch – see _encode_batch_fast) ---
#     t0 = _tick()
#     all_texts = query_prompts + pos_prompts + flat_neg_prompts
#     all_ids = _encode_batch_fast(
#         tokenizer,
#         all_texts,
#         add_special_tokens,
#         _max_content,
#     )
#     _record("tokenize_parallel", t0)

#     query_encs = all_ids[:B]
#     pos_encs = all_ids[B : 2 * B]
#     flat_neg_encs = all_ids[2 * B :]

#     # --- Build sample-ID tensors (pos_ids / query_ids) ---
#     t0 = _tick()
#     pos_ids = torch.tensor(
#         [
#             _str_to_int_id(f"{item['dataset_name']}/{item['positive_id']}")
#             for item in batch
#         ],
#         dtype=torch.long,
#     )
#     q_ids = torch.tensor(
#         [
#             _str_to_int_id(f"{item['dataset_name']}/{item['query_id']}")
#             for item in batch
#         ],
#         dtype=torch.long,
#     )
#     _record("id_build", t0)

#     # --- Pad queries ---
#     t0 = _tick()
#     query_padded, query_mask = _fast_pad(
#         query_encs, pad_id=pad_token_id, eot_id=eot_id, padding_side=padding_side
#     )
#     _record("query_pad", t0)

#     # --- Pad positives + negatives ---
#     t0 = _tick()
#     all_doc_padded, all_doc_mask = _fast_pad(
#         pos_encs + flat_neg_encs,
#         pad_id=pad_token_id,
#         eot_id=eot_id,
#         padding_side=padding_side,
#     )
#     _record("doc_pad", t0)

#     _record("total", t_total)
#     if _bench:
#         timing_stats["_calls"] = timing_stats.get("_calls", 0) + 1

#     return {
#         "query_token_ids": query_padded,
#         "query_attention_mask": query_mask,
#         "all_doc_token_ids": all_doc_padded,
#         "all_doc_attention_mask": all_doc_mask,
#         "pos_ids": pos_ids,
#         "query_ids": q_ids,
#         "num_hard_negatives": num_hard_negatives,
#     }


# def collate_fn_pretokenized(
#     batch,
#     pad_token_id=0,
#     num_hard_negatives=7,
#     padding_side="right",
#     eot_id=None,
#     max_seq_len=None,
#     timing_stats=None,
# ):
#     """Collate function for pre-tokenized batches (no tokenizer calls).

#     Expects each item in *batch* to carry ``query_token_ids``,
#     ``positive_token_ids`` and ``negative_token_ids`` columns produced by
#     ``create_pretokenized_hard_negatives_datasets``.  Only padding and tensor
#     construction happen here — tokenization cost is zero.

#     The return dict has the same schema as ``collate_fn_with_hard_negatives``
#     so callers (Trainer, benchmark script) are interchangeable.

#     Args:
#         timing_stats: optional dict-like for per-step wall-clock accumulation.
#     """
#     # import time as _time

#     # _bench = timing_stats is not None

#     # def _tick() -> float:
#     #     return _time.perf_counter() if _bench else 0.0

#     # def _record(key: str, t0: float) -> None:
#     #     if _bench:
#     #         timing_stats[key] = timing_stats.get(key, 0.0) + (_time.perf_counter() - t0)

#     # t_total = _tick()

#     _max_content = (
#         (max_seq_len - 1)
#         if (max_seq_len is not None and eot_id is not None)
#         else max_seq_len
#     )

#     # --- Extract cached token-ID lists ---
#     # t0 = _tick()
#     query_encs = [item["query_token_ids"] for item in batch]
#     pos_encs = [item["positive_token_ids"] for item in batch]
#     flat_neg_encs: list[list[int]] = []
#     for item in batch:
#         flat_neg_encs.extend(item["negative_token_ids"][:num_hard_negatives])

#     # Apply truncation if max_seq_len was requested
#     if _max_content is not None:
#         query_encs = [ids[:_max_content] for ids in query_encs]
#         pos_encs = [ids[:_max_content] for ids in pos_encs]
#         flat_neg_encs = [ids[:_max_content] for ids in flat_neg_encs]
#     # _record("extract_ids", t0)

#     # # --- Build sample-ID tensors ---
#     # t0 = _tick()
#     pos_ids = torch.tensor(
#         [
#             _str_to_int_id(f"{item['dataset_name']}/{item['positive_id']}")
#             for item in batch
#         ],
#         dtype=torch.long,
#     )
#     q_ids = torch.tensor(
#         [
#             _str_to_int_id(f"{item['dataset_name']}/{item['query_id']}")
#             for item in batch
#         ],
#         dtype=torch.long,
#     )
#     # _record("id_build", t0)

#     # --- Pad queries ---
#     # t0 = _tick()
#     query_padded, query_mask = _fast_pad(
#         query_encs, pad_id=pad_token_id, eot_id=eot_id, padding_side=padding_side
#     )
#     # _record("query_pad", t0)

#     # --- Pad positives + negatives ---
#     # t0 = _tick()
#     all_doc_padded, all_doc_mask = _fast_pad(
#         pos_encs + flat_neg_encs,
#         pad_id=pad_token_id,
#         eot_id=eot_id,
#         padding_side=padding_side,
#     )
#     # _record("doc_pad", t0)

#     # _record("total", t_total)
#     # if _bench:
#     #     timing_stats["_calls"] = timing_stats.get("_calls", 0) + 1

#     return {
#         "query_token_ids": query_padded,
#         "query_attention_mask": query_mask,
#         "all_doc_token_ids": all_doc_padded,
#         "all_doc_attention_mask": all_doc_mask,
#         "pos_ids": pos_ids,
#         "query_ids": q_ids,
#         "num_hard_negatives": num_hard_negatives,
#     }


def collate_fn_pretokenized(
    batch,
    pad_token_id=0,
    num_hard_negatives=7,
    padding_side="right",
    eot_id=None,
):
    query_tokens = [item["query_token_ids"] for item in batch]
    all_docs = [item["positive_token_ids"] for item in batch] + [
        neg for item in batch for neg in item["negative_token_ids"][:num_hard_negatives]
    ]
    if eot_id is not None:
        query_token_ids = [torch.tensor(tok + [eot_id]) for tok in query_tokens]
        all_docs_ids = [torch.tensor(tok + [eot_id]) for tok in all_docs]
    else:
        query_token_ids = [torch.tensor(tok) for tok in query_tokens]
        all_docs_ids = [torch.tensor(tok) for tok in all_docs]

    query_attention_mask = [torch.ones_like(input_ids) for input_ids in query_token_ids]
    all_docs_attention_mask = [torch.ones_like(input_ids) for input_ids in all_docs_ids]

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

    doc_ids_padded = pad_sequence(
        all_docs_ids,
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )

    docs_attention_mask = pad_sequence(
        all_docs_attention_mask,
        batch_first=True,
        padding_value=0,
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
        "query_token_ids": query_token_ids_padded,
        "query_attention_mask": query_attention_mask,
        "all_doc_token_ids": doc_ids_padded,
        "all_doc_attention_mask": docs_attention_mask,
        "pos_ids": pos_ids,
        "query_ids": q_ids,
        "num_hard_negatives": num_hard_negatives,
    }



def collate_fn_pretokenized_fast_pad(
    batch,
    pad_token_id=0,
    num_hard_negatives=7,
    padding_side="right",
    eot_id=None,
):
    query_tokens = [item["query_token_ids"] for item in batch]
    all_docs = [item["positive_token_ids"] for item in batch] + [
        neg for item in batch for neg in item["negative_token_ids"][:num_hard_negatives]
    ]
    # Pad queries and create attention masks
    query_padded, query_mask = _fast_pad(
        _query_tokens,
        pad_id=pad_token_id,
        eot_id=eot_id,
        padding_side=padding_side,
    )

    all_doc_padded, all_doc_mask = _fast_pad(
        all_docs_ids,
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
        "query_token_ids": query_token_ids_padded,
        "query_attention_mask": query_attention_mask,
        "all_doc_token_ids": all_doc_padded,
        "all_doc_attention_mask": all_doc_mask,
        "pos_ids": pos_ids,
        "query_ids": q_ids,
        "num_hard_negatives": num_hard_negatives,
    }
