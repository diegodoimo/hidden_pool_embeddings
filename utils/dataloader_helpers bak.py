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

    Why numpy instead of pure torch:
        ``torch.tensor(python_list)`` must traverse Python objects element by
        element.  NumPy's C layer unpacks a ``list[int]`` directly into a
        pre-allocated contiguous buffer in one shot, and ``torch.from_numpy``
        is a zero-copy view — no allocation, no data copy.

    Build the mask positionally (not value-based) so it is correct even when
    ``pad_id == eot_id``:  a value-based ``padded != pad_id`` would
    incorrectly mark the EOT token as padding in that case.

    Returns:
        padded  – LongTensor of shape (N, max_len)
        mask    – LongTensor of shape (N, max_len), 1 for real tokens, 0 for pad
    """
    n = len(token_lists)
    extra = 1 if eot_id is not None else 0
    max_len = max(len(s) for s in token_lists) + extra
    arr = np.full((n, max_len), pad_id, dtype=np.int64)
    mask_arr = np.zeros((n, max_len), dtype=np.int64)
    if padding_side == "right":
        for i, s in enumerate(token_lists):
            L = len(s)
            arr[i, :L] = s
            mask_arr[i, : L + extra] = 1
            if eot_id is not None:
                arr[i, L] = eot_id
    else:  # left padding
        for i, s in enumerate(token_lists):
            L = len(s)
            start = max_len - L - extra
            arr[i, start : start + L] = s
            mask_arr[i, start : start + L + extra] = 1
            if eot_id is not None:
                arr[i, start + L] = eot_id
    padded = torch.from_numpy(arr)
    mask = torch.from_numpy(mask_arr)
    return padded, mask


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
    """Pre-tokenized collate using ``_fast_pad`` for padding."""
    query_tokens = [item["query_token_ids"] for item in batch]
    all_docs = [item["positive_token_ids"] for item in batch] + [
        neg for item in batch for neg in item["negative_token_ids"][:num_hard_negatives]
    ]
    query_padded, query_mask = _fast_pad(
        query_tokens,
        pad_id=pad_token_id,
        eot_id=eot_id,
        padding_side=padding_side,
    )
    all_doc_padded, all_doc_mask = _fast_pad(
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
