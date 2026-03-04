#!/usr/bin/env python3
"""Tokenize hard-negative datasets and save with the same folder structure.

Reads parquet files from datasets_negatives/ and writes tokenized data to
datasets_tokenized/ preserving the folder structure. The output folder is named
<dataset_folder>_<instruction_template>, e.g. qwen3_600m_data_gemma3_prompt.

Usage:
    python tokenize_data_script.py \
        --input_dir datasets_negatives \
        --output_dir datasets_tokenized \
        --instruction_template gemma3_prompt \
        --tokenizer_path google/t5gemma-2-270m-270m \
        [--num_hard_negatives 8] \
        [--max_seq_len 1024] \
        [--datasets_subset retrieval/general_retrieval/arguana]
"""

import argparse
import glob
import json
import os
import warnings
from collections import Counter
from functools import partial
from typing import Optional

from transformers import AutoTokenizer

from tasks import NAME_TO_TASK_TYPE
from tasks.f2llm_prompts import TASK_TO_PROMPT
from utils.create_datasets import (
    TASK_TYPE_TO_TASK_METADATA,
    instruction_template_embeddinggemma,
    instruction_template_qwen3,
    _build_and_tokenize_hard_negatives_batch,
    _build_prompts_hard_negatives_batch,
    _load_parquet_safe,
)

# Map F2LLM parquet source names (TASK_TO_PROMPT keys) to NAME_TO_TASK keys.
# Only entries that differ are listed; others match by identity.
F2LLM_SOURCE_TO_NAME_TO_TASK = {
    "amazon_qa": "amazonqa",
    "natural_questions": "naturalquestions",
    "mr_tydi": "mrtydi",
    "cnn_dm": "cnndm",
    "stackexchange_dup_questions_s2s": "stackexchange_dup_s2s",
    "stackexchange_dup_questions_p2p": "stackexchange_dup_p2p",
    "stackoverflow_dup_questions": "stackoverflow_dup",
    "sts_benchmark": "stsbenchmark",
    "tweet_sentiment_extraction": "tweet_sentiment",
    "twenty_newsgroups": "twentynewsgroups",
}

INSTRUCTION_TEMPLATES = {
    "qwen3": (instruction_template_qwen3, False),
    "embeddinggemma": (instruction_template_embeddinggemma, True),
}

# Columns that the training dataloader actually consumes.  Stripping the raw
# text columns keeps the output files compact and consistent with what
# create_pretokenized_hard_negatives_datasets produces.
_COLS_TO_KEEP = {
    "query_prompt",
    "positive_prompt",
    "negative_prompts",
    "positive_id",
    "query_id",
    "dataset_name",
    "total_length",
    "query_token_ids",
    "positive_token_ids",
    "negative_token_ids",
}


def get_instruction_template(name: str):
    """Return (template_fn, add_special_tokens) for the given template name."""
    if name not in INSTRUCTION_TEMPLATES:
        raise ValueError(
            f"Unknown instruction_template '{name}'. "
            f"Choices: {list(INSTRUCTION_TEMPLATES.keys())}"
        )
    return INSTRUCTION_TEMPLATES[name]


def _compute_and_save_metadata(
    output_path: str,
    ds_name: str,
    n_rows_raw: int,
    rows_saved: int,
    query_prompts: list,
    positive_prompts: list,
    all_neg_flat: list,
) -> None:
    """Compute deduplication statistics and write metadata.json next to the parquet.

    Fields written:
      total_queries / unique_queries   — rows vs distinct query prompt strings
      total_positives / unique_positives — same for positive passages
      total_negatives                  — sum of all negative slots across rows
      unique_negatives                 — number of distinct negative passages
      negative_count_distribution      — counts-of-counts: {k: n} means n unique
                                         negatives each appeared in exactly k rows
      rows_saved                       — rows written to the output parquet

    Stats are measured on *raw* (pre-filter) data to reflect the underlying
    dataset independently of the chosen max_seq_len threshold.
    """
    neg_counter: Counter = Counter(all_neg_flat)
    count_of_counts: Counter = Counter(neg_counter.values())

    metadata = {
        "dataset": ds_name,
        "total_queries": n_rows_raw,
        "unique_queries": len(set(query_prompts)),
        "total_positives": n_rows_raw,
        "unique_positives": len(set(positive_prompts)),
        "total_negatives": len(all_neg_flat),
        "unique_negatives": len(neg_counter),
        # {"k": n} → n unique negatives each appear as a negative for exactly k queries
        "negative_count_distribution": {
            str(k): v for k, v in sorted(count_of_counts.items())
        },
        "rows_saved": rows_saved,
    }

    metadata_path = os.path.join(os.path.dirname(output_path), "metadata.json")
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)


def _f2llm_source_to_name_to_task(f2llm_source: str) -> str:
    """Map F2LLM parquet source name to NAME_TO_TASK key."""
    return F2LLM_SOURCE_TO_NAME_TO_TASK.get(f2llm_source, f2llm_source)


def _strip_f2llm_prompt(text: str, prompt: str) -> str:
    """Remove F2LLM prompt template from the start of text if present."""
    if not text or not prompt:
        return text
    text = text.strip()
    # Try exact prompt prefix (with optional trailing space/newline/punctuation)
    for prefix in (
        prompt.strip(),
        prompt.strip() + " ",
        prompt.strip() + "\n",
        "Instruct: " + prompt.strip() + "\nQuery:",
        "Instruct: " + prompt.strip() + "\nQuery: ",
        "Instruct: " + prompt.strip() + "\\nQuery:",
        "Instruct: " + prompt.strip() + "\\nQuery: ",
    ):
        if text.startswith(prefix):
            rest = text[len(prefix) :].strip()
            return rest if rest else text
    return text


def tokenize_f2llm_dataset(
    f2llm_source: str,
    output_path: str,
    tokenizer,
    instruction_template,
    add_special_tokens: bool,
    num_hard_negatives: int,
    max_seq_len: Optional[int],
    num_proc: int = 1,
) -> int:
    """Tokenize a single F2LLM dataset source and save to parquet.

    Loads data via the download_data pipeline (codefuse-ai/F2LLM). Only processes
    sources that map to NAME_TO_TASK (for instruction template support). Strips the
    F2LLM prompt template (from TASK_TO_PROMPT) from query/passage text before
    applying the instruction template.

    Args:
        f2llm_source: F2LLM parquet source name (e.g. 'arguana', 'amazon_qa').
        output_path: Path for output data.parquet.
        tokenizer: HuggingFace tokenizer.
        instruction_template: Instruction template callable.
        add_special_tokens: Pass to tokenizer.
        num_hard_negatives: Max hard negatives per row.
        max_seq_len: Filter rows exceeding this length (None = no filter).
        num_proc: Workers for dataset.map.

    Returns:
        Number of rows saved.

    Raises:
        Nothing; logs warning and returns 0 if source not in NAME_TO_TASK.
    """
    from datasets import Dataset
    from download_data import load_f2llm

    ds_name = _f2llm_source_to_name_to_task(f2llm_source)
    if ds_name not in NAME_TO_TASK_TYPE:
        warnings.warn(
            f"F2LLM source '{f2llm_source}' maps to '{ds_name}' which is not in "
            f"NAME_TO_TASK. Skipping (no instruction template available).",
            UserWarning,
            stacklevel=2,
        )
        return 0

    task_type = NAME_TO_TASK_TYPE[ds_name]
    task_metadata = TASK_TYPE_TO_TASK_METADATA[task_type]

    f2llm_prompt = TASK_TO_PROMPT.get(f2llm_source)
    if f2llm_prompt is None:
        warnings.warn(
            f"F2LLM source '{f2llm_source}' not in TASK_TO_PROMPT; "
            "cannot strip F2LLM prompt template. Skipping tokenization.",
            UserWarning,
            stacklevel=2,
        )
        return 0

    ds = load_f2llm(sources=[f2llm_source])
    n_rows_raw = len(ds)

    # Convert F2LLM columns (query, passage, negative_1..24) to expected format
    # and strip F2LLM prompt template if present
    def _convert_and_strip(batch):
        queries = batch["query"]
        passages = batch["passage"]
        stripped_queries = []
        stripped_passages = []
        stripped_negs = []
        for i in range(len(queries)):
            q = queries[i] or ""
            p = passages[i] or ""
            if f2llm_prompt:
                q = _strip_f2llm_prompt(q, f2llm_prompt)
                p = _strip_f2llm_prompt(p, f2llm_prompt)
            stripped_queries.append(q)
            stripped_passages.append(p)
            negs = []
            for j in range(1, 25):
                col = f"negative_{j}"
                n = ""
                if col in batch and i < len(batch[col]):
                    n = batch[col][i] or ""
                n = str(n).strip()
                if n:
                    if f2llm_prompt:
                        n = _strip_f2llm_prompt(n, f2llm_prompt)
                    negs.append(n)
            stripped_negs.append(negs)
        return {
            "query_text": stripped_queries,
            "positive_text": stripped_passages,
            "negative_text": stripped_negs,
            "query_id": [str(k) for k in range(len(queries))],
            "positive_id": [f"pos_{k}" for k in range(len(passages))],
        }

    # HF Dataset columns are lists when indexed (e.g. ds["query"] = [q1, q2, ...])
    batch = {col: ds[col] for col in ds.column_names}
    converted = _convert_and_strip(batch)
    ds = Dataset.from_dict(converted)

    # Reuse tokenize_and_save_dataset_dedup logic: build prompts, dedup, tokenize
    build_fn = partial(
        _build_prompts_hard_negatives_batch,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        task_metadata=task_metadata,
        num_hard_negatives=num_hard_negatives,
    )
    ds = ds.map(build_fn, batched=True, batch_size=10000, num_proc=num_proc)

    query_prompts = ds["query_prompt"]
    positive_prompts = ds["positive_prompt"]
    negative_prompts_lists = ds["negative_prompts"]
    all_neg_flat = [p for row in negative_prompts_lists for p in row]

    all_unique_prompts = list(
        dict.fromkeys(query_prompts + positive_prompts + all_neg_flat)
    )

    print(
        f"  [{ds_name}] tokenizing {len(all_unique_prompts):,} unique prompts "
        f"(from {n_rows_raw:,} rows)"
    )
    all_token_ids = tokenizer(
        all_unique_prompts,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
        truncation=False,
    )["input_ids"]

    prompt_to_ids = dict(zip(all_unique_prompts, all_token_ids))
    query_token_ids = [prompt_to_ids[p] for p in query_prompts]
    positive_token_ids = [prompt_to_ids[p] for p in positive_prompts]
    negative_token_ids = [
        [prompt_to_ids[p] for p in row] for row in negative_prompts_lists
    ]

    keep_indices = list(range(n_rows_raw))
    if max_seq_len is not None:
        keep_indices = [
            i
            for i in range(n_rows_raw)
            if (
                len(query_token_ids[i]) <= max_seq_len
                and len(positive_token_ids[i]) <= max_seq_len
                and all(len(t) <= max_seq_len for t in negative_token_ids[i])
            )
        ]
        n_filtered = n_rows_raw - len(keep_indices)
        if n_filtered:
            print(
                f"  [{ds_name}] filtered {n_filtered} / {n_rows_raw} rows "
                f"exceeding max_seq_len={max_seq_len}"
            )
            query_token_ids = [query_token_ids[i] for i in keep_indices]
            positive_token_ids = [positive_token_ids[i] for i in keep_indices]
            negative_token_ids = [negative_token_ids[i] for i in keep_indices]
            ds = ds.select(keep_indices)

    ds = ds.add_column("query_token_ids", query_token_ids)
    ds = ds.add_column("positive_token_ids", positive_token_ids)
    ds = ds.add_column("negative_token_ids", negative_token_ids)
    ds = ds.add_column("dataset_name", [ds_name] * len(ds))

    ds = ds.sort("total_length", reverse=True)

    cols_to_remove = [c for c in ds.column_names if c not in _COLS_TO_KEEP]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_parquet(output_path)

    _compute_and_save_metadata(
        output_path=output_path,
        ds_name=ds_name,
        n_rows_raw=n_rows_raw,
        rows_saved=len(ds),
        query_prompts=query_prompts,
        positive_prompts=positive_prompts,
        all_neg_flat=all_neg_flat,
    )

    return len(ds)


def tokenize_and_save_dataset_dedup(
    input_path: str,
    output_path: str,
    ds_name: str,
    tokenizer,
    instruction_template,
    add_special_tokens: bool,
    num_hard_negatives: int,
    max_seq_len: Optional[int],
    num_proc: int = 1,
) -> int:
    """Load, tokenize, and save a dataset using a deduplicate-then-tokenize strategy.

    Inspired by tokenize_reference.py:
    1. Build prompt strings for all rows (parallelised via num_proc).
    2. Collect all *unique* prompts across queries, positives and negatives.
    3. Tokenize unique prompts in ONE batched call — passages shared across
       many queries (e.g. MSMARCO corpus) are tokenized only once.
    4. Assemble per-row token-ID lists from the lookup dict in O(1).
    5. Filter, sort, strip columns, save parquet + metadata.json.
    """
    dataset_name = os.path.basename(os.path.dirname(input_path))
    if dataset_name not in NAME_TO_TASK_TYPE:
        raise ValueError(
            f"Dataset '{dataset_name}' not in NAME_TO_TASK_TYPE. "
            "Cannot determine task metadata."
        )

    task_type = NAME_TO_TASK_TYPE[dataset_name]
    task_metadata = TASK_TYPE_TO_TASK_METADATA[task_type]

    ds = _load_parquet_safe(input_path)
    n_rows_raw = len(ds)

    # ------------------------------------------------------------------
    # Step 1: Build prompt strings (no tokenization yet).
    # num_proc parallelises the string-building work across CPU cores.
    # ------------------------------------------------------------------
    build_fn = partial(
        _build_prompts_hard_negatives_batch,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        task_metadata=task_metadata,
        num_hard_negatives=num_hard_negatives,
    )
    ds = ds.map(build_fn, batched=True, batch_size=10000, num_proc=num_proc)
    # ds now has: query_prompt, positive_prompt, negative_prompts, total_length
    # (plus all original columns)

    query_prompts: list[str] = ds["query_prompt"]
    positive_prompts: list[str] = ds["positive_prompt"]
    negative_prompts_lists: list[list[str]] = ds["negative_prompts"]

    # ------------------------------------------------------------------
    # Step 2: Deduplicate across ALL roles together.
    # Queries use a different instruction prefix than documents, so there
    # is no accidental collision between a query prompt and a passage prompt.
    # ------------------------------------------------------------------
    all_neg_flat: list[str] = [p for row in negative_prompts_lists for p in row]

    # dict.fromkeys preserves first-seen order and deduplicates in O(n)
    all_unique_prompts: list[str] = list(
        dict.fromkeys(query_prompts + positive_prompts + all_neg_flat)
    )

    # ------------------------------------------------------------------
    # Step 3: Tokenize unique prompts in ONE batched call.
    # The Rust rayon thread-pool inside the fast tokenizer handles
    # internal parallelism; no need for Python multiprocessing here.
    # ------------------------------------------------------------------
    print(
        f"  [{dataset_name}] tokenizing {len(all_unique_prompts):,} unique prompts "
        f"(from {n_rows_raw:,} rows, "
        f"{len(all_neg_flat):,} total neg slots)"
    )
    all_token_ids: list[list[int]] = tokenizer(
        all_unique_prompts,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
        truncation=False,
    )["input_ids"]

    prompt_to_ids: dict[str, list[int]] = dict(zip(all_unique_prompts, all_token_ids))

    # ------------------------------------------------------------------
    # Step 4: Assemble per-row token-ID lists via O(1) dict lookup.
    # ------------------------------------------------------------------
    query_token_ids: list[list[int]] = [prompt_to_ids[p] for p in query_prompts]
    positive_token_ids: list[list[int]] = [prompt_to_ids[p] for p in positive_prompts]
    negative_token_ids: list[list[list[int]]] = [
        [prompt_to_ids[p] for p in row] for row in negative_prompts_lists
    ]

    # ------------------------------------------------------------------
    # Step 5 (optional): filter rows exceeding max_seq_len.
    # Uses the already-computed token IDs — no re-tokenization.
    # ------------------------------------------------------------------
    keep_indices = list(range(n_rows_raw))
    if max_seq_len is not None:
        keep_indices = [
            i
            for i in range(n_rows_raw)
            if (
                len(query_token_ids[i]) <= max_seq_len
                and len(positive_token_ids[i]) <= max_seq_len
                and all(len(t) <= max_seq_len for t in negative_token_ids[i])
            )
        ]
        n_filtered = n_rows_raw - len(keep_indices)
        if n_filtered:
            print(
                f"  [{dataset_name}] filtered {n_filtered} / {n_rows_raw} rows "
                f"exceeding max_seq_len={max_seq_len}"
            )
            query_token_ids = [query_token_ids[i] for i in keep_indices]
            positive_token_ids = [positive_token_ids[i] for i in keep_indices]
            negative_token_ids = [negative_token_ids[i] for i in keep_indices]
            ds = ds.select(keep_indices)

    # ------------------------------------------------------------------
    # Step 6: Attach token-ID columns and tidy up.
    # ------------------------------------------------------------------
    ds = ds.add_column("query_token_ids", query_token_ids)
    ds = ds.add_column("positive_token_ids", positive_token_ids)
    ds = ds.add_column("negative_token_ids", negative_token_ids)

    if "dataset_name" not in ds.column_names:
        ds = ds.add_column("dataset_name", [ds_name] * len(ds))

    ds = ds.sort("total_length", reverse=True)

    cols_to_remove = [c for c in ds.column_names if c not in _COLS_TO_KEEP]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_parquet(output_path)

    # ------------------------------------------------------------------
    # Step 7: Save metadata.
    # ------------------------------------------------------------------
    _compute_and_save_metadata(
        output_path=output_path,
        ds_name=ds_name,
        n_rows_raw=n_rows_raw,
        rows_saved=len(ds),
        query_prompts=query_prompts,
        positive_prompts=positive_prompts,
        all_neg_flat=all_neg_flat,
    )

    return len(ds)


def tokenize_and_save_dataset_batch(
    input_path: str,
    output_path: str,
    ds_name: str,
    tokenizer,
    instruction_template,
    add_special_tokens: bool,
    num_hard_negatives: int,
    max_seq_len: Optional[int],
    num_proc: int = 1,
) -> int:
    """Load, tokenize, and save a dataset using HF batched map (no deduplication).

    Original pipeline approach: prompt building and tokenization are merged in a
    single Dataset.map() call via _build_and_tokenize_hard_negatives_batch, then
    rows exceeding max_seq_len are removed with Dataset.filter().

    Faster than tokenize_and_save_dataset_dedup when passage reuse is low (e.g.
    STS, NLI, one-to-one pair datasets).  For datasets with heavy passage reuse
    (MSMARCO, NaturalQuestions, …) the dedup variant is significantly faster.
    """
    dataset_name = os.path.basename(os.path.dirname(input_path))
    if dataset_name not in NAME_TO_TASK_TYPE:
        raise ValueError(
            f"Dataset '{dataset_name}' not in NAME_TO_TASK_TYPE. "
            "Cannot determine task metadata."
        )

    task_type = NAME_TO_TASK_TYPE[dataset_name]
    task_metadata = TASK_TYPE_TO_TASK_METADATA[task_type]

    ds = _load_parquet_safe(input_path)
    n_rows_raw = len(ds)

    build_tok_fn = partial(
        _build_and_tokenize_hard_negatives_batch,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        task_metadata=task_metadata,
        num_hard_negatives=num_hard_negatives,
        add_special_tokens=add_special_tokens,
    )
    ds = ds.map(
        build_tok_fn,
        batched=True,
        batch_size=10000,
        num_proc=num_proc,
    )

    # Collect raw prompt lists for metadata before any filtering.
    query_prompts: list[str] = ds["query_prompt"]
    positive_prompts: list[str] = ds["positive_prompt"]
    all_neg_flat: list[str] = [p for row in ds["negative_prompts"] for p in row]

    # Filter using already-computed token IDs — no re-tokenization.
    if max_seq_len is not None:
        n_before = len(ds)
        query_token_ids: list = ds["query_token_ids"]
        positive_token_ids: list = ds["positive_token_ids"]
        negative_token_ids: list = ds["negative_token_ids"]
        keep_indices = [
            i
            for i in range(n_before)
            if (
                len(query_token_ids[i]) <= max_seq_len
                and len(positive_token_ids[i]) <= max_seq_len
                and all(len(t) <= max_seq_len for t in negative_token_ids[i])
            )
        ]
        n_filtered = n_before - len(keep_indices)
        if n_filtered:
            print(
                f"  [{dataset_name}] filtered {n_filtered} / {n_before} rows "
                f"exceeding max_seq_len={max_seq_len}"
            )
            ds = ds.select(keep_indices)

    if "dataset_name" not in ds.column_names:
        ds = ds.add_column("dataset_name", [ds_name] * len(ds))
    ds = ds.sort("total_length", reverse=True)

    cols_to_remove = [c for c in ds.column_names if c not in _COLS_TO_KEEP]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_parquet(output_path)

    _compute_and_save_metadata(
        output_path=output_path,
        ds_name=ds_name,
        n_rows_raw=n_rows_raw,
        rows_saved=len(ds),
        query_prompts=query_prompts,
        positive_prompts=positive_prompts,
        all_neg_flat=all_neg_flat,
    )

    return len(ds)


def _main_f2llm(args):
    """Tokenize F2LLM datasets from HF cache."""
    from download_data import get_f2llm_sources

    instruction_template, add_special_tokens = get_instruction_template(
        args.instruction_template
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    all_sources = get_f2llm_sources()
    want = set(args.f2llm_sources) if args.f2llm_sources else None
    sources_to_process = [s for s in all_sources if want is None or s in want]

    output_folder = f"f2llm_{args.instruction_template}"
    output_dir = os.path.join(args.output_dir, output_folder)
    print(f"F2LLM mode: {len(sources_to_process)} sources from HF cache")
    print(f"Output: {output_dir}/<source>/data.parquet")
    print("(Skipping sources not in NAME_TO_TASK)\n")

    total_rows = 0
    processed = 0
    for i, f2llm_source in enumerate(sources_to_process):
        output_path = os.path.join(output_dir, f2llm_source, "data.parquet")
        ds_name = _f2llm_source_to_name_to_task(f2llm_source)
        if ds_name not in NAME_TO_TASK_TYPE:
            warnings.warn(
                f"Skipping '{f2llm_source}' (maps to '{ds_name}', not in NAME_TO_TASK)",
                UserWarning,
                stacklevel=2,
            )
            continue
        print(f"Processing [{processed + 1}] {f2llm_source} -> {ds_name}")
        try:
            n = tokenize_f2llm_dataset(
                f2llm_source=f2llm_source,
                output_path=output_path,
                tokenizer=tokenizer,
                instruction_template=instruction_template,
                add_special_tokens=add_special_tokens,
                num_hard_negatives=args.num_hard_negatives,
                max_seq_len=args.max_seq_len,
                num_proc=args.num_workers,
            )
            total_rows += n
            processed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            raise

    print(f"\nDone. Processed {processed} sources, total rows saved: {total_rows}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize datasets from datasets_negatives into datasets_tokenized."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets_negatives",
        help="Root directory containing hard-negative datasets (e.g. datasets_negatives)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets_tokenized",
        help="Root directory for tokenized output (e.g. datasets_tokenized)",
    )
    parser.add_argument(
        "--instruction_template",
        type=str,
        default="embeddinggemma",
        choices=list(INSTRUCTION_TEMPLATES.keys()),
        help="Instruction template: qwen3, embeddinggemma",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="HuggingFace model path for the tokenizer (e.g. google/t5gemma-2-270m-270m)",
    )
    parser.add_argument(
        "--num_hard_negatives",
        type=int,
        default=8,
        help="Number of hard negatives per example",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Filter out rows where any component exceeds this token length. If None, no filtering.",
    )
    parser.add_argument(
        "--datasets_subset",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of dataset paths to restrict (e.g. retrieval/general_retrieval/arguana). If omitted, process all.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for dataset map/filter (passed as num_proc to HF datasets). "
        "Use os.cpu_count() for maximum parallelism. Values > 1 set TOKENIZERS_PARALLELISM=false "
        "to avoid forking conflicts with Rust tokenizers.",
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="dedup",
        choices=["dedup", "batch"],
        help=(
            "Tokenization strategy. "
            "'dedup': build prompts first, deduplicate across the dataset, tokenize unique "
            "strings once — best for datasets with heavy passage reuse (MSMARCO, NQ, …). "
            "'batch': combined build+tokenize HF map — simpler, best when passage reuse is low."
        ),
    )
    parser.add_argument(
        "--f2llm",
        action="store_true",
        help="Tokenize F2LLM data (codefuse-ai/F2LLM) from HF cache. Uses download_data pipeline. "
        "Only sources in NAME_TO_TASK are processed. Strips TASK_TO_PROMPT template before applying instruction template.",
    )
    parser.add_argument(
        "--f2llm_sources",
        type=str,
        nargs="*",
        default=None,
        help="Restrict F2LLM sources to process (e.g. arguana amazon_qa msmarco). If omitted, process all in NAME_TO_TASK.",
    )
    args = parser.parse_args()

    if args.f2llm:
        _main_f2llm(args)
        return

    if args.num_workers > 1:
        # Prevent Rust tokenizer threads from being forked inside worker processes.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    instruction_template, add_special_tokens = get_instruction_template(
        args.instruction_template
    )
    # use_fast=True (default): Rust fast tokenizer is 10-100× faster than the
    # pure-Python fallback.  trust_remote_code is kept for custom model repos.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    pattern = os.path.join(args.input_dir, "**", "data.parquet")
    parquet_files = sorted(glob.glob(pattern, recursive=True))

    if not parquet_files:
        print(f"No data.parquet files found under {args.input_dir}")
        return

    if args.datasets_subset is not None:
        subset_set = set(args.datasets_subset)

        # Match by inner path (e.g. retrieval/general_retrieval/arguana) so the
        # subset applies across all dataset folders
        def _inner_path(p):
            rel = os.path.relpath(p, args.input_dir)
            parts = rel.split(os.sep)
            return os.path.join(*parts[1:-1]) if len(parts) > 1 else ""

        parquet_files = [p for p in parquet_files if _inner_path(p) in subset_set]
        print(
            f"Restricted to {len(parquet_files)} datasets (subset of {len(args.datasets_subset)} requested)"
        )

    print(f"Found {len(parquet_files)} datasets under {args.input_dir}")
    print(f"Output: {args.output_dir}/<dataset_folder>_{args.instruction_template}/")

    total_rows = 0
    for i, input_path in enumerate(parquet_files):
        rel_path = os.path.relpath(input_path, args.input_dir)
        # rel_path: e.g. qwen3_600m_data/retrieval/general_retrieval/arguana/data.parquet
        # or: qwen3_600m/retrieval/general_retrieval/arguana/data.parquet
        parts = rel_path.split(os.sep)
        dataset_folder = parts[0]
        inner_path = os.path.join(
            *parts[1:]
        )  # retrieval/general_retrieval/arguana/data.parquet

        output_folder = f"{dataset_folder}_{args.instruction_template}"
        output_path = os.path.join(args.output_dir, output_folder, inner_path)

        print(f"Processing [{i + 1}/{len(parquet_files)}] {rel_path}")
        ds_name = os.path.dirname(
            inner_path
        )  # e.g. retrieval/general_retrieval/arguana
        tokenize_fn = (
            tokenize_and_save_dataset_dedup
            if args.implementation == "dedup"
            else tokenize_and_save_dataset_batch
        )
        n = tokenize_fn(
            input_path=input_path,
            output_path=output_path,
            ds_name=ds_name,
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            add_special_tokens=add_special_tokens,
            num_hard_negatives=args.num_hard_negatives,
            max_seq_len=args.max_seq_len,
            num_proc=args.num_workers,
        )
        total_rows += n

    print(f"\nDone. Total rows saved: {total_rows}")


if __name__ == "__main__":
    main()
