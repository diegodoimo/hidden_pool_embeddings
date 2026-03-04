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


def parse_args():
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
    return args


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


def _get_task_metadata(dataset_name: str):
    """Return task metadata for *dataset_name*, raising ValueError if unknown."""
    if dataset_name not in NAME_TO_TASK_TYPE:
        raise ValueError(
            f"Dataset '{dataset_name}' not in NAME_TO_TASK_TYPE. "
            "Cannot determine task metadata."
        )
    return TASK_TYPE_TO_TASK_METADATA[NAME_TO_TASK_TYPE[dataset_name]]


def _build_prompts(
    ds, tokenizer, instruction_template, task_metadata, num_hard_negatives, num_proc
):
    """Add query_prompt, positive_prompt, negative_prompts and total_length columns."""
    build_fn = partial(
        _build_prompts_hard_negatives_batch,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        task_metadata=task_metadata,
        num_hard_negatives=num_hard_negatives,
    )
    return ds.map(build_fn, batched=True, batch_size=10000, num_proc=num_proc)


def _tokenize_dedup(
    tokenizer,
    add_special_tokens: bool,
    label: str,
    q_prompts: list,
    p_prompts: list,
    neg_prompts_lists: list,
) -> tuple:
    """Deduplicate then tokenize all prompts in one batched call.

    Returns (q_ids, p_ids, n_ids, all_neg_flat) where *all_neg_flat* is the
    flattened list of all negative prompt strings (useful for metadata).
    """
    all_neg_flat = [p for row in neg_prompts_lists for p in row]
    all_unique = list(dict.fromkeys(list(q_prompts) + list(p_prompts) + all_neg_flat))
    print(
        f"  [{label}] tokenizing {len(all_unique):,} unique prompts "
        f"(from {len(q_prompts):,} rows, {len(all_neg_flat):,} total neg slots)"
    )
    all_ids = tokenizer(
        all_unique,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
        truncation=False,
    )["input_ids"]
    id_map = dict(zip(all_unique, all_ids))
    q_ids = [id_map[p] for p in q_prompts]
    p_ids = [id_map[p] for p in p_prompts]
    n_ids = [[id_map[p] for p in row] for row in neg_prompts_lists]
    return q_ids, p_ids, n_ids, all_neg_flat


def _apply_seq_len_filter(ds, q_ids, p_ids, n_ids, max_seq_len, label):
    """Remove rows where any token sequence exceeds *max_seq_len*.

    Returns (ds, q_ids, p_ids, n_ids) with offending rows dropped.
    No-op when *max_seq_len* is None.
    """
    if max_seq_len is None:
        return ds, q_ids, p_ids, n_ids
    n_before = len(ds)
    keep = [
        i
        for i in range(n_before)
        if (
            len(q_ids[i]) <= max_seq_len
            and len(p_ids[i]) <= max_seq_len
            and all(len(t) <= max_seq_len for t in n_ids[i])
        )
    ]
    n_filtered = n_before - len(keep)
    if n_filtered:
        print(
            f"  [{label}] filtered {n_filtered} / {n_before} rows "
            f"exceeding max_seq_len={max_seq_len}"
        )
    ds = ds.select(keep)
    q_ids = [q_ids[i] for i in keep]
    p_ids = [p_ids[i] for i in keep]
    n_ids = [n_ids[i] for i in keep]
    return ds, q_ids, p_ids, n_ids


def _finalize_and_save(
    ds,
    output_path: str,
    ds_name: str,
    n_rows_raw: int,
    q_ids,
    p_ids,
    n_ids,
    q_prompts: list,
    p_prompts: list,
    neg_flat: list,
) -> int:
    """Attach token columns (if absent), sort, strip, save parquet + metadata.json.

    *q_prompts*, *p_prompts*, *neg_flat* should be the **pre-filter** prompt lists
    so that metadata stats reflect the full raw dataset.

    Token-ID columns are only added when absent, so the batch tokenization path
    (which already stores them inside *ds*) is handled transparently.

    Returns the number of rows saved.
    """
    if "query_token_ids" not in ds.column_names:
        ds = ds.add_column("query_token_ids", q_ids)
        ds = ds.add_column("positive_token_ids", p_ids)
        ds = ds.add_column("negative_token_ids", n_ids)
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
        query_prompts=q_prompts,
        positive_prompts=p_prompts,
        all_neg_flat=neg_flat,
    )
    return len(ds)


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


def _convert_f2llm_batch(batch: dict, f2llm_prompt: str) -> dict:
    """Convert F2LLM columns to the standard format, stripping prompt prefixes.

    Input columns: query, passage, negative_1 … negative_24.
    Output columns: query_text, positive_text, negative_text, query_id, positive_id.
    """
    queries = batch["query"]
    passages = batch["passage"]
    n = len(queries)
    stripped_queries, stripped_passages, stripped_negs = [], [], []
    for i in range(n):
        q = _strip_f2llm_prompt(queries[i] or "", f2llm_prompt)
        p = _strip_f2llm_prompt(passages[i] or "", f2llm_prompt)
        stripped_queries.append(q)
        stripped_passages.append(p)
        negs = []
        for j in range(1, 25):
            col = f"negative_{j}"
            if col in batch and i < len(batch[col]):
                n_text = str(batch[col][i] or "").strip()
                if n_text:
                    negs.append(_strip_f2llm_prompt(n_text, f2llm_prompt))
        stripped_negs.append(negs)
    return {
        "query_text": stripped_queries,
        "positive_text": stripped_passages,
        "negative_text": stripped_negs,
        "query_id": [str(k) for k in range(n)],
        "positive_id": [f"pos_{k}" for k in range(n)],
    }


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

    Returns the number of rows saved, or 0 if the source is skipped.
    """
    from datasets import Dataset
    from download_data import load_f2llm

    ds_name = _f2llm_source_to_name_to_task(f2llm_source)
    if ds_name not in NAME_TO_TASK_TYPE:
        warnings.warn(
            f"F2LLM source '{f2llm_source}' maps to '{ds_name}' which is not in "
            "NAME_TO_TASK. Skipping (no instruction template available).",
            UserWarning,
            stacklevel=2,
        )
        return 0

    f2llm_prompt = TASK_TO_PROMPT.get(f2llm_source)
    if f2llm_prompt is None:
        warnings.warn(
            f"F2LLM source '{f2llm_source}' not in TASK_TO_PROMPT; "
            "cannot strip F2LLM prompt template. Skipping tokenization.",
            UserWarning,
            stacklevel=2,
        )
        return 0

    task_metadata = _get_task_metadata(ds_name)
    ds = load_f2llm(sources=[f2llm_source])
    n_rows_raw = len(ds)

    # Convert F2LLM columns and strip prompt prefixes, then rebuild as HF Dataset.
    converted = _convert_f2llm_batch(
        {col: ds[col] for col in ds.column_names}, f2llm_prompt
    )
    ds = Dataset.from_dict(converted)

    ds = _build_prompts(
        ds, tokenizer, instruction_template, task_metadata, num_hard_negatives, num_proc
    )
    q_prompts: list = ds["query_prompt"]
    p_prompts: list = ds["positive_prompt"]
    neg_lists: list = ds["negative_prompts"]

    q_ids, p_ids, n_ids, neg_flat = _tokenize_dedup(
        tokenizer, add_special_tokens, ds_name, q_prompts, p_prompts, neg_lists
    )
    ds, q_ids, p_ids, n_ids = _apply_seq_len_filter(
        ds, q_ids, p_ids, n_ids, max_seq_len, ds_name
    )
    return _finalize_and_save(
        ds,
        output_path,
        ds_name,
        n_rows_raw,
        q_ids,
        p_ids,
        n_ids,
        q_prompts,
        p_prompts,
        neg_flat,
    )


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

    1. Build prompt strings for all rows (parallelised via num_proc).
    2. Collect all *unique* prompts across queries, positives and negatives.
    3. Tokenize unique prompts in ONE batched call — passages shared across
       many queries (e.g. MSMARCO corpus) are tokenized only once.
    4. Assemble per-row token-ID lists from the lookup dict in O(1).
    5. Optionally filter rows exceeding max_seq_len, sort, save.
    """
    dataset_name = os.path.basename(os.path.dirname(input_path))
    task_metadata = _get_task_metadata(dataset_name)

    ds = _load_parquet_safe(input_path)
    n_rows_raw = len(ds)

    ds = _build_prompts(
        ds, tokenizer, instruction_template, task_metadata, num_hard_negatives, num_proc
    )
    q_prompts: list[str] = ds["query_prompt"]
    p_prompts: list[str] = ds["positive_prompt"]
    neg_lists: list[list[str]] = ds["negative_prompts"]

    q_ids, p_ids, n_ids, neg_flat = _tokenize_dedup(
        tokenizer, add_special_tokens, dataset_name, q_prompts, p_prompts, neg_lists
    )
    ds, q_ids, p_ids, n_ids = _apply_seq_len_filter(
        ds, q_ids, p_ids, n_ids, max_seq_len, dataset_name
    )
    return _finalize_and_save(
        ds,
        output_path,
        ds_name,
        n_rows_raw,
        q_ids,
        p_ids,
        n_ids,
        q_prompts,
        p_prompts,
        neg_flat,
    )


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

    Prompt building and tokenization are merged in a single Dataset.map() call via
    _build_and_tokenize_hard_negatives_batch.  Best when passage reuse is low (STS,
    NLI, one-to-one pairs).  For heavy passage reuse (MSMARCO, NQ, …) prefer the
    dedup variant.
    """
    dataset_name = os.path.basename(os.path.dirname(input_path))
    task_metadata = _get_task_metadata(dataset_name)

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
    ds = ds.map(build_tok_fn, batched=True, batch_size=10000, num_proc=num_proc)

    # Capture pre-filter prompt lists for metadata, then extract token IDs for filtering.
    q_prompts: list[str] = ds["query_prompt"]
    p_prompts: list[str] = ds["positive_prompt"]
    neg_flat: list[str] = [p for row in ds["negative_prompts"] for p in row]
    q_ids: list = ds["query_token_ids"]
    p_ids: list = ds["positive_token_ids"]
    n_ids: list = ds["negative_token_ids"]

    ds, q_ids, p_ids, n_ids = _apply_seq_len_filter(
        ds, q_ids, p_ids, n_ids, max_seq_len, dataset_name
    )
    return _finalize_and_save(
        ds,
        output_path,
        ds_name,
        n_rows_raw,
        q_ids,
        p_ids,
        n_ids,
        q_prompts,
        p_prompts,
        neg_flat,
    )


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

    output_dir = os.path.join(args.output_dir, f"f2llm_{args.instruction_template}")
    print(f"F2LLM mode: {len(sources_to_process)} sources from HF cache")
    print(f"Output: {output_dir}/<source>/data.parquet")
    print("(Skipping sources not in NAME_TO_TASK)\n")

    total_rows = 0
    processed = 0
    for f2llm_source in sources_to_process:
        output_path = os.path.join(output_dir, f2llm_source, "data.parquet")
        print(f"Processing [{processed + 1}] {f2llm_source}")
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
            if n > 0:
                processed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            raise

    print(f"\nDone. Processed {processed} sources, total rows saved: {total_rows}")


def main():
    args = parse_args()

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
