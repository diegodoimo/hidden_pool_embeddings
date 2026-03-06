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
        [--data_subset retrieval/general_retrieval/arguana]
"""

import argparse
import glob
import json
import os
import warnings
from collections import Counter
from functools import partial
from typing import Optional

import pandas as pd
from transformers import AutoTokenizer

from tasks import NAME_TO_TASK_TYPE, TRANSLATE_F2LLM_NAME
from tasks.f2llm_prompts import TASK_TO_PROMPT
from tasks.retrieval_loaders import deduplicate
from utils.create_datasets import (
    FULL_TRAIN_DATA,
    TASK_TYPE_TO_TASK_METADATA,
    instruction_template_embeddinggemma,
    instruction_template_qwen3,
    _build_and_tokenize_hard_negatives_batch,
    _build_and_tokenize_hard_negatives_batch_fast,
    _build_prompts_hard_negatives_batch,
    _load_parquet_safe,
)

"""Tokenize F2LLM datasets from HF cache."""
from utils.load_f2llm_data import get_f2llm_sources, load_f2llm
from datasets import Dataset
import time

# Build a per-dataset-name → canonical inner path lookup from the full training
# data manifest.  Keys are leaf dataset names (e.g. "arguana"); values are the
# full inner path including task type and retrieval subtype when applicable
# (e.g. "retrieval/general_retrieval/arguana").  Used to construct output paths
# for F2LLM sources that have no on-disk directory hierarchy.
_DATASET_TO_INNER_PATH: dict[str, str] = {
    p.rstrip("/").split("/")[-1]: p.rstrip("/") for p in FULL_TRAIN_DATA
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
        default="results/datasets_negatives",
        help="Root directory containing hard-negative datasets (e.g. datasets_negatives)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/datasets_tokenized",
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
        default=15,
        help="Number of hard negatives per example",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Filter out rows where any component exceeds this token length. If None, no filtering.",
    )
    parser.add_argument(
        "--data_subset",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of datasets to restrict. For --f2llm: source names (e.g. arguana amazon_qa). "
        "For hard-negative datasets: inner paths (e.g. retrieval/general_retrieval/arguana). If omitted, process all.",
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
        default="batch",
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
        "--teacher_model",
        type=str,
        default="qwen3_600m",
        help="Name of the teacher-model subfolder inside --input_dir (e.g. qwen3_600m).",
    )
    args = parser.parse_args()
    return args


# -----------------------------------------


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

    # dict.fromkeys() preserves insertion order while removing duplicates, whereas set does not guarantee any order.
    # This matters here because the result (all_unique) is used to build id_map by zipping it with all_ids (the tokenizer output). The tokenizer processes the list positionally, so the order of all_unique must stay consistent between the input list and the zip. If a set were used, the pairing could be scrambled and id_map would map prompts to the wrong token IDs.
    # In short: dict.fromkeys() = dedup + stable order. set = dedup only.
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
    total_length = [
        len(q) + len(p) + sum(len(n) for n in n_list)
        for q, p, n_list in zip(q_ids, p_ids, n_ids)
    ]

    if max_seq_len is None:
        return ds, q_ids, p_ids, n_ids, total_length

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
    total_length = [total_length[i] for i in keep]

    return ds, q_ids, p_ids, n_ids, total_length


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
    total_length: list,
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
    # Overwrite any character-based total_length (from _build_prompts_hard_negatives_batch)
    # with the token-based value computed in _apply_seq_len_filter.
    if "total_length" in ds.column_names:
        ds = ds.remove_columns(["total_length"])

    ds = ds.add_column("total_length", total_length)
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


def _strip_f2llm_prompt(text: str, prompt: str) -> str:
    """Remove F2LLM prompt template from the start of text if present."""
    if not text or not prompt:
        assert False
    text = text.strip()
    # Try exact prompt prefix (with optional trailing space/newline/punctuation)
    prefix = "Instruct: " + prompt.strip() + "\nQuery:"
    if text.startswith(prefix):
        rest = text[len(prefix) :].strip()
        return rest
    else:
        assert False, text


def _convert_f2llm_batch(
    f2llm_prompt: str,
    num_hard_negatives: int,
    batch: dict,
) -> dict:
    """Step 1 for F2LLM: strip the instruct-prefix from queries, reshape negatives.

    Input columns : query, passage, negative_1 … negative_<N>  (1-indexed).
    Output columns: query_text, positive_text, negative_text
                    (negative_text is [batch_size][num_hard_negatives]).

    query_id / positive_id are NOT set here; call _assign_dedup_ids afterwards.
    """
    queries = batch["query"]
    passages = batch["passage"]
    batch_size = len(queries)

    prefix = "Instruct: " + f2llm_prompt.strip() + "\nQuery:"

    stripped_queries = [query[len(prefix) :].strip() for query in queries]
    stripped_passages = list(passages)

    # Outer loop = example index, inner loop = negative slot (1-indexed columns).
    stripped_negs = [
        [
            str(batch[f"negative_{j + 1}"][i] or "").strip()
            for j in range(num_hard_negatives)
        ]
        for i in range(batch_size)
    ]

    return {
        "query_text": stripped_queries,
        "positive_text": stripped_passages,
        "negative_text": stripped_negs,
    }


def _assign_dedup_ids(ds) -> "Dataset":
    """Step 2 for F2LLM: assign query_id and positive_id via fast pandas deduplication.

    Same-text queries/positives receive the same ID (matching the logic used
    when building hard-negative parquets for the non-F2LLM path).
    Uses ``tasks.retrieval_loaders.deduplicate`` which is backed by C-optimised
    pandas hash tables and is significantly faster than a Python loop.
    """
    q_series = pd.Series(ds["query_text"])
    p_series = pd.Series(ds["positive_text"])

    query_ids, *_ = deduplicate(q_series, prefix="query")
    positive_ids, *_ = deduplicate(p_series, prefix="positive")

    ds = ds.add_column("query_id", query_ids)
    ds = ds.add_column("positive_id", positive_ids)
    return ds


def get_f2llm_paths(subset_list):
    """Return the list of F2LLM source names to process.

    Validates that our naming convention matches the dataset, then filters out
    sources whose mapped dataset name is not in NAME_TO_TASK_TYPE (those cannot
    receive an instruction template and are silently skipped).
    """
    all_sources = get_f2llm_sources()

    # Check that our naming convention matches the dataset's known source names.
    set_all_f2llm = set(TRANSLATE_F2LLM_NAME[task] for task in all_sources)
    set_known_f2llm_prompts = set(TASK_TO_PROMPT.keys())

    assert set_all_f2llm == set_known_f2llm_prompts, set_all_f2llm.symmetric_difference(
        set_known_f2llm_prompts
    )
    if subset_list is not None:
        for task in subset_list:
            assert (
                task in set_all_f2llm
            ), f"{task} misspelled or not in the f2llm dataset"

    # Filter out sources with no instruction-template mapping; warn instead of crash.
    skipped = []
    for f2llm_source in all_sources:
        ds_name = TRANSLATE_F2LLM_NAME[
            f2llm_source
        ]  # F2LLM_SOURCE_TO_NAME_TO_TASK.get(f2llm_source, f2llm_source)
        if ds_name not in NAME_TO_TASK_TYPE:
            skipped.append(f2llm_source)
            print(
                f"  [f2llm] WARNING: source '{f2llm_source}' maps to '{ds_name}' "
                "which is not in NAME_TO_TASK_TYPE — skipping (no instruction template)."
            )
    if skipped:
        all_sources = [s for s in all_sources if s not in skipped]

    want = set(subset_list) if subset_list else None
    sources_to_process = [s for s in all_sources if want is None or s in want]
    return sources_to_process


def get_data_paths(root_folder, subset_list, teacher_model):

    root_folder = os.path.join(root_folder, teacher_model)
    if not os.path.isdir(root_folder):
        raise FileNotFoundError(f"Input directory not found: {root_folder}")
    pattern = os.path.join(root_folder, "**", "data.parquet")
    parquet_files = sorted(glob.glob(pattern, recursive=True))

    if not parquet_files:
        print(f"No data.parquet files found under {root_folder}")
        return

    if subset_list is not None:
        subset_set = set(subset_list)

        # Match by inner path (e.g. retrieval/general_retrieval/arguana) so the
        # subset applies across all dataset folders
        def _inner_path(p):
            rel = os.path.relpath(p, root_folder)
            parts = rel.split(os.sep)
            return os.path.join(*parts[1:-1]) if len(parts) > 1 else ""

        parquet_files = [p for p in parquet_files if _inner_path(p) in subset_set]
        print(
            f"Restricted to {len(parquet_files)} datasets (subset of {len(subset_list)} requested)"
        )
    return parquet_files


def main():
    args = parse_args()

    if args.instruction_template not in INSTRUCTION_TEMPLATES:
        raise ValueError(
            f"Unknown instruction_template '{args.instruction_template}'. "
            f"Choices: {list(INSTRUCTION_TEMPLATES.keys())}"
        )
    instruction_template, add_special_tokens = INSTRUCTION_TEMPLATES[
        args.instruction_template
    ]

    # use_fast=True (default): Rust fast tokenizer is 10-100× faster than the
    # pure-Python fallback.  trust_remote_code is kept for custom model repos.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    if "t5gemma-2" in args.tokenizer_path.lower():
        tokenizer_name = "t5gemma-2"
    else:
        raise ValueError("wrong model name")
    if args.num_workers > 1:
        # Prevent Rust tokenizer threads from being forked inside worker processes.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.f2llm:
        # items: list of F2LLM source-name strings (e.g. "arguana", "amazon_qa")
        items = get_f2llm_paths(subset_list=args.data_subset)
        output_folder = os.path.join(args.output_dir, "f2llm")
        middle_folder = f"qwen3_600m-teacher_{args.instruction_template}-prompt_{tokenizer_name}-tok"
    else:
        # items: list of absolute paths to data.parquet files
        items = get_data_paths(
            root_folder=args.input_dir,
            subset_list=args.data_subset,
            teacher_model=args.teacher_model,
        )
        output_folder = os.path.join(args.output_dir, "hiddengemma")
        middle_folder = f"{args.teacher_model}-teacher_{args.instruction_template}-prompt-{tokenizer_name}-tok"

    if not items:
        print("No datasets to process. Exiting.")
        return

    print(
        f"Found {len(items)} datasets to process "
        f"({'f2llm' if args.f2llm else args.teacher_model})"
    )

    total_rows = 0
    processed = 0
    for i, item in enumerate(items):
        # ------------------------------------------------------------------
        # Resolve ds_name (leaf, e.g. "arguana") and inner_path
        # (e.g. "retrieval/general_retrieval/arguana/data.parquet").
        # ------------------------------------------------------------------
        if args.f2llm:
            f2llm_source: str = item
            ds_name = TRANSLATE_F2LLM_NAME[f2llm_source]
            # inner_path is the dataset directory, e.g.
            # "retrieval/general_retrieval/arguana"
            # inner_path = _inner_path_for_ds(ds_name)

            if ds_name in _DATASET_TO_INNER_PATH:
                inner_path = _DATASET_TO_INNER_PATH[ds_name]
            else:
                task_type = NAME_TO_TASK_TYPE.get(ds_name)
                assert task_type is not None, ds_name
                task_type = task_type.lower()
                inner_path = os.path.join(task_type, ds_name)

        else:
            input_path: str = item
            ds_name = os.path.basename(os.path.dirname(input_path))
            # Strip the teacher-model prefix and the trailing "data.parquet" so
            # inner_path is the dataset directory only, e.g.
            # "retrieval/general_retrieval/arguana"
            inner_path = os.path.relpath(
                os.path.dirname(input_path),
                os.path.join(args.input_dir, args.teacher_model),
            )

        # Always write to a consistently named parquet file.
        output_path = os.path.join(
            output_folder, middle_folder, inner_path, "data.parquet"
        )
        print(f"Processing [{i + 1}/{len(items)}] {inner_path}")

        start = time.time()
        # ------------------------------------------------------------------
        # Load raw dataset.
        # ------------------------------------------------------------------
        if args.f2llm:
            f2llm_prompt = TASK_TO_PROMPT.get(ds_name)
            ds = load_f2llm(sources=[f2llm_source])
            # Step 1: strip F2LLM instruct-prefix from queries via parallel map.
            convert_fn = partial(
                _convert_f2llm_batch, f2llm_prompt, args.num_hard_negatives
            )

            ds = ds.map(
                convert_fn,
                batched=True,
                batch_size=10000,
                num_proc=args.num_workers,
                remove_columns=ds.column_names,
            )
            # Step 2: assign query_id / positive_id via fast pandas deduplication.
            # Same-text queries / positives across the whole source receive the
            # same ID, consistent with the hard-negative parquet convention.
            ds = _assign_dedup_ids(ds)
        else:
            ds = _load_parquet_safe(input_path)

        print(f"dataset loaded in {(time.time()-start)/60:.1f}min")
        start = time.time()

        # ------------------------------------------------------------------
        # Step 3: build prompts and tokenize.
        #
        # 'batch'  — build + tokenize in a single Dataset.map() pass
        #            (best when passage reuse is low).
        # 'dedup'  — build prompts first, then deduplicate across the full
        #            dataset and tokenize each unique string exactly once,
        #            remapping results back to every row.  Best for datasets
        #            with heavy passage reuse (MSMARCO, NQ, …).
        # ------------------------------------------------------------------
        task_metadata = _get_task_metadata(ds_name)
        n_rows_raw = len(ds)

        # --- Step 3a: build prompts only (no tokenization) ---
        build_fn = partial(
            _build_prompts_hard_negatives_batch,
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            task_metadata=task_metadata,
            num_hard_negatives=args.num_hard_negatives,
        )
        ds = ds.map(
            build_fn,
            batched=True,
            batch_size=10000,
            num_proc=args.num_workers,
        )

        print(f"prompts built in {(time.time()-start)/60:.1f}min")
        start = time.time()

        # Capture pre-filter prompt lists (used for metadata stats).
        q_prompts: list = ds["query_prompt"]
        p_prompts: list = ds["positive_prompt"]
        neg_prompts_lists: list = ds["negative_prompts"]
        neg_flat: list = [p for row in neg_prompts_lists for p in row]

        # --- Step 3b: dedup across the full dataset, tokenize once ---
        q_ids, p_ids, n_ids, _ = _tokenize_dedup(
            tokenizer,
            add_special_tokens,
            ds_name,
            q_prompts,
            p_prompts,
            neg_prompts_lists,
        )
        print(f"tokenization done in {(time.time()-start)/60:.1f}min")
        start = time.time()

        # Apply optional seq-len filter before saving.
        ds, q_ids, p_ids, n_ids, total_length = _apply_seq_len_filter(
            ds, q_ids, p_ids, n_ids, args.max_seq_len, ds_name
        )

        n = _finalize_and_save(
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
            total_length,
        )
        total_rows += n
        if n > 0:
            processed += 1

    print(
        f"\nDone. Processed {processed}/{len(items)} datasets, "
        f"total rows saved: {total_rows:,}"
    )


if __name__ == "__main__":
    main()
