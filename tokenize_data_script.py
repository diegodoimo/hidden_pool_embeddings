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
import os
from functools import partial
from typing import Optional

from transformers import AutoTokenizer

from tasks import NAME_TO_TASK_TYPE
from utils.create_datasets import (
    TASK_TYPE_TO_TASK_METADATA,
    instruction_template_embeddinggemma,
    instruction_template_qwen3,
    _build_and_tokenize_hard_negatives_batch,
    _filter_long_hard_negatives_batch,
    _load_parquet_safe,
)

INSTRUCTION_TEMPLATES = {
    "qwen3": (instruction_template_qwen3, False),
    "embeddinggemma": (instruction_template_embeddinggemma, True),
    "gemma3_prompt": (instruction_template_embeddinggemma, True),
}


def get_instruction_template(name: str):
    """Return (template_fn, add_special_tokens) for the given template name."""
    if name not in INSTRUCTION_TEMPLATES:
        raise ValueError(
            f"Unknown instruction_template '{name}'. "
            f"Choices: {list(INSTRUCTION_TEMPLATES.keys())}"
        )
    return INSTRUCTION_TEMPLATES[name]


def tokenize_and_save_dataset(
    input_path: str,
    output_path: str,
    ds_name: str,
    tokenizer,
    instruction_template,
    add_special_tokens: bool,
    num_hard_negatives: int,
    max_seq_len: Optional[int],
) -> int:
    """Load, tokenize, and save a single parquet dataset. Returns number of rows saved."""
    # dataset_name is the leaf folder (e.g. arguana) used for NAME_TO_TASK_TYPE
    dataset_name = os.path.basename(os.path.dirname(input_path))
    if dataset_name not in NAME_TO_TASK_TYPE:
        raise ValueError(
            f"Dataset '{dataset_name}' not in NAME_TO_TASK_TYPE. "
            "Cannot determine task metadata."
        )

    task_type = NAME_TO_TASK_TYPE[dataset_name]
    task_metadata = TASK_TYPE_TO_TASK_METADATA[task_type]

    ds = _load_parquet_safe(input_path)

    build_tok_fn = partial(
        _build_and_tokenize_hard_negatives_batch,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        task_metadata=task_metadata,
        num_hard_negatives=num_hard_negatives,
        add_special_tokens=add_special_tokens,
    )
    ds = ds.map(build_tok_fn, batched=True, batch_size=10000)

    if max_seq_len is not None:
        n_before = len(ds)
        filter_fn = partial(
            _filter_long_hard_negatives_batch,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        ds = ds.filter(filter_fn, batched=True, batch_size=1000)
        print(f"  [{dataset_name}] filtered {n_before - len(ds)} / {n_before} rows exceeding max_seq_len={max_seq_len}")

    if "dataset_name" not in ds.column_names:
        ds = ds.add_column("dataset_name", [ds_name] * len(ds))
    ds = ds.sort("total_length", reverse=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_parquet(output_path)
    return len(ds)


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
        default="gemma3_prompt",
        choices=list(INSTRUCTION_TEMPLATES.keys()),
        help="Instruction template: qwen3, embeddinggemma, or gemma3_prompt",
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
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    instruction_template, add_special_tokens = get_instruction_template(
        args.instruction_template
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, use_fast=False, trust_remote_code=True
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
        parquet_files = [
            p for p in parquet_files
            if _inner_path(p) in subset_set
        ]
        print(f"Restricted to {len(parquet_files)} datasets (subset of {len(args.datasets_subset)} requested)")

    print(f"Found {len(parquet_files)} datasets under {args.input_dir}")
    print(f"Output: {args.output_dir}/<dataset_folder>_{args.instruction_template}/")
    print()

    total_rows = 0
    for i, input_path in enumerate(parquet_files):
        rel_path = os.path.relpath(input_path, args.input_dir)
        # rel_path: e.g. qwen3_600m_data/retrieval/general_retrieval/arguana/data.parquet
        # or: qwen3_600m/retrieval/general_retrieval/arguana/data.parquet
        parts = rel_path.split(os.sep)
        dataset_folder = parts[0]
        inner_path = os.path.join(*parts[1:])  # retrieval/general_retrieval/arguana/data.parquet

        output_folder = f"{dataset_folder}_{args.instruction_template}"
        output_path = os.path.join(args.output_dir, output_folder, inner_path)

        print(f"Processing [{i + 1}/{len(parquet_files)}] {rel_path}")
        ds_name = os.path.dirname(inner_path)  # e.g. retrieval/general_retrieval/arguana
        try:
            n = tokenize_and_save_dataset(
                input_path=input_path,
                output_path=output_path,
                ds_name=ds_name,
                tokenizer=tokenizer,
                instruction_template=instruction_template,
                add_special_tokens=add_special_tokens,
                num_hard_negatives=args.num_hard_negatives,
                max_seq_len=args.max_seq_len,
            )
            total_rows += n
        except Exception as e:
            print(f"  ERROR: {e}")
            raise

    print(f"\nDone. Total rows saved: {total_rows}")


if __name__ == "__main__":
    main()
