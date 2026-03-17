#!/usr/bin/env python3
"""Annotate F2LLM parquet files with retrieval-quality signals.

Discovers all *.parquet files under --input_dir (default:
results/f2llm_data_no_instruct), validates their schema, builds
F2LLMParquetTask objects, and runs F2LLMValidator.mine_false_negatives_f2llm
to append four model-prefixed columns to each file:

    {model_name}_false_negatives  – list<struct<doc_id, relative_score>>
    {model_name}_hard_negatives   – list<string>  (top-24 non-FN docs)
    {model_name}_log_info_nce     – float32
    {model_name}_positive_rank    – int32

Expected parquet schema (written by tokenize_data.py --save_raw_data_only):
    query_text    : string
    positive_text : string
    negative_text : list<string>
    query_id      : string        ("query_<n>")
    positive_id   : string        ("doc_<n>")
    negative_id   : list<string>  ("doc_<n>")

Usage (torchrun, single node):
    torchrun --nproc_per_node=4 submit_f2llm_false_negatives.py \\
        --model_name_or_path Qwen/Qwen3-Embedding-0.6B \\
        --input_dir  results/f2llm_data_no_instruct \\
        --output_dir results/f2llm_annotated

Usage (SLURM / srun):
    srun --ntasks=$SLURM_NTASKS --gpus-per-task=1 \\
        python submit_f2llm_false_negatives.py \\
        --model_name_or_path Qwen/Qwen3-Embedding-0.6B

Optional subset:
    ... --data_subset arguana hotpotqa msmarco
"""

import glob
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer

from inference.f2llm_false_negative_mining import F2LLMValidator
from models.modules import add_pooling_layers, last_token_pool
from tasks import NAME_TO_TASK, TRANSLATE_F2LLM_NAME
from tasks.f2llm_data_loaders import make_f2llm_task
from utils.create_datasets import instruction_template_qwen3
from utils.helpers import print_memory_consumed
from tasks.helpers import validate_and_select_tasks

# ---------------------------------------------------------------------------
# Required parquet columns (from_f2llm_parquet contract)
# ---------------------------------------------------------------------------
_REQUIRED_COLUMNS = {
    "query_text",
    "positive_text",
    "negative_text",
    "query_id",
    "positive_id",
    "negative_id",
}

# Short names used for output directory / column prefix
_PATH_TO_SHORT_NAME = {
    "Qwen/Qwen3-Embedding-0.6B": "qwen3_600m",
    "Qwen/Qwen3-Embedding-8B": "qwen3_8b",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Annotate F2LLM parquet files with retrieval-quality signals."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model identifier or local path (e.g. Qwen/Qwen3-Embedding-0.6B).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/f2llm_data_no_instruct",
        help="Directory containing <source_name>.parquet files produced by "
        "tokenize_data.py --f2llm --save_raw_data_only. Default: %(default)s",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/f2llm_annotated",
        help="Directory where annotated parquets will be written. Default: %(default)s",
    )
    parser.add_argument(
        "--data_subset",
        type=str,
        nargs="*",
        default=None,
        metavar="SOURCE",
        help="Optional list of F2LLM source names to process "
        "(e.g. arguana hotpotqa msmarco). If omitted, all files are processed.",
    )
    parser.add_argument(
        "--task_names",
        type=str,
        nargs="*",
        default=None,
        metavar="SOURCE",
        help="Optional list of F2LLM source names to process "
        "(e.g. arguana hotpotqa msmarco). If omitted, all files are processed.",
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs="*",
        default=None,
        metavar="SOURCE",
        help="Optional list of F2LLM source names to process "
        "(e.g. arguana hotpotqa msmarco). If omitted, all files are processed.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum token length passed to the encoder. Default: %(default)s",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Encoding batch size per GPU. Default: %(default)s",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top corpus documents retrieved per query. Default: %(default)s",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _check_parquet_schema(parquet_path: str, rank: int) -> bool:
    """Return True iff the parquet has all columns expected by from_f2llm_parquet.

    Only rank-0 prints warnings so output is not duplicated.
    """
    import pyarrow.parquet as pq

    schema = pq.read_schema(parquet_path)
    present = set(schema.names)
    missing = _REQUIRED_COLUMNS - present
    if missing:
        if rank == 0:
            print(
                f"  [SCHEMA ERROR] {os.path.basename(parquet_path)}: "
                f"missing columns {missing} — skipping"
            )
        return False
    return True


def _verify_all_schemas(parquet_files: list[str], rank: int) -> None:
    """Print a schema report for every parquet file; raise if any is invalid.

    Called once at startup (rank-0 only) so the user sees all problems before
    any GPU work is done.
    """
    if rank != 0:
        return

    bad = []
    print("\n--- Schema verification ---")
    for path in parquet_files:
        ok = _check_parquet_schema(path, rank=0)
        stem = Path(path).stem
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {stem}")
        if not ok:
            bad.append(stem)

    if bad:
        raise RuntimeError(
            f"Schema validation failed for {len(bad)} file(s): {bad}\n"
            "All parquets must contain: " + ", ".join(sorted(_REQUIRED_COLUMNS))
        )
    print("All schemas OK.\n")


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


def build_tasks(input_dir: str, data_subset: list[str] | None, rank: int):
    """Discover parquet files, validate schemas, and build F2LLMParquetTask objects.

    Each *.parquet filename stem must be a key in TRANSLATE_F2LLM_NAME, whose
    value must exist in NAME_TO_TASK.  Files that fail either check are skipped
    with a warning (only printed on rank-0).

    Parameters
    ----------
    input_dir   : directory to scan
    data_subset : optional list of source-name stems to restrict to
    rank        : distributed rank

    Returns
    -------
    list[F2LLMParquetTask]
    """
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No *.parquet files found in '{input_dir}'")

    if rank == 0:
        print(f"Found {len(parquet_files)} parquet file(s) in {input_dir}")

    # Validate every schema before building tasks (fast — reads metadata only).
    _verify_all_schemas(parquet_files, rank)

    tasks = []
    skipped_no_translate = []
    skipped_no_task = []
    skipped_subset = []

    for path in parquet_files:
        stem = Path(path).stem

        # --- optional subset filter ---
        if data_subset is not None and stem not in data_subset:
            skipped_subset.append(stem)
            continue

        # --- F2LLM source name → registered task name ---
        ds_name = TRANSLATE_F2LLM_NAME.get(stem)
        if ds_name is None:
            skipped_no_translate.append(stem)
            if rank == 0:
                print(f"  [SKIP] '{stem}': not in TRANSLATE_F2LLM_NAME")
            continue

        if ds_name not in NAME_TO_TASK:
            skipped_no_task.append(stem)
            if rank == 0:
                print(
                    f"  [SKIP] '{stem}' → '{ds_name}': "
                    "not in NAME_TO_TASK (no registered task object)"
                )
            continue

        tasks.append(make_f2llm_task(ds_name=ds_name, parquet_path=path))

    if rank == 0:
        print(
            f"\nTasks to process : {len(tasks)}"
            + (f"\nSkipped (subset) : {len(skipped_subset)}" if skipped_subset else "")
            + (
                f"\nSkipped (no map) : {skipped_no_translate}"
                if skipped_no_translate
                else ""
            )
            + (f"\nSkipped (no task): {skipped_no_task}" if skipped_no_task else "")
        )
        if tasks:
            names = [Path(t.parquet_path).stem for t in tasks]
            print(f"Processing       : {names}")

    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    dist.init_process_group(
        "nccl",
        device_id=local_rank,
        timeout=timedelta(seconds=1800),
    )
    torch.cuda.set_device(dist.get_rank())
    torch.set_float32_matmul_precision("high")

    selected_tasks = None
    if args.task_names or args.task_types:
        selected_tasks = validate_and_select_tasks(args.task_names, args.task_types)

    # Discover and validate tasks (schema check on rank-0 before loading model).
    # tasks = build_tasks(args.input_dir, args.data_subset, rank)
    tasks = build_tasks(args.input_dir, selected_tasks, rank)
    if not tasks:
        if rank == 0:
            print("No tasks to process. Exiting.")
        dist.destroy_process_group()
        sys.exit(0)

    # Short name used as column prefix and output sub-directory.
    model_short = _PATH_TO_SHORT_NAME.get(
        args.model_name_or_path,
        # Fallback: sanitise the HF id into a safe identifier.
        args.model_name_or_path.replace("/", "__").replace(".", "_"),
    )

    if rank == 0:
        print(f"\nModel            : {args.model_name_or_path}  (short: {model_short})")
        print(f"Input dir        : {args.input_dir}")
        print(f"Output dir       : {args.output_dir}")
        print(f"top_k            : {args.top_k}")
        print(f"batch_size       : {args.batch_size}")
        print(f"max_length       : {args.max_length}")

    # -----------------------------------------------------------------------
    # Load tokenizer + model
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to("cuda")

    max_length = min(args.max_length, model.config.max_position_embeddings)

    # -----------------------------------------------------------------------
    # Build validator
    # -----------------------------------------------------------------------
    validator = F2LLMValidator(
        path=args.output_dir,  # used only to mkdir on rank-0
        model_name=model_short,
        task_names=None,  # not used by F2LLMValidator
        tokenizer=tokenizer,
        instruction_template=instruction_template_qwen3,
        padding_side="right",
        max_length=max_length,
        add_special_tokens=False,
        eot_id=tokenizer.pad_token_id,
    )

    dist.barrier()
    if rank == 0:
        print("Tokenizer and model loaded")

    model = model.eval()
    model = torch.compile(model)
    model = add_pooling_layers(model, pool_fn=last_token_pool)

    dist.barrier()
    if rank == 0:
        print("Model compiled and pooling layers attached")
        print_memory_consumed()

    # -----------------------------------------------------------------------
    # Run annotation
    # -----------------------------------------------------------------------
    validator.mine_false_negatives_f2llm(
        tasks=tasks,
        model=model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
