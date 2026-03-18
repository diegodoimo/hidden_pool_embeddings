#!/usr/bin/env python3
"""Tokenize F2LLM-annotated parquet files for training with doc-ID columns.

Reads *.parquet files from --input_dir (default: results/f2llm_annotated/)
that were produced by find_f2llm_false_negatives.py, and writes tokenized
parquets to --output_dir.

Expected input schema per parquet:
    query_text              : string   (F2LLM query stripped of instruct prefix)
    positive_text           : string
    negative_text           : list<string>
    query_id                : string   ("query_<n>")
    positive_id             : string   ("doc_<n>")
    negative_id             : list<string>
    {model_prefix}_hard_negatives : list<string>  (up to 24 hard-neg doc_ids)

Output columns:
    query_input_ids         – tokenized "Instruct: …\\nQuery: …" prompt
                              (identical to tokenize_data_qwen.py output)
    passage_input_ids       – tokenized positive passage (same)
    negative_input_ids      – list of tokenized hard-negative passages
                              (all in one column; replaces negative_1_input_ids …)
    positive_doc_id         – positive_text  (raw passage, no template)
    negative_doc_id         – list of hard-negative passage texts
    dataset_name            – our internal dataset name (e.g. "arguana")
    query_id, positive_id   – passed through from the source parquet

The tokenization of query_input_ids and passage_input_ids matches
f2llm_repro/tokenize_data_qwen.py exactly:
    • Qwen3 / causal LM  – add_special_tokens=False, append EOS manually,
                           reserve 1 slot so max token count = max_seq_len.
    • T5-Gemma2           – add_special_tokens=True (tokenizer adds BOS/EOS),
                           no manual EOS, full max_seq_len slots available.

Usage:
    python tokenize_f2llm_annotated.py \\
        --tokenizer_path Qwen/Qwen3-Embedding-0.6B \\
        --model_prefix   qwen3_600m \\
        [--input_dir  results/f2llm_annotated] \\
        [--output_dir results/f2llm_annotated_tokenized] \\
        [--max_seq_len 1024] \\
        [--min_hard_negatives 7] \\
        [--num_workers 8] \\
        [--data_subset arguana hotpotqa msmarco]
"""

from multiprocessing import Pool
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from tasks import TRANSLATE_F2LLM_NAME
from tasks.f2llm_prompts import TASK_TO_PROMPT

# ── globals initialised by main() before any worker fork ─────────────────────
tokenizer = None
max_seq_length = None
_add_special_tokens = None
_append_eos = None

# ── model-type presets (mirrors tokenize_data_qwen.py / tokenize_data.py) ────
MODEL_TYPE_PRESETS = {
    "qwen3": {"add_special_tokens": False, "append_eos": True, "seq_len_reserve": 1},
    "t5-gemma2": {"add_special_tokens": True, "append_eos": False, "seq_len_reserve": 0},
}


def _infer_model_type(tokenizer_path: str) -> str:
    p = tokenizer_path.lower()
    if "t5gemma" in p or "t5-gemma" in p:
        return "t5-gemma2"
    if "qwen" in p:
        return "qwen3"
    raise ValueError(
        f"Cannot infer model type from '{tokenizer_path}'. "
        f"Path must contain 'qwen' or 't5gemma'. "
        f"Known types: {list(MODEL_TYPE_PRESETS)}"
    )


# ── per-sentence tokenization (runs inside worker processes) ─────────────────

def process_sent(sentence):
    ids = tokenizer(
        sentence,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=_add_special_tokens,
    )["input_ids"]
    if _append_eos:
        ids = ids + [tokenizer.eos_token_id]
    return np.array(ids, dtype=object)


def process_sent_batch(s: pd.Series) -> pd.Series:
    return s.apply(process_sent)


def parallelize(data: pd.Series, func, num_of_processes: int) -> pd.Series:
    if num_of_processes <= 1:
        return func(data)
    chunks = np.array_split(data.index, num_of_processes)
    splits = [data.loc[idx] for idx in chunks if len(idx)]
    with Pool(num_of_processes) as pool:
        result = pd.concat(pool.map(func, splits))
    return result


# ── corpus helper ─────────────────────────────────────────────────────────────

def _build_corpus_map(df: pd.DataFrame) -> dict:
    """Return {doc_id: passage_text} from positive_id/negative_id columns."""
    corpus: dict = {}
    for pid, ptxt in zip(df["positive_id"], df["positive_text"]):
        corpus[pid] = ptxt
    for nid_list, ntxt_list in zip(df["negative_id"], df["negative_text"]):
        for nid, ntxt in zip(nid_list, ntxt_list):
            corpus[nid] = ntxt
    return corpus


# ── main processing loop ──────────────────────────────────────────────────────

def process_dataset(
    parquet_path: str,
    output_path: str,
    ds_name: str,
    task_prompt: str,
    model_prefix: str,
    min_hard_negatives: int,
    num_workers: int,
) -> int:
    """Tokenize one annotated parquet and write to *output_path*.

    Returns the number of rows written (0 if skipped).
    """
    df = pd.read_parquet(parquet_path)

    hard_neg_col = f"{model_prefix}_hard_negatives"
    if hard_neg_col not in df.columns:
        print(f"  [SKIP] column '{hard_neg_col}' not found — skipping {ds_name}")
        return 0

    # ── 1. Filter rows with too few hard negatives ────────────────────────────
    n_before = len(df)
    df = df[df[hard_neg_col].apply(len) >= min_hard_negatives].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [{ds_name}] dropped {n_dropped}/{n_before} rows "
              f"(fewer than {min_hard_negatives} hard negatives)")
    if df.empty:
        print(f"  [{ds_name}] no rows remaining after filter — skipping")
        return 0

    # ── 2. Build corpus doc_id → text mapping ─────────────────────────────────
    corpus_map = _build_corpus_map(df)

    # ── 3. Reconstruct F2LLM query prompts ────────────────────────────────────
    # This reproduces the format that was originally in the `query` column of
    # the raw F2LLM dataset (before tokenize_data_qwen.py tokenised it).
    query_prompts = pd.Series(
        [f"Instruct: {task_prompt}\nQuery: {q}" for q in df["query_text"]],
        index=df.index,
    )

    # ── 4. Collect all unique passage texts (dedup before tokenisation) ────────
    # Hard-negative texts are looked up from the corpus map.
    all_passage_texts: list[str] = list(df["positive_text"])
    for hn_ids in df[hard_neg_col]:
        for hid in hn_ids[:min_hard_negatives]:      # only what we'll actually use
            text = corpus_map.get(hid)
            if text is not None:
                all_passage_texts.append(text)
    # Include ALL hard-negative slots so they are available at train time
    for hn_ids in df[hard_neg_col]:
        for hid in hn_ids:
            text = corpus_map.get(hid)
            if text is not None:
                all_passage_texts.append(text)

    unique_texts = list(dict.fromkeys(all_passage_texts))   # dedup, stable order
    print(f"  [{ds_name}] tokenizing {len(query_prompts):,} queries "
          f"and {len(unique_texts):,} unique passage texts …")

    df_tok = pd.DataFrame({"text": unique_texts})
    df_tok["input_ids"] = parallelize(df_tok["text"], process_sent_batch, num_workers)
    df_tok = df_tok.set_index("text")

    # ── 5. Tokenize queries ───────────────────────────────────────────────────
    query_input_ids = parallelize(query_prompts, process_sent_batch, num_workers).tolist()

    # ── 6. Map passage texts to token ids ─────────────────────────────────────
    passage_input_ids = df["positive_text"].map(df_tok["input_ids"]).tolist()

    # ── 7. Build negative_input_ids and negative_doc_id per row ───────────────
    negative_input_ids: list[list] = []
    negative_doc_id: list[list[str]] = []

    for hn_ids in df[hard_neg_col]:
        row_ids: list = []
        row_texts: list[str] = []
        for hid in hn_ids:
            text = corpus_map.get(hid)
            if text is None or text not in df_tok.index:
                continue
            row_ids.append(df_tok.loc[text, "input_ids"].tolist())
            row_texts.append(text)
        negative_input_ids.append(row_ids)
        negative_doc_id.append(row_texts)

    # ── 8. Build positive_doc_id (raw passage text, no template) ─────────────
    positive_doc_id: list[str] = df["positive_text"].tolist()

    # ── 9. Assemble output parquet ────────────────────────────────────────────
    n_rows = len(df)
    table = pa.table(
        {
            "query_input_ids": pa.array(query_input_ids),
            "passage_input_ids": pa.array(passage_input_ids),
            "negative_input_ids": pa.array(negative_input_ids),
            "positive_doc_id": pa.array(positive_doc_id, type=pa.large_utf8()),
            "negative_doc_id": pa.array(
                negative_doc_id, type=pa.list_(pa.large_utf8())
            ),
            "dataset_name": pa.array(
                [ds_name] * n_rows, type=pa.large_utf8()
            ),
            "query_id": pa.array(df["query_id"].tolist(), type=pa.large_utf8()),
            "positive_id": pa.array(df["positive_id"].tolist(), type=pa.large_utf8()),
        }
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    return n_rows


def main(args):

    parquet_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".parquet")
    )
    if args.data_subset:
        subset = set(args.data_subset)
        parquet_files = [f for f in parquet_files if f.replace(".parquet", "") in subset]
        print(f"Restricted to {len(parquet_files)} datasets "
              f"(subset of {len(args.data_subset)} requested)")

    print(f"Found {len(parquet_files)} parquet file(s) in {args.input_dir}")

    total_rows = 0
    processed = 0

    for parquet_file in tqdm(parquet_files):
        stem = parquet_file.replace(".parquet", "")

        # stem is our internal name; TRANSLATE_F2LLM_NAME maps it to the
        # F2LLM source name which is the key in TASK_TO_PROMPT.
        f2llm_source = TRANSLATE_F2LLM_NAME.get(stem)
        if f2llm_source is None:
            print(f"  [SKIP] '{stem}': not in TRANSLATE_F2LLM_NAME")
            continue

        task_prompt = TASK_TO_PROMPT.get(f2llm_source)
        if task_prompt is None:
            print(f"  [SKIP] '{stem}' → '{f2llm_source}': no entry in TASK_TO_PROMPT")
            continue

        ds_name = stem   # use our internal name as dataset_name
        output_path = os.path.join(args.output_dir, parquet_file)

        if os.path.isfile(output_path) and not args.force_recompute:
            print(f"  [SKIP] {ds_name}: output already exists")
            continue

        print(f"\nProcessing {ds_name}  (f2llm_source={f2llm_source})")
        n = process_dataset(
            parquet_path=os.path.join(args.input_dir, parquet_file),
            output_path=output_path,
            ds_name=ds_name,
            task_prompt=task_prompt,
            model_prefix=args.model_prefix,
            min_hard_negatives=args.min_hard_negatives,
            num_workers=args.num_workers,
        )
        total_rows += n
        if n > 0:
            processed += 1
        print(f"  → {n:,} rows written to {output_path}")

    print(f"\nDone. Processed {processed}/{len(parquet_files)} datasets, "
          f"total rows saved: {total_rows:,}")


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(
            description=(
                "Tokenize F2LLM-annotated parquets for training with "
                "false-negative / hard-negative doc-ID masking."
            )
        )
        parser.add_argument(
            "--input_dir",
            type=str,
            default="results/f2llm_annotated",
            help="Directory of annotated *.parquet files (default: %(default)s)",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="results/f2llm_annotated_tokenized",
            help="Output directory for tokenized parquets (default: %(default)s)",
        )
        parser.add_argument(
            "--tokenizer_path",
            type=str,
            required=True,
            help="HuggingFace model path for the tokenizer "
                 "(e.g. Qwen/Qwen3-Embedding-0.6B)",
        )
        parser.add_argument(
            "--model_prefix",
            type=str,
            required=True,
            help="Column prefix for hard-negative annotations "
                 "(e.g. 'qwen3_600m' → reads 'qwen3_600m_hard_negatives')",
        )
        parser.add_argument(
            "--max_seq_len",
            type=int,
            default=1024,
            help="Maximum sequence length cap (default: %(default)s). "
                 "Sequences are truncated to this length; for Qwen3 one slot "
                 "is reserved for the appended EOS token.",
        )
        parser.add_argument(
            "--min_hard_negatives",
            type=int,
            default=7,
            help="Minimum number of hard negatives required per row; "
                 "rows with fewer are dropped (default: %(default)s).",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="Parallel tokenisation workers (default: %(default)s). "
                 "Set to 1 to disable multiprocessing.",
        )
        parser.add_argument(
            "--data_subset",
            type=str,
            nargs="*",
            default=None,
            metavar="DATASET",
            help="Optional list of dataset stems to process "
                 "(e.g. arguana hotpotqa msmarco). "
                 "If omitted, all *.parquet files are processed.",
        )
        parser.add_argument(
            "--force_recompute",
            action="store_true",
            help="Re-tokenize even when the output file already exists.",
        )
        return parser.parse_args()

    args = parse_args()

    # ── configure globals before forking workers ──────────────────────────────
    model_type = _infer_model_type(args.tokenizer_path)
    preset = MODEL_TYPE_PRESETS[model_type]
    _add_special_tokens = preset["add_special_tokens"]
    _append_eos = preset["append_eos"]
    max_seq_length = args.max_seq_len - preset["seq_len_reserve"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    print(
        f"Tokenizer  : {args.tokenizer_path}\n"
        f"Model type : {model_type}\n"
        f"add_special_tokens={_add_special_tokens}, "
        f"append_eos={_append_eos}, "
        f"max_seq_length={max_seq_length} (cap {args.max_seq_len})\n"
        f"model_prefix: {args.model_prefix} "
        f"→ column '{args.model_prefix}_hard_negatives'\n"
        f"min_hard_negatives: {args.min_hard_negatives}\n"
    )

    main(args)
